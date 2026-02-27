import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from src.text_rec.OpenOCR.tools.engine.config import Config
from src.text_rec.OpenOCR.openrec.optimizer import build_optimizer
from src.text_rec.OpenOCR.openrec.preprocess import transform
from src.text_rec.OpenOCR.openrec.preprocess import create_operators
from src.text_rec.OpenOCR.openrec.modeling.base_recognizer import BaseRecognizer
from src.text_rec.OpenOCR.openrec.modeling import build_model
from src.text_rec.OpenOCR.openrec.postprocess import build_post_process
from src.text_rec.OpenOCR.openrec.losses import build_loss
from ratio_sample import RatioSampler
from dataset import SignboardText
from utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
import argparse
import numpy as np
import copy
import warnings
import os
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

def get_state_dict(state_dict): 
    new_state_dict = OrderedDict()
    for p, v in state_dict.items(): 
        key = p.split('.')
        if 'encoder' in key: 
            new_state_dict[p] = v
        if 'decoder' in key:
            if 'embedding' in p or 'ques1_head' in p:
                print(f"[SKIP] {p}") 
                continue 
            else:
                new_state_dict[p] = v
                    
    return new_state_dict

class SMTRFinetuner(pl.LightningModule):
    def __init__(self, model, criterion=None, cfg_optimizer=None, cfg_lr_scheduler=None, train_dataloader=None, val_dataloader=None,
                 metrics_interval=100, postprocess=None, max_epochs=100):
        super(SMTRFinetuner, self).__init__()
        self.model = model
        self.criterion = criterion
        self.cfg_optimizer = cfg_optimizer
        self.cfg_lr_scheduler = cfg_lr_scheduler
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.max_epochs = max_epochs

        # Postprocess outputs
        self.postprocess = postprocess

        self.train_predictions, self.train_gt = [], []
        self.val_predictions, self.val_gt = [], []
        self.val_loss_dict = {'val_loss': [], 'val_ctc_loss': []}

    def forward(self, x, data=None): 
        if self.model.training: 
            preds = self.model(x, data=data)
        else: 
            preds = self.model(x)

        return preds
        
    def compute_loss(self, batch, batch_idx): 
        batch_tensor = [t.to(self.device) for t in batch['inputs']]
        batch_numpy = [t.cpu().numpy() for t in batch['inputs']]

        preds = self(batch_tensor[0], batch_tensor[1:])
        if self.model.training: 
            loss_dict = self.criterion(preds, batch_tensor)
        else: 
            loss_dict = None

        post_results = self.postprocess(preds, batch_numpy)
        predictions = np.array([pred_trans for pred_trans, _ in post_results[0]]).astype(str)
        gt = np.array([gt_trans for gt_trans, _ in post_results[1]]).astype(str)

        return loss_dict, predictions, gt

    def compute_accuracy(self, predictions, gt): 
        # Calculate case-sensitive accuracy 
        accuracy_strict = np.sum(predictions == gt) / len(gt)
        # Calculate case-insensitive accuracy
        predictions_lower = np.char.lower(predictions)
        gt_lower = np.char.lower(gt)
        accuracy_lenient = np.sum(predictions_lower == gt_lower) / len(gt_lower)

        return accuracy_strict, accuracy_lenient

    def on_train_epoch_start(self):
        """Ensure training mode"""
        self.model.train()
    
    def on_validation_epoch_start(self):
        """Ensure eval mode"""
        self.model.eval()
            
    def training_step(self, batch, batch_idx):
        batch_size = batch['inputs'][0].shape[0] # get batch_size
        loss_dict, predictions, gt = self.compute_loss(batch, batch_idx)

        self.train_predictions.extend(predictions)
        self.train_gt.extend(gt)

        if batch_idx % self.metrics_interval == 0:
            accuracy_strict, accuracy_lenient = self.compute_accuracy(predictions=np.array(self.train_predictions),
                                                                     gt=np.array(self.train_gt))
            self.train_predictions.clear()
            self.train_gt.clear()
            metrics = {'train_loss': loss_dict['loss'], 'train_gtc_loss': loss_dict['loss'],
                      'train_acc_strict': accuracy_strict, 'train_acc_lenient': accuracy_lenient}
            
            for k,v in metrics.items():
                self.log(k, v)

            return {'loss': loss_dict['loss'], 
                   'train_acc_strict': accuracy_strict,
                   'train_acc_lenient': accuracy_lenient}

        else: 
            return {'loss': loss_dict['loss']}
        
    def validation_step(self, batch, batch_idx):
        batch_size = batch['inputs'][0].shape[0] # get batch_size
        _, predictions, gt = self.compute_loss(batch, batch_idx)

        self.val_predictions.extend(predictions)
        self.val_gt.extend(gt)
            
        accuracy_strict, accuracy_lenient = self.compute_accuracy(predictions=np.array(predictions),
                                                                 gt=np.array(gt))

        return {'val_acc_strict': accuracy_strict, 
               'val_acc_lenient': accuracy_lenient} 

    def on_validation_epoch_end(self):
        accuracy_strict, accuracy_lenient = self.compute_accuracy(predictions=np.array(self.val_predictions),
                                                                     gt=np.array(self.val_gt))
        self.val_predictions.clear()
        self.val_gt.clear()
        
        metrics = {'val_acc_strict': accuracy_strict, 'val_acc_lenient': accuracy_lenient}
        
        for k,v in metrics.items():
            self.log(k, v)

        return {'val_acc_strict': accuracy_strict, 
               'val_acc_lenient': accuracy_lenient}

    def configure_optimizers(self):
        optimizer, lr_scheduler = build_optimizer(self.cfg_optimizer, self.cfg_lr_scheduler, 
                                                 epochs=self.max_epochs, step_each_epoch=len(self.train_dl),
                                                 model=self.model)
        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }]

    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
def train(config, train_ops, val_ops, pretrained_model_path, max_epochs, device='cpu', seed=42):
    set_seed(seed)
    # initialize train and val dataset 
    train_dataset = SignboardText(rootDir='signboardtext/word_images/train',      
                                  label='signboardtext/word_images/text_image.json',
                                  transforms=transform,
                                  ops=train_ops,
                                  ds_width=True)
    val_dataset = SignboardText(rootDir='signboardtext/word_images/val',
                                label='signboardtext/word_images/text_image.json',
                                transforms=transform,
                                ops=val_ops,
                                ds_width=True)
    # create dataloader
    train_sampler = RatioSampler(data_source=train_dataset, scales=[[128, 32]],
                                 first_bs=32, fix_bs=False, divided_factor=[4,8], 
                                 is_training=True, seed=seed)
    val_sampler = RatioSampler(data_source=val_dataset, scales=[[128, 32]],
                               first_bs=16, fix_bs=False, divided_factor=[4,8], 
                               is_training=False, seed=seed)
    train_dataloader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler)
    val_dataloader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler)
    # initialize model, postprocess and loss
    post_process_class = build_post_process(config['PostProcess'], 
                                            config['Global'])
    config['Architecture']['Decoder']['out_channels'] = post_process_class.get_character_num()
    model = BaseRecognizer(config['Architecture'])
    criterion = build_loss(config['Loss'])
    # Load state dict from pretrained model
    checkpoint = torch.load(pretrained_model_path, map_location=device)
    new_state_dict = get_state_dict(checkpoint['state_dict'])
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)

    smtr_finetuner = SMTRFinetuner(model=model, criterion=criterion, cfg_optimizer=config['Optimizer'], 
                                   cfg_lr_scheduler=config['LRScheduler'],
                                   train_dataloader=train_dataloader, 
                                   val_dataloader=val_dataloader, max_epochs=max_epochs, postprocess=post_process_class)

    wandb_logger = WandbLogger(project='Text Recognition', name='smtr')

    early_stop_callback = EarlyStopping(
        monitor="val_acc_lenient", 
        min_delta=0.00, 
        patience=5, 
        verbose=True, 
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(file_name='best_model-{epoch:02d}',
                                          save_top_k=1, monitor="val_acc_lenient", 
                                          mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=device,
        devices=1,  # number of devices (e.g., GPUs if using CUDA)
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.1
    )
    # start training
    trainer.fit(smtr_finetuner)

def test(config, val_ops, best_model_path, device='cpu', seed=42): 
    set_seed(seed)
    # initialize test dataset
    test_dataset = SignboardText(rootDir='signboardtext/word_images/test',
                                 label='signboardtext/word_images/text_image.json',
                                 transforms=transform,
                                 ops=val_ops, 
                                 ds_width=True)
    # create dataloader
    test_sampler = RatioSampler(data_source=test_dataset, scales=[[128, 32]],
                                first_bs=16, fix_bs=False, divided_factor=[4,8], 
                                is_training=False, seed=seed)
    test_dataloader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler)
    # initialize model 
    post_process_class = build_post_process(config['PostProcess'], 
                                            config['Global'])
    config['Architecture']['Decoder']['out_channels'] = post_process_class.get_character_num()
    model = BaseRecognizer(config['Architecture'])
    # Load state dict best model
    best_model = SMTRFinetuner.load_from_checkpoint(best_model_path,
                                                    model=model, postprocess=post_process_class)
    best_model = best_model.to(device)
    # start testing
    best_model.eval()
    predictions, gt = [], []
    corpus_predictions = {'vietsignboard': [], 'english': [], 'vin': []}
    corpus_gt = {'vietsignboard': [], 'english': [], 'vin': []}
    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            batch_tensor = [t.to(device) for t in batch['inputs']]
            batch_numpy = [t.cpu().numpy() for t in batch['inputs']]
            list_image = batch['file_name']
            preds = best_model(batch_tensor[0])
            post_results = post_process_class(preds, batch_numpy)
            for pred_result, gt_result, image_name in zip(post_results[0], post_results[1], list_image):
                pred_trans, _ = pred_result
                gt_trans, _ = gt_result
                corpus_name = str(Path(image_name).parent)
                corpus_predictions[corpus_name].append(pred_trans)
                corpus_gt[corpus_name].append(gt_trans)
                predictions.append(pred_trans)
                gt.append(gt_trans)
    # evaluate each corpus
    for k in corpus_gt.keys(): 
        accuracy_strict, accuracy_lenient = best_model.compute_accuracy(np.array(corpus_predictions[k]), 
                                                                        np.array(corpus_gt[k]))
        print(f'Accuracy in case-sensitve for {k} corpus: {accuracy_strict:.4f}')
        print(f'Accuracy in case-insensitive for {k} corpus: {accuracy_lenient:.4f}')


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train', required=True, help='')
    parser.add_argument('-best_model_path', type=str, default=None, required=False, help='')
    parser.add_argument('-pretrained_model_path', type=str, default=None, required=False, help='')
    parser.add_argument('-epochs', type=int, default=100, required=False, help='')
    # parser.add_argument('-threshold', type=float, default=0.5, required=False, help='')
    args = parser.parse_args()

    # initialize config
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    config = copy.deepcopy(Config('src/text_rec/OpenOCR/focalsvtr_smtr.yml').cfg)
    config['Global']['character_dict_path'] = 'charset/vietnamese_charset.txt'
    config['PostProcess']['character_dict_path'] = 'charset/vietnamese_charset.txt'
    config['Train']['dataset']['transforms'][2]['SMTRLabelEncode']['character_dict_path'] = 'charset/vietnamese_charset.txt'
    config['Eval']['dataset']['transforms'][1]['ARLabelEncode']['character_dict_path'] = 'charset/vietnamese_charset.txt'
    config['Train']['dataset']['transforms'][2]['SMTRLabelEncode']['max_text_length'] = 25
    config['Eval']['dataset']['transforms'][1]['ARLabelEncode']['max_text_length'] = 25
    config['Global']['max_text_length'] = 25
    config['Architecture']['Decoder']['max_len'] = 25

    # Replace augmentation
    config['Train']['dataset']['transforms'].pop(1)
    new_aug = {'PARSeqAugPIL': {}}
    config['Train']['dataset']['transforms'].insert(1, new_aug)

    train_ops = create_operators(config['Train']['dataset']['transforms'][1:], config['Global'])
    val_ops = create_operators(config['Eval']['dataset']['transforms'][1:], config['Global'])

    if args.mode == 'train': 
        pretrained_model_path = args.pretrained_model_path
        max_epochs = args.max_epochs
        assert pretrained_model_path is not None
        train(config, train_ops, val_ops, pretrained_model_path=pretrained_model_path, 
              max_epochs=max_epochs, device=device)
    
    elif args.mode == 'val': 
        best_model_path = args.best_model_path
        assert best_model_path is not None
        test(config, val_ops, best_model_path=best_model_path, device=device)