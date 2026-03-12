"""
The PolynomialDecayLR scheduler is adapted from PaddleOCR and reimplemented in PyTorch.
"""
import os
import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from TextPMs.util.config import config, update_config
from TextPMs.util.option import BaseOptions
from TextPMs.util.augmentation import BaseTransform, Augmentation
from DBNet.modeling.architectures.base_model import BaseModel
from DBNet.postprocess.db_postprocess import DBPostProcess
from DBNet.loss.db_loss import DBLoss
from DBNet.data.make_border_map import MakeBorderMap
from DBNet.data.make_shrink_map import MakeShrinkMap
import argparse
from torch.utils.data import DataLoader
import math
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from .dataset import SignboardText
from utils.utility import set_seed, save_output_txt, prepare_file_to_evaluate, parse_yaml
from utils.transforms import rescale_box
import numpy as np
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

class PolynomialDecayLR(_LRScheduler):
    r"""
    Applies polynomial decay to the initial learning rate.
    
    PyTorch equivalent of PaddlePaddle's PolynomialDecay scheduler.

    The algorithm can be described as following.

    If cycle is set to True, then:

    .. math::

        decay\_steps & = decay\_steps * math.ceil(\frac{epoch}{decay\_steps})

        new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\frac{epoch}{decay\_steps})^{power}+end\_lr

    If cycle is set to False, then:

    .. math::

        epoch & = min(epoch, decay\_steps)

        new\_learning\_rate & = (learning\_rate-end\_lr)*(1-\frac{epoch}{decay\_steps})^{power}+end\_lr

    Args:
        optimizer (Optimizer): Wrapped PyTorch optimizer.
        decay_steps (int): The decay step size. It determines the decay cycle. 
            It must be a positive integer.
        end_lr (float, optional): The minimum final learning rate. Default: 0.0001.
        power (float, optional): Power of polynomial, should greater than 0.0 to 
            get learning rate decay. Default: 1.0.
        cycle (bool, optional): Whether the learning rate rises again. If True, 
            then the learning rate will rise when it decrease to ``end_lr``. 
            If False, the learning rate is monotone decreasing. Default: False.
        last_epoch (int, optional): The index of last epoch. Can be set to restart 
            training. Default: -1, means initial learning rate.
        verbose (bool, optional): If ``True``, prints a message to stdout for 
            each update. Default: ``False``.

    Returns:
        ``PolynomialDecayLR`` instance to schedule learning rate.

    Examples:

        .. code-block:: python

            import torch
            import torch.nn as nn
            import torch.optim as optim

            # Create model and optimizer
            model = nn.Linear(10, 10)
            optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
            
            # Create scheduler
            scheduler = PolynomialDecayLR(
                optimizer, 
                decay_steps=20, 
                end_lr=0.0001,
                power=1.0,
                cycle=False,
                verbose=True
            )
            
            # Training loop
            for epoch in range(20):
                for batch_id in range(5):
                    # Forward pass
                    x = torch.randn(10, 10)
                    out = model(x)
                    loss = out.mean()
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update learning rate each step
                    scheduler.step()
                    
                # Or update learning rate each epoch
                # scheduler.step()
    """

    def __init__(
        self,
        optimizer,
        decay_steps,
        end_lr=0.0001,
        power=1.0,
        cycle=False,
        last_epoch=-1,
        verbose=False,
    ):
        assert decay_steps > 0 and isinstance(
            decay_steps, int
        ), "'decay_steps' must be a positive integer."
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        assert (
            power > 0.0
        ), "'power' must be greater than 0.0 so that the learning rate will decay."
        self.power = power
        self.cycle = cycle
        
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        Calculate learning rate for each parameter group.
        
        Returns:
            list: Learning rates for each parameter group.
        """
        tmp_epoch_num = self.last_epoch
        tmp_decay_steps = self.decay_steps
        
        if self.cycle:
            div_res = math.ceil(
                float(self.last_epoch) / float(self.decay_steps)
            )
            
            if self.last_epoch == 0:
                div_res = 1
            tmp_decay_steps = self.decay_steps * div_res
        else:
            tmp_epoch_num = min(self.last_epoch, self.decay_steps)
        
        # Calculate new learning rate for each parameter group
        return [
            (base_lr - self.end_lr) * (
                (1 - float(tmp_epoch_num) / float(tmp_decay_steps)) ** self.power
            ) + self.end_lr
            for base_lr in self.base_lrs
        ]

def collate_fn(batch): 
    result = {
    'file_name': [b['file_name'] for b in batch],
    'image_size': [b['image_size'] for b in batch],
    'image': torch.stack([b['image'] for b in batch]),
    'polygons': [b['polygons'] for b in batch],
    'transcriptions': [b['transcriptions'] for b in batch]
    }
    
    if all('threshold_map' in b for b in batch):
        result['threshold_map'] = torch.stack([b['threshold_map'] for b in batch])

    if all('threshold_mask' in b for b in batch):
        result['threshold_mask'] = torch.stack([b['threshold_mask'] for b in batch])
    
    if all('shrink_map' in b for b in batch):
        result['shrink_map'] = torch.stack([b['shrink_map'] for b in batch])

    if all('shrink_mask' in b for b in batch):
        result['shrink_mask'] = torch.stack([b['shrink_mask'] for b in batch])
    
    return result

class DBNetFinetuner(pl.LightningModule):
    def __init__(self, model, criterion=None, lr=None, momentum=None, weight_decay=None, factor=None,
                 train_dataloader=None, val_dataloader=None,
                 metrics_interval=100, postprocess=None):
        super(DBNetFinetuner, self).__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.factor = factor
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

        # Postprocess outputs
        self.postprocess = postprocess

    def forward(self, img):
        outputs = self.model(img)
        return outputs

    def predict_one_image(self, image, shape_list=None, threshold=0.5): 
        assert len(image.shape) == 4 #(B, C, H, W)
        self.postprocess.threshold = threshold
        with torch.inference_mode(): 
            output = self(image)
            preds = dict(
                maps = output['maps'].cpu().numpy()
            )
            results = self.postprocess(preds, shape_list)[0]
            boxes = results['points']

        return boxes
        
    def compute_loss(self, batch, batch_idx): 
        img = batch['image']
        output = self(img)
        loss = self.criterion(output['maps'], batch)

        return loss
            
    def training_step(self, batch, batch_idx):
        batch_size = batch['image'].shape[0] # get batch_size
        loss = self.compute_loss(batch, batch_idx)

        if batch_idx % self.metrics_interval == 0:
            self.log('train_loss', loss['loss'], batch_size=batch_size)
            for k, v in loss.items():
                if k != 'loss': 
                    self.log(f'train_{k}', v , batch_size=batch_size)

        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_size = batch['image'].shape[0] # get batch_size
        loss = self.compute_loss(batch, batch_idx)

        self.log('val_loss', loss['loss'], batch_size=batch_size)
        for k, v in loss.items():
            if k != 'loss':
                self.log(f'val_{k}', v, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                   weight_decay=self.weight_decay)
        lr_scheduler = PolynomialDecayLR(optimizer, decay_steps=1000, end_lr=0, power=self.factor)
        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'frequency': 1
        }]
        
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
def train(config_model, train_transforms, val_transforms, make_border_map, make_shrink_map,
          pretrained_model_path, max_epochs, device='cpu', seed=42): 
    set_seed(seed)
    # initialize train and val dataset
    train_dataset = SignboardText(rootDir='signboardtext/images/train',      
                                  label='signboardtext/annotations.json',
                                  transforms=train_transforms,
                                  make_border_map=make_border_map,
                                  make_shrink_map=make_shrink_map)
    val_dataset = SignboardText(rootDir='signboardtext/images/val',
                                label='signboardtext/annotations.json',
                                transforms=val_transforms,
                                make_border_map=make_border_map,
                                make_shrink_map=make_shrink_map)
    # create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, 
                                  shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, 
                                shuffle=False, collate_fn=collate_fn)
    
    # initialize model and loss
    model = BaseModel(config_model['Architecture'])
    criterion = DBLoss(alpha=config_model['Loss']['alpha'], beta=config_model['Loss']['beta'],
                       ohem_ratio=config_model['Loss']['ohem_ratio'])
    # Load state dict from pretrained model
    checkpoint_state_dict = torch.load(pretrained_model_path, map_location=device)
    model.load_state_dict(checkpoint_state_dict, strict=True)
    model = model.to(device)
    
    dbnet_finetuner = DBNetFinetuner(model=model, criterion=criterion, lr=config_model['Optimizer']['lr']['learning_rate'],
                                    momentum=config_model['Optimizer']['momentum'], weight_decay=config_model['Optimizer']['weight_decay'],
                                    factor=config_model['Optimizer']['lr']['factor'],
                                    train_dataloader=train_dataloader, val_dataloader=val_dataloader)

    wandb_logger = WandbLogger(project='Text Detection', name='DBNet++')

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=True, 
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(file_name='best_model-{epoch:02d}',
                                          save_top_k=1, monitor="val_loss", 
                                          mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=device,
        devices=1, # number of devices (e.g., GPUs if using CUDA)
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.1
    )
    # start training
    trainer.fit(dbnet_finetuner)

def test(config_model, val_transforms, best_model_path, threshold, device='cpu', seed=42): 
    set_seed(seed)
    # initialize test dataset
    test_dataset = SignboardText(rootDir='signboardtext/images/test',
                                 label='signboardtext/annotations.json',
                                 transforms=val_transforms,
                                 is_training=False)
    # create dataloader
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, 
                                 shuffle=False, collate_fn=collate_fn)
    
    # initialize model 
    model = BaseModel(config_model['Architecture'])
    model = model.to(device)
    postprocess = DBPostProcess(thresh=config_model['PostProcess']['thresh'], box_thresh=0.5,
                                max_candidates=config_model['PostProcess']['max_candidates'], 
                                unclip_ratio=config_model['PostProcess']['unclip_ratio'])
    # Load state dict best model
    best_model = DBNetFinetuner.load_from_checkpoint(best_model_path,
                                                     model=model,
                                                     postprocess=postprocess)
    # Start testing
    best_model.eval()
    save_folder = 'Prediction'
    for data in tqdm(test_dataloader): 
        file_name, image = data['file_name'][0], data['image'].to(device)
        polygons = data['polygons'][0]
        orig_size = data['image_size'][0]
        transcriptions = data['transcriptions'][0]

        orig_h, orig_w = data['image_size'][0]
        size_h, size_w = data['image'].shape[2:]
        
        shape_list = [float(orig_h), float(orig_w), float(size_h / orig_h), float(size_w / orig_w)]
        shape_list = np.expand_dims(shape_list, axis=0)
        boxes = best_model.predict_one_image(image, shape_list=shape_list, threshold=threshold)

        save_output_txt(file_name, polygons, save_folder=save_folder, boxes=boxes)
    
    # zip file each corpus
    for corpus in os.listdir(save_folder): 
        print(f'------------{corpus}--------------')
        corpus_path = os.path.join(save_folder, corpus)
        file_zip = os.path.join(corpus_path, 'result.zip')
        prepare_file_to_evaluate(file_zip, corpus_path)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train', required=False, help='Operation mode: "train" for training the model, "test" for testing the model')
    parser.add_argument('-best_model_path', type=str, default=None, required=False, help='Path to the best trained model checkpoint to load for testing')
    parser.add_argument('-pretrained_model_path', type=str, default=None, required=False, help='Path to the pretrained trained model checkpoint to load for training')
    parser.add_argument('-epochs', type=int, default=100, required=False, help='Number of training epochs')
    parser.add_argument('-threshold', type=float, default=0.5, required=False, help='Confidence threshold used to filter detected text regions')
    args = parser.parse_args()

    # initialize config
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    option = BaseOptions()
    args_cfg = option.parser.parse_known_args()
    update_config(config, args_cfg[0])
    config.device = device
    if device == "cpu": 
        config.cuda = False

    config_model = parse_yaml('src/text_det/PaddleOCR2Pytorch/configs/det/det_r50_db++_icdar15.yml')
    # initialize transformation
    train_transforms = Augmentation(size=config.input_size, mean=config.means, std=config.stds)
    val_transforms = BaseTransform(size=config.test_size, mean=config.means, std=config.stds)
    make_border_map = MakeBorderMap()
    make_shrink_map = MakeShrinkMap()

    if args.mode == 'train': 
        pretrained_model_path = args.pretrained_model_path
        max_epochs = args.epochs
        assert pretrained_model_path is not None
        train(config_model, train_transforms, val_transforms, make_border_map, make_shrink_map,
              pretrained_model_path=pretrained_model_path, max_epochs=max_epochs, device=device)
    
    elif args.mode == 'val': 
        best_model_path = args.best_model_path
        threshold=args.threshold
        assert best_model_path is not None
        test(config_model, val_transforms, best_model_path=best_model_path, threshold=threshold, 
             device=device)
