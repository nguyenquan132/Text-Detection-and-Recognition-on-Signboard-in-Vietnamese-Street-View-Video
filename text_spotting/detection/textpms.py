import os
import torch
import pytorch_lightning as pl 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from text_spotting.detection.TextPMs.util.config import config, update_config
from text_spotting.detection.TextPMs.util.option import BaseOptions
from text_spotting.detection.TextPMs.util.augmentation import BaseTransform, Augmentation
from text_spotting.detection.TextPMs.util.detection import watershed_segment
from text_spotting.detection.TextPMs.util.misc import data_transfer_ICDAR
from text_spotting.detection.TextPMs.network.textnet import TextNet
from text_spotting.detection.TextPMs.network.loss import TextLoss
from .dataset import SignboardText
import argparse
from torch.utils.data import DataLoader
from utils.utility import set_seed, save_output_txt, prepare_file_to_evaluate
from utils.transforms import rescale_box
import numpy as np
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

def collate_fn(batch): 
    result = {
    'file_name': [b['file_name'] for b in batch],
    'image_size': [b['image_size'] for b in batch],
    'image': torch.stack([b['image'] for b in batch]),
    'polygons': [b['polygons'] for b in batch],
    'transcriptions': [b['transcriptions'] for b in batch]
    }
    
    if all('tr_mask' in b for b in batch):
        result['tr_mask'] = torch.stack([b['tr_mask'] for b in batch])
    
    if all('train_mask' in b for b in batch):
        result['train_mask'] = torch.stack([b['train_mask'] for b in batch])
    
    return result

class TextPMsFinetuner(pl.LightningModule):
    def __init__(self, model, criterion=None, lr=None, train_dataloader=None, val_dataloader=None,
                 metrics_interval=100, postprocess=None):
        super(TextPMsFinetuner, self).__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

        # Postprocess outputs
        self.postprocess = postprocess

    def forward(self, img):
        outputs, *_ = self.model(img)
        return outputs

    def predict_one_image(self, image, cfg=None, orig_size=None, threshold=0.5): 
        assert len(image.shape) == 4 #(B, C, H, W)
        b, c, h, w = image.shape
        cfg.threshold = threshold
        with torch.inference_mode(): 
            preds = self(image)
            preds = torch.sigmoid(preds[0, :, :h//cfg.scale, :w//cfg.scale])
            preds = preds.cpu().numpy()
    
            labels, _, contours = self.postprocess(preds, cfg, cfg.scale)
            boxes = data_transfer_ICDAR(contours)
            boxes = rescale_box(image, orig_size, boxes=boxes)

        return boxes
        
    def compute_loss(self, batch, batch_idx): 
        train_mask, tr_mask = batch['train_mask'], batch['tr_mask']
        img = batch['image']
        output = self(img)
        loss = self.criterion(output, train_mask, tr_mask)

        return loss
            
    def training_step(self, batch, batch_idx):
        batch_size = batch['image'].shape[0] # get batch_size
        loss = self.compute_loss(batch, batch_idx)

        if batch_idx % self.metrics_interval == 0:
            self.log('train_loss', loss, batch_size=batch_size)

        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_size = batch['image'].shape[0] # get batch_size
        loss = self.compute_loss(batch, batch_idx)

        self.log('val_loss', loss, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
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
    
def train(config, train_transforms, val_transforms, pretrained_model_path, max_epochs, 
          device='cpu', seed=42): 
    set_seed(seed)
    # initialize train and val dataset
    train_dataset = SignboardText(rootDir='signboardtext/images/train',      
                                    label='signboardtext/annotations.json',
                                    transforms=train_transforms,
                                    scale=config.scale, 
                                    mask_cnt=len(config.fuc_k),
                                    alpha=config.fuc_k)
    val_dataset = SignboardText(rootDir='signboardtext/images/val',
                                label='signboardtext/annotations.json',
                                transforms=val_transforms,
                                scale=config.scale, 
                                mask_cnt=len(config.fuc_k),
                                alpha=config.fuc_k)
    # create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, 
                                  shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=1, 
                                shuffle=False, collate_fn=collate_fn)
    
    # initialize model and loss
    model = TextNet(is_training=True, backbone=config.net)
    criterion = TextLoss()
    # Load state dict from pretrained model
    checkpoint = torch.load(pretrained_model_path, 
                            map_location=device)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(config.device)
    
    textpms_finetuner = TextPMsFinetuner(model=model, criterion=criterion, lr=config.lr,
                                         train_dataloader=train_dataloader, 
                                         val_dataloader=val_dataloader)

    wandb_logger = WandbLogger(project='Text Detection', name='TextPMs')

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
        devices=1,  # number of devices (e.g., GPUs if using CUDA)
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.1
    )
    # start training
    trainer.fit(textpms_finetuner)

def test(config, val_transforms, best_model_path, threshold, device='cpu', seed=42): 
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
    model = TextNet(is_training=False, backbone=config.net)
    model = model.to(device)
    # Load state dict best model
    best_model = TextPMsFinetuner.load_from_checkpoint(best_model_path,
                                                       model=model,
                                                       postprocess=watershed_segment)
    # Start testing
    best_model.eval()
    save_folder = 'Prediction'
    for data in tqdm(test_dataloader): 
        file_name, image = data['file_name'][0], data['image'].to(device)
        polygons = data['polygons'][0]
        orig_size = data['image_size'][0]
        transcriptions = data['transcriptions'][0]
        boxes = best_model.predict_one_image(image, cfg=config, 
                                             orig_size=orig_size, threshold=threshold)

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

    # initialize transformation
    train_transforms = Augmentation(size=config.input_size, mean=config.means, std=config.stds)
    val_transforms = BaseTransform(size=config.test_size, mean=config.means, std=config.stds)

    if args.mode == 'train': 
        pretrained_model_path = args.pretrained_model_path
        max_epochs = args.epochs
        assert pretrained_model_path is not None
        train(config, train_transforms, val_transforms, 
              pretrained_model_path=pretrained_model_path, max_epochs=max_epochs, device=device)
    
    elif args.mode == 'val': 
        best_model_path = args.best_model_path
        threshold=args.threshold
        assert best_model_path is not None
        test(config, val_transforms, best_model_path=best_model_path, 
             threshold=threshold, device=device)
