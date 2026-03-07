import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, get_cosine_schedule_with_warmup
from .dataset import SignboardSegmentation
from utils.utility import set_seed
from PIL import Image
import evaluate 
import numpy as np
import re
import albumentations as A
import argparse
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

def param_groups(model, base_lr=6e-5, base_weight_decay=0.01):
    param_dict = {
        'pos_block': {'params': [], 'lr': base_lr, 'weight_decay': 0.                
        },
        'norm': {'params': [], 'lr': base_lr, 'weight_decay': 0.                
        },
        'head': {'params': [], 'lr': base_lr * 10., 'weight_decay': base_weight_decay
        },
        'default': {'params': [], 'lr': base_lr, 'weight_decay': base_weight_decay 
        }
    }
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'pos_block' in name:
            param_dict['pos_block']['params'].append(param)
        elif re.search(r'\.norm|norm\.|layer_norm|batch_norm', name.lower()):
            param_dict['norm']['params'].append(param)
        elif 'decode_head' in name:
            param_dict['head']['params'].append(param)
        else:
            param_dict['default']['params'].append(param)
    
    param_groups = [v for v in param_dict.values() if v['params']]
    
    return param_groups

class SegformerFinetuner(pl.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100, max_epochs=100):
        super(SegformerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", 
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.max_epochs = max_epochs
        
    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)
    
    def training_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
            
            metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}
            
            for k,v in metrics.items():
                self.log(k,v)
            
            return(metrics)
        else:
            return({'loss': loss})
    
    def validation_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        self.validation_step_outputs.append(loss)
        
        return({'val_loss': loss})
    
    def on_validation_epoch_end(self):
        metrics = self.val_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
        
        avg_val_loss = torch.stack(self.validation_step_outputs).mean()
        self.validation_step_outputs.clear() # free memory
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]
        
        metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
        for k,v in metrics.items():
            self.log(k,v)

        return metrics
    
    def test_step(self, batch, batch_nb):
        
        images, masks = batch['pixel_values'], batch['labels']
        
        outputs = self(images, masks)
        
        loss, logits = outputs[0], outputs[1]
        
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=masks.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        self.test_step_outputs.append(loss) 
            
        return({'test_loss': loss})
    
    def on_test_epoch_end(self):
        metrics = self.test_mean_iou.compute(
              num_labels=self.num_classes, 
              ignore_index=255, 
              reduce_labels=False,
          )
       
        avg_test_loss = torch.stack(self.test_step_outputs).mean()
        self.test_step_outputs.clear() # free memory 
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}
        
        for k,v in metrics.items():
            self.log(k,v)
        
        return metrics
    
    def configure_optimizers(self):
        params_dict = param_groups(self.model, base_lr=6e-5, base_weight_decay=0.01)
        optimizer = torch.optim.AdamW(params_dict, betas=(0.9, 0.999))
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.train_dl) * self.max_epochs,
            num_cycles=0.5 
        )
        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }]
    
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
    def test_dataloader(self):
        return self.test_dl
    
def train(train_transforms, val_transforms, feature_extractor, id2label, 
          max_epochs, device='cpu', seed=42):
    set_seed(seed)
    # initialize train and val dataset
    train_dataset = SignboardSegmentation(rootDir='signboardtext/images/train',
                                          label='signboardtext/annotations_signboard.json',
                                          feature_extractor=feature_extractor,
                                          transforms=train_transforms)
    val_dataset = SignboardSegmentation(rootDir='signboardtext/images/val',
                                        label='signboardtext/annotations_signboard.json',
                                        feature_extractor=feature_extractor,
                                        transforms=val_transforms)
    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=3, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, num_workers=3, shuffle=False)

    segformer_finetuner = SegformerFinetuner(
        id2label=id2label, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        metrics_interval=10,
        max_epochs=max_epochs
    )

    wandb_logger = WandbLogger(project='Signboard_Segmentation', name='segformer')

    early_stop_callback = EarlyStopping(
        monitor="val_mean_iou", 
        min_delta=0.00, 
        patience=5, 
        verbose=True, 
        mode="max",
    )

    checkpoint_callback = ModelCheckpoint(file_name='best_model-{epoch:02d}', 
                                          save_top_k=1, monitor="val_mean_iou", 
                                          mode='max')
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
    trainer.fit(segformer_finetuner) 

def test(val_transforms, feature_extractor, best_model_path, id2label, device='cpu', seed=42):
    set_seed(seed)
    # initialize test dataset
    test_dataset = SignboardSegmentation(rootDir='signboardtext/images/test',
                                         label='signboardtext/annotations_signboard.json',
                                         feature_extractor=feature_extractor,
                                         transforms=val_transforms)
    # create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=3, shuffle=False)

    segformer_finetuner = SegformerFinetuner(
        id2label=id2label, 
        test_dataloader=test_dataloader
    )
    # start testing
    wandb_logger = WandbLogger(project='Signboard_Segmentation', name='segformer')
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=device,
        devices=1, # number of devices (e.g., GPUs if using CUDA)
        inference_mode=True
    )
    trainer.test(segformer_finetuner, ckpt_path=best_model_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train', required=False, help='Operation mode: "train" for training the model, "test" for testing, or "infer" for inference')
    parser.add_argument('-best_model_path', type=str, default=None, required=False, help='Path to the best trained model checkpoint to load for testing or inference')
    parser.add_argument('-epochs', type=int, default=100, required=False, help='Number of training epochs')
    parser.add_argument('-size', type=int, default=640, required=False, help='Input image size (images will be resized to size x size)')
    parser.add_argument('-image_path', type=str, default=None, required=False, help='Path to a single image for inference')
    args = parser.parse_args()

    # initialize object
    id2label = {0: 'background', 1: 'signboard'}
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # initilize transformation
    mean=np.array([123.675, 116.28, 103.53]) / 255
    std=np.array([58.395, 57.12, 57.375]) / 255
    size = args.size
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    feature_extractor.reduce_labels = False
    feature_extractor.do_resize = False
    feature_extractor.do_rescale = False
    feature_extractor.do_normalize = False 

    train_transforms = A.Compose([
            A.Resize(size, size, always_apply=True),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.25),
            A.Rotate(limit=25),
            A.Normalize(mean=mean, std=std)
        ], is_check_shapes=False)

    val_transforms = A.Compose([
            A.Resize(size, size, always_apply=True),
            A.Normalize(mean=mean, std=std)
        ], is_check_shapes=False)
    
    if args.mode == 'train': 
        max_epochs = args.epochs
        train(train_transforms, val_transforms, feature_extractor=feature_extractor, 
              id2label=id2label, max_epochs=max_epochs, device=device)
    elif args.mode == 'test':
        best_model_path = args.best_model_path
        assert best_model_path is not None
        test(val_transforms, feature_extractor=feature_extractor, 
             best_model_path=best_model_path, id2label=id2label, device=device)
    