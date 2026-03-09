import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor, get_cosine_schedule_with_warmup
from .dataset import SignboardSegmentation
from utils.visualize import overlay
from utils.utility import set_seed
import albumentations as A
import evaluate 
import numpy as np
import cv2
import re
import argparse
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

def create_mask(mask_labels, class_labels):
    """
    Args: 
        mask_labels: List(torch.Tensor(batch_size, num_classes, height, width)) 
        class_labels: List(torch.Tensor(batch_size, num_classes))
    Returns: 
        Batch numpy mask with shape: List((batch_size, height, width)) 
    """
    batch_size = len(mask_labels)
    H, W = mask_labels[0].shape[1:]
    batch_mask = []

    for mask_label, class_label in zip(mask_labels, class_labels):
        mask_label = mask_label.detach().cpu().numpy() 
        class_label = class_label.detach().cpu().numpy()

        mask = np.zeros((H, W), dtype=np.uint8)
        for i, cls in enumerate(class_label): 
            mask[mask_label[i] == 1] = cls.item()

        batch_mask.append(mask)

    return batch_mask

def param_groups(model, base_lr=1e-4, base_weight_decay=0.05):
    param_dict = {
        'encoder': {'params': [], 'lr': base_lr * 0.1, 'weight_decay': base_weight_decay               
        },
        'norm': {'params': [], 'lr': base_lr, 'weight_decay': 0.                
        },
        'pos': {'params': [], 'lr': base_lr, 'weight_decay': 0.
        },
        'embed': {'params': [], 'lr': base_lr, 'weight_decay': 0.
        },
        'default': {'params': [], 'lr': base_lr, 'weight_decay': base_weight_decay 
        }
    }

    embedding_patterns = [
        'embed',  # word_embedding, token_embedding
        'pos_embed',  # position embedding
        'cls_token',  
        'mask_token',  # mask token trong MAE
    ]
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'encoder' in name:
            param_dict['encoder']['params'].append(param)
        elif re.search(r'\.norm|norm\.|layer_norm|batch_norm', name.lower()):
            param_dict['norm']['params'].append(param)
        elif 'relative_position_bias_table' in name or 'absolute_pos_embed' in name:
            param_dict['head']['params'].append(param)
        elif any(pattern in name.lower() for pattern in embedding_patterns):
            print(name)
            param_dict['embed']['params'].append(param)
        else:
            param_dict['default']['params'].append(param)
    
    param_groups = [v for v in param_dict.values() if v['params']]
    
    return param_groups

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([b['pixel_values'] for b in batch]),
        'pixel_mask': torch.stack([b['pixel_mask'] for b in batch]),
        'mask_labels': [b['mask_labels'][0] for b in batch], 
        'class_labels': [b['class_labels'][0] for b in batch]
    }

class Mask2FormerFinetuner(pl.LightningModule):
    
    def __init__(self, id2label, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100, max_epochs=50):
        super(Mask2FormerFinetuner, self).__init__()
        self.id2label = id2label
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader
        self.max_epochs = max_epochs
        
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-tiny-cityscapes-semantic", 
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
        
    def forward(self, images, pixel_mask=None, mask_labels=None, class_labels=None):
        if mask_labels is not None: 
            mask_labels = [mask_label.to(self.device) for mask_label in mask_labels]
        if class_labels is not None: 
            class_labels = [class_label.to(self.device) for class_label in class_labels]
        outputs = self.model(pixel_values=images,
                             mask_labels=mask_labels,
                             class_labels=class_labels,
                             pixel_mask=pixel_mask)
        return(outputs)
    
    def predict_one_image(self, images, target_size): 
        assert len(target_size) == 2
        images = images.to(self.device)
        with torch.inference_mode(): 
            outputs = self.model(images)

            class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
            masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
            
            # Remove the null class `[..., :-1]
            masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
            masks_probs = masks_queries_logits.sigmoid()
            
            # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
            segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
            
            upsampled_logits = nn.functional.interpolate(
                segmentation, 
                size=target_size, 
                mode="bilinear", 
                align_corners=False
            )
            
            predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()

        return predicted_mask
    
    def training_step(self, batch, batch_nb):
        
        images, pixel_mask = batch['pixel_values'], batch['pixel_mask']
        mask_labels, class_labels = batch['mask_labels'], batch['class_labels']
        
        outputs = self(images, pixel_mask, mask_labels, class_labels)

        loss = outputs.loss
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        
        # Remove the null class `[..., :-1]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        
        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        
        upsampled_logits = nn.functional.interpolate(
            segmentation, 
            size=images.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        masks = create_mask(mask_labels, class_labels)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=np.array(masks)
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
        
        images, pixel_mask = batch['pixel_values'], batch['pixel_mask']
        mask_labels, class_labels = batch['mask_labels'], batch['class_labels']
        
        outputs = self(images, pixel_mask, mask_labels, class_labels)

        loss = outputs.loss
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        
        # Remove the null class `[..., :-1]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        
        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        
        upsampled_logits = nn.functional.interpolate(
            segmentation, 
            size=images.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        masks = create_mask(mask_labels, class_labels)
        
        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=np.array(masks)
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
        
        images, pixel_mask = batch['pixel_values'], batch['pixel_mask']
        mask_labels, class_labels = batch['mask_labels'], batch['class_labels']
        
        outputs = self(images, pixel_mask, mask_labels, class_labels)

        loss = outputs.loss 
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]
        
        # Remove the null class `[..., :-1]
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()
        
        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        
        upsampled_logits = nn.functional.interpolate(
            segmentation, 
            size=images.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        predicted = upsampled_logits.argmax(dim=1)
        masks = create_mask(mask_labels, class_labels)
        
        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=np.array(masks)
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
        params_dict = param_groups(self.model, base_lr=1e-4, base_weight_decay=0.05)
        optimizer = torch.optim.AdamW(params_dict)
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
    train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=3, shuffle=True, 
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=2, num_workers=3, shuffle=False,
                                collate_fn=collate_fn)

    mask2former_finetuner = Mask2FormerFinetuner(
        id2label=id2label, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader, 
        metrics_interval=10,
        max_epochs=max_epochs
    )

    wandb_logger = WandbLogger(project='Signboard_Segmentation', name='mask2former')

    early_stop_callback = EarlyStopping(
        monitor="val_mean_iou", 
        min_delta=0.00, 
        patience=5, 
        verbose=False, 
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
    trainer.fit(mask2former_finetuner) 

def test(val_transforms, feature_extractor, best_model_path, id2label, device='cpu', seed=42):
    set_seed(seed)
    # initialize test dataset
    test_dataset = SignboardSegmentation(rootDir='signboardtext/images/test',
                                         label='signboardtext/annotations_signboard.json',
                                         feature_extractor=feature_extractor,
                                         transforms=val_transforms)
    # create dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=2, num_workers=3, shuffle=False, 
                                 collate_fn=collate_fn)

    mask2former_finetuner = Mask2FormerFinetuner(
        id2label=id2label, 
        test_dataloader=test_dataloader, 
    )
    # start testing
    wandb_logger = WandbLogger(project='Signboard_Segmentation', name='mask2former')
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=device,
        devices=1, # number of devices (e.g., GPUs if using CUDA)
        inference_mode=True
    )
    trainer.test(mask2former_finetuner, ckpt_path=best_model_path)

def inference(model, image_path, transforms, feature_extractor, device='cpu', seed=42):
    set_seed(seed)
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # preprocess image
    pixel_values = transforms(image=image)['image']
    encoded_inputs = feature_extractor(pixel_values, return_tensors='pt')
    for k, v in encoded_inputs.items(): 
        encoded_inputs[k].squeeze_() 

    pixel_values = encoded_inputs['pixel_values']
    # Prediction
    model.eval() 
    model = model.to(device)
    predicted_mask = model.predict_one_image(pixel_values.unsqueeze(dim=0),
                                                target_size=image.shape[:2])
    show_image = overlay(image, predicted_mask.squeeze(), (0, 255, 0), alpha=0.7)

    # Save annotated image
    cv2.imwrite('image_prediction.jpg', show_image)

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
    feature_extractor = Mask2FormerImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-cityscapes-semantic")
    feature_extractor.do_reduce_labels = False
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
    elif args.mode == 'infer':
        best_model_path = args.best_model_path
        image_path = args.image_path
        assert best_model_path is not None
        assert image_path is not None
        # initialize postprocess and load best model
        best_model = Mask2FormerFinetuner.load_from_checkpoint(best_model_path,
                                                               id2label=id2label)
        inference(best_model, image_path, val_transforms, feature_extractor, device=device)
    