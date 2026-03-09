import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import DetrForObjectDetection, DetrImageProcessor
import signboard_det.DETR.datasets.transforms as T
from signboard_det.DETR.util.misc import collate_fn
from signboard_det.DETR.postprocess import PostProcess
from .dataset import SignboardDetection
from .metrics import bounding_boxes, calculate_mAP
from signboard_det.evaluation.BoundingBoxes import BoundingBoxes
from signboard_det.evaluation.Evaluator import *
from utils.visualize import draw_bounding_box
from utils.utility import set_seed, calculate_font_scale
from utils.transforms import filter_bboxes, scale_boxes
import albumentations as A
from PIL import Image
import numpy as np
import cv2
import argparse
from tqdm import tqdm
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

class Transforms(object):
    def __init__(self, image_set, size):
        self.image_set = image_set
        self.transforms = T.Compose([
            T.RandomResize([size], max_size=1333),
            T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ])
        
        self.aug = A.Compose(
            [
                A.Perspective(p=0.1),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=['labels'], 
                                     clip=True, min_area=64, min_width=1, min_height=1),
        )
    def __call__(self, image, target): 
        assert isinstance(image, np.ndarray)
        if self.image_set == 'train':
            aug_data = self.aug(image=image, bboxes=target['boxes'].cpu().numpy(), labels=target['labels'].cpu().numpy())
            aug_image, aug_boxes, aug_labels = aug_data['image'], aug_data['bboxes'], aug_data['labels']
            target['boxes'] = torch.tensor(aug_boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(aug_labels, dtype=torch.int64)
            area = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in aug_boxes]
            target['area'] = torch.tensor(area, dtype=torch.float32)
            image, target = self.transforms(Image.fromarray(aug_image), target)

        if self.image_set == 'val': 
            image, target = self.transforms(Image.fromarray(image), target)

        return image, target

class DETRFinetuner(pl.LightningModule):
    def __init__(self, id2label, label2id, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, lr_drops=200,
                 train_dataloader=None, val_dataloader=None,
                 metrics_interval=100, postprocess=None):
        super(DETRFinetuner, self).__init__()
        self.id2label = id2label
        self.label2id = label2id
        self.num_classes = len(id2label.keys())
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50',
                                                            num_labels = self.num_classes,
                                                            id2label=self.id2label,
                                                            label2id=self.label2id,
                                                            ignore_mismatched_sizes=True)
        # optimizers
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.lr_drops = lr_drops
        
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

        # Postprocess outputs
        self.postprocess = postprocess

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        outputs = self.model(pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels)
        return outputs

    def predict_one_image(self, pixel_values, pixel_mask=None, 
                          original_size=None, threshold=0.25): 
        assert len(pixel_values.shape) == 4 #(B, C, H, W)
        if pixel_mask is None:
            pixel_mask = torch.ones((1, pixel_values.shape[-2], pixel_values.shape[-1]))
        with torch.inference_mode(): 
            outputs = self(pixel_values=pixel_values.to(self.device), 
                           pixel_mask=pixel_mask.to(self.device))

        # Postprocessing
        target_size = torch.tensor(pixel_values.shape[2:]).to(self.device) # Resized image
        output = {k: v for k, v in outputs.items()}
        output['pred_logits'] = output.pop('logits')
        results = self.postprocess(output, target_size.unsqueeze(dim=0))[0]

        filtered_result = filter_bboxes(results, threshold)
        if original_size is not None: 
            current_size = pixel_values.shape[2:]
            filtered_result['bboxes'] = scale_boxes(current_size, filtered_result['bboxes'], target_size)

        return filtered_result
        
    def compute_loss(self, batch, batch_idx): 
        pixel_values, pixel_mask = batch[0].decompose() 
        pixel_mask = (~pixel_mask).int()  
        labels = [{key: label[key].to(self.device) for key in label.keys()} for label in batch[1]]

        outputs = self(pixel_values, pixel_mask, labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict
    def training_step(self, batch, batch_idx):
        batch_size = len(batch[1]) # len(target)
        loss, loss_dict = self.compute_loss(batch, batch_idx)

        if batch_idx % self.metrics_interval == 0:
            self.log('train_loss', loss, batch_size=batch_size)
            for k, v in loss_dict.items(): 
                self.log(f'train_{k}', v, batch_size=batch_size)

        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch[1]) # len(target)
        loss, loss_dict = self.compute_loss(batch, batch_idx)

        self.log('val_loss', loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            self.log(f'val_{k}', v, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                      weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_drops)

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }]
        
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
def train(train_transforms, val_transforms, label2id, id2label, max_epochs=100, device='cpu', seed=42): 
    set_seed(seed=seed)
    # initialize train and val dataset
    train_dataset = SignboardDetection(rootDir='signboardtext/images/train',
                                       label='signboardtext/annotations_signboard.json',
                                       label2id=label2id, 
                                       transforms=train_transforms)
    val_dataset = SignboardDetection(rootDir='signboardtext/images/val',
                                     label='signboardtext/annotations_signboard.json',
                                     label2id =label2id,
                                     transforms=val_transforms)
    # create dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False,
                                collate_fn=collate_fn)
    
    detr_finetuner = DETRFinetuner(id2label=id2label,
                                   label2id=label2id,
                                   lr=1e-4, 
                                   lr_backbone=1e-5, 
                                   weight_decay=1e-4,
                                   train_dataloader=train_dataloader,
                                   val_dataloader=val_dataloader,
                                   metrics_interval=10,) 
    wandb_logger = WandbLogger(project='Signboard_Detection_Rectangle', name='detr_v1')

    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=5, 
        verbose=True, 
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(file_name='best_model-{epoch:02d}',
                                          save_top_k=1, monitor="val_loss",
                                          mode="min")
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
    # Start training
    trainer.fit(detr_finetuner)
    
def test(val_transforms, best_model_path, device='cpu', seed=42):
    set_seed(seed=seed)
    # initialize test dataset
    test_dataset = SignboardDetection(rootDir='signboardtext/images/test',
                                      label='signboardtext/annotations_signboard.json',
                                      label2id=label2id,
                                      transforms=val_transforms)
    # create dataloader
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn)
    # initialize postprocess, best model
    postprocess = PostProcess()
    best_model = DETRFinetuner.load_from_checkpoint(best_model_path,
                                                    id2label=id2label,
                                                    label2id=label2id,
                                                    postprocess=postprocess)
    # start testing
    best_model.eval()
    best_model.to(device)
    allBoundingBoxes = BoundingBoxes()
    evaluator = Evaluator()
    for i, image in enumerate(tqdm(test_dataloader)): 
        gt = image[1][0]
        pixel_values, pixel_mask = image[0].decompose()
        pixel_mask = (~pixel_mask).int() 
        results = best_model.predict_one_image(pixel_values, pixel_mask)
        bounding_boxes(allBoundingBoxes , i + 1, 
                       gt, results, current_gtbox_format='yolo', current_predbox_format='xyxy')
    
    # caculate mAP@50, 50-95
    mAP_50 = calculate_mAP(evaluator, allBoundingBoxes, iou_threshold=0.5)
    iou_thresholds = np.arange(0.50, 1.00, 0.05)
    mAP_scores = []
    for iou_threshold in iou_thresholds:
        mAP = calculate_mAP(evaluator, allBoundingBoxes, iou_threshold=iou_threshold)
        mAP_scores.append(mAP)

    print(f'mAP@50: {mAP_50*100:.2f}%')
    print(f'mAP@50-95: {np.mean(mAP_scores)*100:.2f}%') 

def inference(model, image_path, transforms, device='cpu', threshold=0.5, seed=42):
    set_seed(seed)
    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_size = image.shape[:2]
    # preprocess image
    pixel_values, _ = transforms(image, target=None)
    # Prediction
    model.eval() 
    model = model.to(device)
    results = model.predict_one_image(pixel_values.unsqueeze(dim=0), 
                                      original_size=orig_size, 
                                      threshold=threshold)
    # annotate image
    show_image = image.copy()
    for obj in range(len(results['labels'])):
        boxes = results['bboxes'][obj].detach().cpu().numpy()
        class_id = results['labels'][obj].item() 
        conf = results['scores'][obj].item() 
        font_scale = round(calculate_font_scale(show_image), 2)
        draw_bounding_box(show_image, boxes, class_id, conf, box_thickness=3, 
                          text_thickness=3, text_scale=font_scale, format_box='xyxy', text_color=(0, 165, 255))
    # Save annotated image
    cv2.imwrite('image_prediction.jpg', show_image)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', type=str, default='train', required=False, help='Operation mode: "train" for training the model, "test" for testing, or "infer" for inference')
    parser.add_argument('-best_model_path', type=str, default=None, required=False, help='Path to the best trained model checkpoint to load for testing or inference')
    parser.add_argument('-epochs', type=int, default=100, required=False, help='Number of training epochs')
    parser.add_argument('-size', type=int, default=800, required=False, help='Input image size (images will be resized to size x size)')
    parser.add_argument('-image_path', type=str, default=None, required=False, help='Path to a single image for inference')
    parser.add_argument('-threshold', type=float, default=0.5, required=False, help='Confidence threshold used to filter detected signboard regions')
    args = parser.parse_args()
    # initialize object
    label2id = {'signboard': 0}
    id2label = {v: k for k, v in label2id.items()}
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # initilize dataset and dataloader
    size = args.size
    train_transforms = Transforms(image_set='train', size=size)
    val_transforms = Transforms(image_set='val', size=size)
    
    if args.mode == 'train': 
        max_epochs = args.epochs
        train(train_transforms, val_transforms, label2id=label2id, id2label=id2label, 
              max_epochs=max_epochs, device=device)
    elif args.mode == 'test': 
        best_model_path = args.best_model_path
        assert best_model_path is not None
        test(val_transforms, best_model_path=best_model_path, device=device)
    elif args.mode == 'infer':
        best_model_path = args.best_model_path
        image_path = args.image_path
        threshold = args.threshold
        assert best_model_path is not None
        assert image_path is not None
        # initialize postprocess and load best model
        postprocess = PostProcess()
        best_model = DETRFinetuner.load_from_checkpoint(best_model_path,
                                                        id2label=id2label,
                                                        label2id=label2id,
                                                        postprocess=postprocess)
        inference(best_model, image_path, val_transforms, device=device, threshold=threshold)