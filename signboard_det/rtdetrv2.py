"""
The functions resize and box_xyxy_to_cxcywh, and the classes ToTensor, Normalize, Resize, and Compose are adapted from RT-DETRv2.
"""
import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from signboard_det.RTDETRv2.core.yaml_config import YAMLConfig
from signboard_det.RTDETRv2.optim.warmup import LinearWarmup
from signboard_det.RTDETRv2.data.dataloader import BatchImageCollateFunction
from signboard_det.RTDETRv2.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
from .dataset import SignboardDetection
from .metrics import bounding_boxes, calculate_mAP
from signboard_det.evaluation.BoundingBoxes import BoundingBoxes
from signboard_det.evaluation.Evaluator import *
from utils.visualize import draw_bounding_box
from utils.utility import set_seed, calculate_font_scale
from utils.transforms import filter_bboxes
import albumentations as A
from PIL import Image
import numpy as np
import cv2
import argparse
from tqdm import tqdm
os.environ["WANDB_API_KEY"] = "YOUR API KEY"

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
    
def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target

class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target

class Resize(object):
    def __init__(self, size, max_size=None):
        assert isinstance(size, (list, tuple))
        self.size = size
        self.max_size = max_size

    def __call__(self, img, target=None):
        return resize(img, target, self.size, self.max_size)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Transforms(object):
    def __init__(self, image_set, size):
        self.image_set = image_set
        self.transforms = Compose([
            Resize([size, size]),
            Compose([
                ToTensor(),
                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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


class RTDETRV2Finetuner(pl.LightningModule):
    def __init__(self, id2label, label2id, train_dataloader=None, val_dataloader=None,
                 metrics_interval=100, postprocess=None, optimizer_cfg_path=None):
        super(RTDETRV2Finetuner, self).__init__()
        self.id2label = id2label
        self.label2id = label2id
        self.num_classes = len(id2label.keys())
        self.model = RTDetrV2ForObjectDetection.from_pretrained('PekingU/rtdetr_v2_r50vd',
                                                                num_labels = self.num_classes,
                                                                id2label=self.id2label,
                                                                label2id=self.label2id,
                                                                ignore_mismatched_sizes=True)
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader

        # Postprocess outputs
        self.postprocess = postprocess

        # Config path contains optimizer and scheduler
        self.optimizer_cfg_path = optimizer_cfg_path

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        outputs = self.model(pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels)
        return outputs

    def predict_one_image(self, pixel_values, pixel_mask=None, 
                          original_size=None, threshold=0.25, denormalize=False): 
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
        results = self.postprocess(output)[0]
        
        if denormalize: 
            img_h, img_w = target_size.unbind(0)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
            results['boxes'] = results['boxes'] * scale_fct

        return filter_bboxes(results, pixel_values.shape[2:], original_size, threshold)
        
    def compute_loss(self, batch, batch_idx): 
        pixel_values = batch[0]
        pixel_mask = torch.any(pixel_values != 0, dim=1).int() 
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
                if '_aux_' not in k: 
                    self.log(f'train_{k}', v, batch_size=batch_size)

        return loss
        
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch[1]) # len(target)
        loss, loss_dict = self.compute_loss(batch, batch_idx)

        self.log('val_loss', loss, batch_size=batch_size)
        for k, v in loss_dict.items():
            if '_aux_' not in k: 
                self.log(f'val_{k}', v, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        assert self.optimizer_cfg_path is not None
        load_cfg = YAMLConfig(self.optimizer_cfg_path)
        cfg_optimizer = load_cfg.yaml_cfg['optimizer']
        cfg_lr_scheduler = load_cfg.yaml_cfg['lr_scheduler']
        cfg_lr_warmup_scheduler = load_cfg.yaml_cfg['lr_warmup_scheduler']
        
        params = load_cfg.get_optim_params(cfg_optimizer, self.model)
        optimizer = torch.optim.AdamW(params, lr=cfg_optimizer['lr'], 
                                     betas=cfg_optimizer['betas'], weight_decay=cfg_optimizer['weight_decay'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg_lr_scheduler['milestones'],
                                                           gamma=cfg_lr_scheduler['gamma'])
        self.lr_warmup_scheduler = LinearWarmup(lr_scheduler, warmup_duration=cfg_lr_warmup_scheduler['warmup_duration'])

        return [optimizer], [{
            'scheduler': lr_scheduler,
            'interval': 'step',
            'monitor': 'val_loss',
            'frequency': 1
        }]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.lr_warmup_scheduler:
            self.lr_warmup_scheduler.step()

            if hasattr(self, 'global_step') and self.global_step < 5:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Step {self.global_step}, LR: {current_lr:.8f}")

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        
    def train_dataloader(self):
        return self.train_dl
    
    def val_dataloader(self):
        return self.val_dl
    
def train(train_transforms, val_transforms, label2id, id2label, max_epochs=100, device='cpu', seed=42): 
    set_seed(seed=seed)
    #initialize train and val dataset
    train_dataset = SignboardDetection(rootDir='signboardtext/images/train',
                                       label='signboardtext/annotations_signboard.json',
                                       label2id=label2id, 
                                       transforms=train_transforms)
    val_dataset = SignboardDetection(rootDir='signboardtext/images/val',
                                     label='signboardtext/annotations_signboard.json',
                                     label2id =label2id,
                                     transforms=val_transforms)
    # create dataloader
    collate_fn = BatchImageCollateFunction()
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False,
                                collate_fn=collate_fn)
    
    rtdetrv2_finetuner = RTDETRV2Finetuner(id2label=id2label,
                                           label2id=label2id,
                                           train_dataloader=train_dataloader,
                                           val_dataloader=val_dataloader,
                                           metrics_interval=10,
                                           optimizer_cfg_path='src/signboard_det/rtdetr/rtdetrv2_pytorch/configs/rtdetrv2/include/optimizer.yml')
    
    wandb_logger = WandbLogger(project='Signboard_Detection_Rectangle', name='rtdetrv2')

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
        accelerator=device, # number of devices (e.g., GPUs if using CUDA)
        devices=1,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        gradient_clip_val=0.1
    )

    # Start training0
    trainer.fit(rtdetrv2_finetuner)
    
def test(val_transforms, best_model_path, device='cpu', seed=42):
    set_seed(seed=seed)
    # initialize test dataset
    test_dataset = SignboardDetection(rootDir='signboardtext/images/test',
                                      label='signboardtext/annotations_signboard.json',
                                      label2id=label2id,
                                      transforms=val_transforms)
    # create dataloader
    collate_fn = BatchImageCollateFunction()
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                 collate_fn=collate_fn)
    # initialize postprocess, best model
    postprocess = DetDETRPostProcessor(num_classes=len(id2label.keys()))
    best_model = RTDETRV2Finetuner.load_from_checkpoint(best_model_path,
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
        pixel_values = image[0]
        results = best_model.predict_one_image(pixel_values, denormalize=True)
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
    # preprocess image
    pixel_values, _ = transforms(image, target=None)
    # Prediction
    model.eval() 
    model = model.to(device)
    results = model.predict_one_image(pixel_values.unsqueeze(dim=0), 
                                      original_size=image.shape[:2], 
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
    parser.add_argument('-size', type=int, default=640, required=False, help='Input image size (images will be resized to size x size)')
    parser.add_argument('-image_path', type=str, default=None, required=False, help='Path to a single image for inference')
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
        assert best_model_path is not None
        assert image_path is not None
        # initialize postprocess and load best model
        postprocess = DetDETRPostProcessor(num_classes=len(id2label.keys()))
        best_model = RTDETRV2Finetuner.load_from_checkpoint(best_model_path,
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            postprocess=postprocess)
        inference(best_model, image_path, val_transforms, device=device, threshold=0.5)