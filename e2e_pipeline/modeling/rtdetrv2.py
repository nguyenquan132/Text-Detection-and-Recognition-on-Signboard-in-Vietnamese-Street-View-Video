from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from signboard_det.RTDETRv2.nn.postprocessor.detr_postprocessor import DetDETRPostProcessor
import torch
import torchvision.transforms.functional as F
from torch.nn import functional
from utils.utility import get_state_dict
import numpy as np
from PIL import Image
from signboard_det.rtdetrv2 import Transforms

class RTDETRV2(object):
    def __init__(self, id2label, checkpoint_path, device='cpu', threshold=0.5):
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.num_classes = len(id2label.keys())
        self.model = RTDetrV2ForObjectDetection.from_pretrained('PekingU/rtdetr_v2_r50vd',
                                                                num_labels = self.num_classes,
                                                                id2label=self.id2label,
                                                                label2id=self.label2id,
                                                                ignore_mismatched_sizes=True)
        
        self.device = device
        self.threshold = threshold
        # Preprocess 
        self.preprocess = Transforms(image_set='val', size=640) 
        # Postprocess 
        self.postprocess = DetDETRPostProcessor(num_classes=len(id2label.keys()))

        # Load model's state dict
        self.load_state_dict(checkpoint_path)
        self.model.eval()

    def forward(self, pixel_values, pixel_mask, labels=None):
        outputs = self.model(pixel_values=pixel_values,
                            pixel_mask=pixel_mask,
                            labels=labels)
        return outputs

    def load_state_dict(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        new_state_dict = get_state_dict(state_dict, idx=6)
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device)
        print('Load state dict for RTDETRv2 successfully!')

    def _preprocess(self, image):
        pixel_values = self.preprocess(image).unsqueeze(dim=0)
        pixel_mask = torch.ones((1, pixel_values.shape[-2], pixel_values.shape[-1]))
        return pixel_values, pixel_mask

    def filter_bboxes(self, outputs, current_size, target_size): 
        scores, labels, bboxes = outputs['scores'], outputs['labels'], outputs['boxes']
        keep = scores >= self.threshold 
    
        scores_keep = scores[keep]
        labels_keep = labels[keep]
        bboxes = bboxes[keep]
    
        if target_size is not None: 
            # Rescale resized box to original box
            target_size = target_size.repeat(1, 2).to(self.device)
            current_size = current_size.repeat(1, 2).to(self.device)
            bboxes = bboxes * (target_size / current_size)
    
        return {'scores': scores_keep, 'labels': labels_keep, 'bboxes': bboxes}

    def _postprocess(self, preds, pixel_values, orig_size):
        target_size = torch.tensor(pixel_values.shape[2:]) # Resized image
        orig_size = torch.tensor(orig_size)
        pred = {k: v for k, v in preds.items()}
        pred['pred_logits'] = pred.pop('logits')
        results = self.postprocess(pred)[0]

        # denormalize bbox
        img_h, img_w = target_size.unbind(0)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0).to(self.device)
        results['boxes'] = results['boxes'] * scale_fct

        results = self.filter_bboxes(results, target_size, orig_size)
        
        return results

    def __call__(self, image): 
        orig_size = Image.fromarray(image).size # W, H
        pixel_values, pixel_mask = self._preprocess(image)

        with torch.inference_mode(): 
            preds = self.forward(pixel_values=pixel_values.to(self.device), 
                                 pixel_mask=pixel_mask.to(self.device))

        results = self._postprocess(preds, pixel_values, orig_size)
        return results