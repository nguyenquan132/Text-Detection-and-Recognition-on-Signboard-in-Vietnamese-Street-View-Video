from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import albumentations as A
from imantics import Polygons, Mask
import torch
import numpy as np
from utils.utility import get_state_dict

class Transforms(object): 
    def __init__(self, size): 
        self.mean=np.array([123.675, 116.28, 103.53]) / 255
        self.std=np.array([58.395, 57.12, 57.375]) / 255
        self.transform = A.Compose([
            A.Resize(size, size, always_apply=True),
            A.Normalize(mean=self.mean, std=self.std)
        ], is_check_shapes=False)

        self.feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
        self.feature_extractor.reduce_labels = False
        self.feature_extractor.do_resize = False
        self.feature_extractor.do_rescale = False
        self.feature_extractor.do_normalize = False 
        
    def __call__(self, image): 
        image = self.transform(image=image)['image']
        encoded_inputs = self.feature_extractor(image, return_tensors='pt')
        for k, v in encoded_inputs.items(): 
            encoded_inputs[k].squeeze_() 

        pixel_values = encoded_inputs['pixel_values']
        return pixel_values

        
class SegFormer(object): 
    def __init__(self, id2label, checkpoint_path, device='cpu'):
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.num_classes = len(id2label.keys())
        self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-cityscapes-1024-1024", 
                return_dict=False, 
                num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        self.device = device
        # Preprocess
        self.preprocess = Transforms(size=640)
        # Postprocess
        self.postprocess = torch.nn.functional.interpolate

        # Load model's state dict
        self.load_state_dict(checkpoint_path)
        self.model.eval()

    def forward(self, pixel_values): 
        outputs = self.model(pixel_values=pixel_values)
        return(outputs)
        
    def load_state_dict(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        new_state_dict = get_state_dict(state_dict, idx=6)
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device)
        print('Load state dict for SegFormer successfully!')

    def _preprocess(self, image):
        pixel_values = self.preprocess(image)
        pixel_values = pixel_values.unsqueeze(dim=0)
        return pixel_values

    def _postprocess(self, preds, target_size): 
        logits = preds[0]
        upsampled_logits = self.postprocess(logits, size=target_size, 
                                           mode='bilinear', align_corners=False)
        predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
        predicted_mask = predicted_mask[0, ...]

        return predicted_mask

    def mask_to_polygons(self, mask): 
        assert isinstance(mask, np.ndarray)
        polygons = Mask(mask).polygons()
        return polygons.points

    def __call__(self, image): 
        orig_size = image.shape[:2]
        pixel_values = self._preprocess(image)
        
        with torch.inference_mode(): 
            preds = self.forward(pixel_values.to(self.device))

        mask = self._postprocess(preds, target_size=orig_size)
        polygons = self.mask_to_polygons(mask)

        results = dict(mask=mask, polygons=polygons)
        return results