"""
The functions resize_norm_img are adapted from OpenOCR
"""
from text_spotting.recognition.OpenOCR.tools.engine.config import Config
from text_spotting.recognition.OpenOCR.openrec.modeling.base_recognizer import BaseRecognizer
from text_spotting.recognition.OpenOCR.openrec.postprocess import build_post_process
from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch
import math
from PIL import Image
import random
import copy
import numpy as np
from utils.utility import get_state_dict

class RecTransform(object):
    def __init__(self, padding=False): 
        norms = []
        norms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.norms = T.Compose(norms)
        self.padding = padding
        self.padding_rand = False
        self.padding_doub = False
        self.interpolation = T.InterpolationMode.BICUBIC
        # self.base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]
        # self.base_h = 32
        # self.ceil = True

    def resize_norm_img(self, data, padding=True):
        img = data['image']
        w, h = img.size
        w, h = img.size
        # if self.ceil:
        #     gen_ratio = int(float(w) / float(h)) + 1
        # else:
        #     gen_ratio = max(1, round(float(w) / float(h)))
            
        # if self.padding_rand and random.random() < 0.5:
        #     padding = not padding
        # imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
        #     self.base_h * gen_ratio, self.base_h
        # ]
        imgW, imgH = 128, 32
        use_ratio = imgW // imgH
        if not padding:
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(
                    math.ceil(imgH * ratio * (random.random() + 0.5)))
                resized_w = min(imgW, resized_w)
        resized_image = F.resize(img, (imgH, resized_w),
                                 interpolation=self.interpolation)
        img = self.norms(resized_image)
        if resized_w < imgW and padding:
            # img = F.pad(img, [0, 0, imgW-resized_w, 0], fill=0.)
            if self.padding_doub and random.random() < 0.5:
                img = F.pad(img, [0, 0, imgW - resized_w, 0], fill=0.)
            else:
                img = F.pad(img, [imgW - resized_w, 0, 0, 0], fill=0.)
        valid_ratio = min(1.0, float(resized_w / imgW))
        data['image'] = img
        data['valid_ratio'] = valid_ratio
        return data
    
    def __call__(self, image): 
        assert isinstance(image, Image.Image)
        data = dict(image=image)
        data = self.resize_norm_img(data, self.padding)
        transformed_image = data['image']
        
        return transformed_image

class PARSeq(object): 
    def __init__(self, config_file, checkpoint_path, charset_file, max_text_length=25, device='cpu'):
        self.config = copy.deepcopy(self.load_config(config_file, charset_file, max_text_length))
        self.device = device

        # Preprocess 
        self.preprocess = RecTransform()

        # Postprocess and model
        self.postprocess = build_post_process(self.config['PostProcess'], self.config['Global'])
        self.config['Architecture']['Decoder']['out_channels'] = self.postprocess.get_character_num()
        self.model = BaseRecognizer(self.config['Architecture'])

        # Load model's state dict
        self.load_state_dict(checkpoint_path)
        self.model.eval()

    def load_config(self, config_file, charset_file, max_text_length): 
        config = Config(config_file).cfg
        config['Global']['character_dict_path'] = charset_file
        config['PostProcess']['character_dict_path'] = charset_file
        config['Train']['dataset']['transforms'][2]['ARLabelEncode']['character_dict_path'] = charset_file
        config['Eval']['dataset']['transforms'][1]['ARLabelEncode']['character_dict_path'] = charset_file
        config['Train']['dataset']['transforms'][2]['ARLabelEncode']['max_text_length'] = max_text_length
        config['Eval']['dataset']['transforms'][1]['ARLabelEncode']['max_text_length'] = max_text_length
        config['Global']['max_text_length'] = max_text_length
        config['Architecture']['Decoder']['max_label_length'] = max_text_length

        return config

    def forward(self, pixel_values):
        outputs = self.model(pixel_values)
        return outputs
        
    def load_state_dict(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        new_state_dict = get_state_dict(state_dict, idx=6)
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device)
        print('Load state dict for PARSeq successfully!')

    def _preprocess(self, image): 
        pixel_values = self.preprocess(Image.fromarray(image))
        pixel_values = pixel_values.unsqueeze(dim=0)
        return pixel_values

    def _postprocess(self, preds): 
        text, score = self.postprocess(preds)[0]
        return text, score

    def __call__(self, image):
        assert isinstance(image, np.ndarray)
        pixel_values = self._preprocess(image)
        with torch.inference_mode(): 
            preds = self.forward(pixel_values.to(self.device))

        text, score = self._postprocess(preds)
        return text, score