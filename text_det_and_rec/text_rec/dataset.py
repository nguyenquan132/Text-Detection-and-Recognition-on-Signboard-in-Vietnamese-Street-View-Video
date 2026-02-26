import torchvision.transforms.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
from collections import defaultdict
import os
from pathlib import Path
import json
import cv2
import numpy as np
import random
import math
from PIL import Image

class SignboardText(Dataset):
    def __init__(self, rootDir, label, transforms=None, ops=None, ds_width=None, epoch=1, min_ratio=1, max_ratio=12,
                 fixed_size=None):
        assert isinstance(rootDir, str)
        assert isinstance(label, str)
        self.rootDir = rootDir
        self.label = label

        self.list_folder = os.listdir(rootDir)
        self.list_image = []
        # Check rootDir contains folder or not
        if Path(os.path.join(rootDir, self.list_folder[0])).is_dir(): 
            for folder in self.list_folder:
                folder_path = os.path.join(rootDir, folder)
                for img in os.listdir(folder_path): 
                    img_path = os.path.join(folder, img)
                    self.list_image.append(img_path)

        self.annotation_lookup = self.read_json(label)
        self.transforms = transforms
        self.ds_width = ds_width
        self.seed = epoch
        self.max_ratio = max_ratio
        self.padding = False
        self.padding_rand = False
        self.padding_doub = False
        self.base_shape = [[64, 64], [96, 48], [112, 40], [128, 32]]
        self.base_h = 32
        self.interpolation = T.InterpolationMode.BICUBIC
        self.ops = ops
        self.fixed_size = fixed_size

        norms = []
        norms.extend([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])
        self.norms = T.Compose(norms)

        wh_ratio = np.around(np.array(self.get_wh_ratio()))
        self.wh_ratio = np.clip(wh_ratio, a_min=min_ratio, a_max=max_ratio)
        count_error = (1 * (self.wh_ratio == 0)).sum()
        if count_error > 0: 
            print(f'Number of error image: {count_error}')
        self.wh_ratio_sort = np.argsort(self.wh_ratio)

    def read_json(self, file): 
        with open(file, 'r') as file: 
            data = json.load(file)

        # Create mapping: id images and image_id annotations
        annotations_lookup = defaultdict(list)
        for info in data['text_info']:
            annotations_lookup[info['text_image_id']].append(info)

        return annotations_lookup

    def __len__(self):
        return len(self.list_image)

    def get_wh_ratio(self):
        wh_ratio = []
        for image_id in self.list_image:
            info = self.annotation_lookup.get(image_id)[0]
            w, h = info['width'], info['height']
            wh_ratio.append(float(w) / float(h))
        return wh_ratio

    def resize_norm_img(self, data, gen_ratio, padding=True):
        img = data['image']
        w, h = img.size
        if self.padding_rand and random.random() < 0.5:
            padding = not padding
        if self.fixed_size is None:
            imgW, imgH = self.base_shape[gen_ratio - 1] if gen_ratio <= 4 else [
                self.base_h * gen_ratio, self.base_h
            ]
        else: imgW, imgH = self.fixed_size
        use_ratio = imgW // imgH
        # if use_ratio >= (w // h) + 2:
        #     # self.error += 1
        #     return None
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
    
    def __getitem__(self, properties):
        idx = properties[2]
        ratio = properties[3]
        image_id = self.list_image[idx]
        image_path = os.path.join(self.rootDir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        
        info = self.annotation_lookup.get(image_id)[0]
        if info is not None: label = info['transcription']
        else: 
            print(image_path)
            label = None 

        gt = {
            'file_name': image_id, 
            'image_size': image.shape[:2],
            'label': label
        }
        
        if self.transforms is not None: 
            data = dict(image=Image.fromarray(image), label=label)
            inputs = self.transforms(data, self.ops[:-1])

        if inputs is None:
            ratio_ids = np.where(self.wh_ratio == ratio)[0].tolist()
            ids = random.sample(ratio_ids, 1)
            return self.__getitem__([image_width, image_height, ids[0], ratio])
                
        inputs = self.resize_norm_img(inputs, ratio, padding=self.padding)
        inputs = self.transforms(inputs, self.ops[-1:])
        gt['inputs'] = inputs
        return gt