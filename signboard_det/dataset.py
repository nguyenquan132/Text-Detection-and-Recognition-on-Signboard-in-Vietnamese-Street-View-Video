from torch.utils.data import Dataset
from pathlib import Path
import os
from collections import defaultdict
import json
import cv2
import torch
import numpy as np
from utils import convert_box_format

class SignboardDetection(Dataset):
    def __init__(self, rootDir, label, label2id, transforms=None):
        assert isinstance(rootDir, str)
        assert isinstance(label, str)
        self.rootDir = rootDir
        self.label = label
        self.label2id = label2id

        self.list_folder = os.listdir(rootDir)
        # Check rootDir contains folder or not
        if Path(os.path.join(rootDir, self.list_folder[0])).is_dir(): 
            self.list_image = []
            for folder in self.list_folder:
                folder_path = os.path.join(rootDir, folder)
                for img in os.listdir(folder_path): 
                    img_path = os.path.join(folder, img)
                    self.list_image.append(img_path)
            
        self.annotation_lookup = self.read_json(label)
        self.transforms = transforms

    def read_json(self, file): 
        with open(file, 'r', encoding='utf-8') as file: 
            data = json.load(file)

        # Create mapping: id images and image_id annotations
        annotations_lookup = defaultdict(list)
        for info in data['annotations_signboard']:
            annotations_lookup[info['image_id']].append(info)

        return annotations_lookup
        
    def __len__(self):
        return len(self.list_image)
    def __getitem__(self, idx):
        image_id = self.list_image[idx]
        image_path = os.path.join(self.rootDir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        annotations = self.annotation_lookup.get(image_id)

        if annotations is not None: 
            target = {
                'boxes': [],
                'labels': [],
                'area': [],
            }
            for ann in annotations: 
                bbox = convert_box_format(ann['bbox'], current_format_box='xywh', output_format_box='xyxy')
                target['boxes'].append(bbox)
                label_id = self.label2id[ann['class']]
                target['labels'].append(label_id)
                target['area'].append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) # (width * height)

            target["boxes"] = torch.tensor(target["boxes"], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
            target['area'] = torch.tensor(target['area'], dtype=torch.float32)
            target["iscrowd"] = torch.zeros((len(target["labels"]),), dtype=torch.int64)
            target['orig_size'] = torch.tensor([img_h, img_w])
            target['size'] = torch.tensor([img_h, img_w])

        else: 
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),  
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64),
                'orig_size': torch.tensor([img_h, img_w]),
                'size': torch.tensor([img_h, img_w]),
            }

        if self.transforms is not None: 
            image, target = self.transforms(image, target)

        target['class_labels'] = target.pop('labels')

        return image, target
    

class SignboardSegmentation(Dataset):
    def __init__(self, rootDir, label, feature_extractor, transforms=None):
        self.rootDir = rootDir
        self.label = label
        self.feature_extractor = feature_extractor

        self.list_folder = os.listdir(rootDir)
        # Check rootDir contains folder or not
        if Path(os.path.join(rootDir, self.list_folder[0])).is_dir(): 
            self.list_image = []
            for folder in self.list_folder:
                folder_path = os.path.join(rootDir, folder)
                for img in os.listdir(folder_path): 
                    img_path = os.path.join(folder, img)
                    self.list_image.append(img_path)
            
        self.annotations_lookup = self.read_json(label)
        self.transforms = transforms

    def read_json(self, file): 
        with open(file, 'r', encoding='utf-8') as file: 
            data = json.load(file)

        # Create mapping: id images and image_id annotations
        annotations_lookup = defaultdict(list)
        for info in data['annotations_signboard']:
            annotations_lookup[info['image_id']].append(info)

        return annotations_lookup
        
    def __len__(self):
        return len(self.list_image)
    def __getitem__(self, idx):
        image_id = self.list_image[idx]
        image_path = os.path.join(self.rootDir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation_map = np.zeros(image.shape[:2], dtype=np.uint8)
        segmentation = self.annotations_lookup.get(image_id)

        if segmentation is not None: 
            for seg in segmentation: 
                points = np.array(seg['segmentation'])
                cv2.fillPoly(segmentation_map, [points], 1)
                
        if self.transforms: 
            augmented = self.transforms(image=image, mask=segmentation_map)
            image = augmented['image']
            segmentation_map = augmented['mask']
                
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs