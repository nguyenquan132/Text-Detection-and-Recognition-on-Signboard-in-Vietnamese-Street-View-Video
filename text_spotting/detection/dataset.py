import torch
from torch.utils.data import Dataset
from collections import defaultdict
import os
from pathlib import Path
import json
import cv2
import numpy as np
from text_spotting.detection.TextPMs.util.instance import TextInstance, make_text_region
from utils import check_and_refine_valid_box

class SignboardText(Dataset):
    def __init__(self, rootDir, label, transforms=None, is_training=True, 
                 scale=None, mask_cnt=None, alpha=None, make_border_map=None, make_shrink_map=None):
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
        self.is_training = is_training
        # Parameter of TextPMs
        self.scale = scale
        self.mask_cnt = mask_cnt
        self.alpha = alpha
        # Parameter of DBNet++
        self.make_border_map = make_border_map
        self.make_shrink_map = make_shrink_map

    def read_json(self, file): 
        with open(file, 'r') as file: 
            data = json.load(file)

        # Create mapping: id images and image_id annotations
        annotations_lookup = defaultdict(list)
        for info in data['annotations']:
            annotations_lookup[info['image_id']].append(info)

        return annotations_lookup

    def extract_info(self, info):
        labels = []
        for p in info:
            object = {}
            bbox = p['bbox']
            seg = p['segmentation']
            transcript = p['transcription']

            object['bbox'] = bbox
            object['segmentation'] = seg
            object['transcription'] = transcript

            labels.append(object)

        return labels

    def __len__(self):
        return len(self.list_image)
    
    def __getitem__(self, idx):
        image_id = self.list_image[idx]
        image_path = os.path.join(self.rootDir, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        info = self.annotation_lookup.get(image_id)
        if info is not None: labels = self.extract_info(info)
        else: 
            print(image_path)
            labels = None 

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]
        gt = {
            'file_name': image_id, 
            'image_size': image.shape[:2],
        }
        polygons, transcriptions = [], []
        if labels is not None: 
            for i, label in enumerate(labels): 
                segmentation = label['segmentation']
                transcription = '#' if label['transcription'] in ['#', '##', '###'] else label['transcription']
                bbox = label['bbox']
                new_bbox = check_and_refine_valid_box(bbox, image_width, image_height)
                if new_bbox is None: continue
                polygons.append(TextInstance(segmentation, 'c', transcription))

        if self.transforms is not None:
            if len(polygons) == 0: image, _ = self.transforms(image, None)
            else: image, polygons = self.transforms(image, polygons)

            if self.is_training:
                if self.make_border_map is None:
                    tr_mask, train_mask = make_text_region(image, polygons, self.scale, self.mask_cnt, self.alpha)
                    # clip value (0, 1)
                    tr_mask = np.clip(tr_mask, 0, 1)
                    train_mask = np.clip(train_mask, 0, 1)
                    train_mask = torch.from_numpy(train_mask).byte()
                    tr_mask = torch.from_numpy(tr_mask).float()
                    gt['tr_mask'] = tr_mask
                    gt['train_mask'] = train_mask
                else: 
                    ignore_tags = [True if polygon.text == '#' else False for polygon in polygons]
                    data = dict(img=image, text_polys=np.array([polygon.points for polygon in polygons], dtype=np.int32), 
                                texts=[polygon.text for polygon in polygons], ignore_tags=ignore_tags)
                    data = self.make_border_map(data)
                    data = self.make_shrink_map(data)
                    gt['threshold_map'] = torch.from_numpy(data['threshold_map'])
                    gt['threshold_mask'] = torch.from_numpy(data['threshold_mask'])
                    gt['shrink_map'] = torch.from_numpy(data['shrink_map'])
                    gt['shrink_mask'] = torch.from_numpy(data['shrink_mask'])

        if image.shape[-1] == 3: 
            image = image.transpose(2, 0, 1) # C, H, W

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
            
        gt['image'] = image
        gt['polygons'] = np.array([polygon.points for polygon in polygons], dtype=np.int32) if polygons is not None else []
        gt['transcriptions'] = [polygon.text for polygon in polygons] if polygons is not None else []

        return gt