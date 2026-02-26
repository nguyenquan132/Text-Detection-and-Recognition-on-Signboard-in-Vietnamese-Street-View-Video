"""
    Functions sigmoid_alpha and make_text_region are adapted from:
    TextPMs repository:
    https://github.com/GXYM/TextPMs.git
    MIT License - Copyright (c) 2020
    Implements: Zhang et al., IEEE TPAMI 2023

    Class MakeBorderMap, MakeShrinkMap and Functions shrink_polygon_py, shrink_polygon_pyclipper are adapted from:
    PaddleOCR2Pytorch repository:
    https://github.com/frotms/PaddleOCR2Pytorch.git
    Licensed under the Apache License 2.0.
"""

import torch
from torch.utils.data import Dataset
from collections import defaultdict
import os
from pathlib import Path
import json
import cv2
import numpy as np
from src.text_det.TextPMs.dataset.dataload import TextInstance
import pyclipper
from shapely.geometry import Polygon

def sigmoid_alpha(x, k):
    betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
    dm = max(np.max(x), 0.0001)
    res = (2 / (1 + np.exp(-x*k/dm)) - 1)*betak
    return np.maximum(0, res)

def make_text_region(img, polygons, scale, mask_cnt, alpha):
    h, w = img.shape[0]//scale, img.shape[1]//scale
    mask_ones = np.ones(img.shape[:2], np.uint8)
    mask_zeros = np.zeros(img.shape[:2], np.uint8)

    train_mask = np.ones((h, w), np.uint8)
    tr_mask = np.zeros((h, w, mask_cnt), np.float32)
    if polygons is None:
        return tr_mask, train_mask

    for polygon in polygons:
        instance_mask = mask_zeros.copy()
        cv2.fillPoly(instance_mask, [polygon.points.astype(np.int32)], color=(1,))
        # dmp = ndimg.distance_transform_edt(instance_mask[::scale, ::scale])  # distance transform
        dmp = cv2.distanceTransform(instance_mask[::scale, ::scale], cv2.DIST_L2, 5)
        for i, k in enumerate(alpha):
            tr_mask[:, :, i] = np.maximum(tr_mask[:, :, i], sigmoid_alpha(dmp, k))

        if polygon.text == '#':
            cv2.fillPoly(mask_ones, [polygon.points.astype(np.int32)], color=(0,))
            continue

    train_mask = mask_ones[::scale, ::scale]

    return tr_mask, train_mask

class MakeBorderMap():
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        canvas = np.zeros(im.shape[:2], dtype=np.float32)
        mask = np.zeros(im.shape[:2], dtype=np.float32)

        for i in range(len(text_polys)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        data['threshold_map'] = canvas
        data['threshold_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return
        distance = polygon_shape.area * (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)

        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)

        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2

def shrink_polygon_py(polygon, shrink_ratio):
    """
    对框进行缩放，返回去的比例为1/shrink_ratio 即可
    """
    cx = polygon[:, 0].mean()
    cy = polygon[:, 1].mean()
    polygon[:, 0] = cx + (polygon[:, 0] - cx) * shrink_ratio
    polygon[:, 1] = cy + (polygon[:, 1] - cy) * shrink_ratio
    return polygon


def shrink_polygon_pyclipper(polygon, shrink_ratio):
    polygon_shape = Polygon(polygon)
    distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrinked = padding.Execute(-distance)
    if shrinked == []:
        shrinked = np.array(shrinked)
    else:
        shrinked = np.array(shrinked[0]).reshape(-1, 2)
    return shrinked


class MakeShrinkMap():
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''

    def __init__(self, min_text_size=8, shrink_ratio=0.4, shrink_type='pyclipper'):
        shrink_func_dict = {'py': shrink_polygon_py, 'pyclipper': shrink_polygon_pyclipper}
        self.shrink_func = shrink_func_dict[shrink_type]
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        image = data['img']
        text_polys = data['text_polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                shrinked = self.shrink_func(polygon, self.shrink_ratio)
                if shrinked.size == 0:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)

        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        return cv2.contourArea(polygon)
        # edge = 0
        # for i in range(polygon.shape[0]):
        #     next_index = (i + 1) % polygon.shape[0]
        #     edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])
        #
        # return edge / 2.

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