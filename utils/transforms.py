import torch
import numpy as np
from imantics import Mask

def denormalize(image_width, image_height, boxes):
    """
    Format boxes can both xyxy and yolo
    """
    [x1, y1, x2, y2] = boxes
    x1 = float(x1) * image_width
    y1 = float(y1) * image_height
    x2 = float(x2) * image_width
    y2 = float(y2) * image_height

    return [x1, y1, x2, y2]

def normalize(image_width, image_height, boxes):
    """
    Format boxes can both xyxy and yolo
    """
    [x1, y1, x2, y2] = boxes
    x1 = float(x1) / image_width
    y1 = float(y1) / image_height
    x2 = float(x2) / image_width
    y2 = float(y2) / image_height

    return [x1, y1, x2, y2]

def convert_box_format(boxes, current_format_box: str='yolo', output_format_box: str='none'):
    """
    Args:
        format_box: str ('yolo', 'xyxy', 'xywh')
            yolo: x_center, y_center, width, height
            xyxy: xmin, ymin, xmax, ymax
            xywh: xmin, ymin, width, height
        EX: 
            If your format box is yolo, you should set: current_format_box: 'yolo'. 
            Then, you want to convert to 'xyxy' format box, you should set: output_format_box: 'xyxy'
    Return: 
        output format box, you need
        EX: output format: [int(x_min), int(y_min), int(x_max), int(y_max)]
    """
    # Yolo format
    if current_format_box == 'yolo':
        [x_center, y_center, width, height] = boxes 
        if output_format_box == 'xyxy': 
            x_min, y_min, x_max, y_max = x_center - (width / 2), y_center - (height / 2), x_center + (width / 2), y_center + (height / 2)

            return [x_min, y_min, x_max, y_max]
        if output_format_box == 'xywh': 
            x_min, y_min, width, height = x_center - (width / 2), y_center - (height / 2), width, height

            return [x_min, y_min, width, height]
    # xyxy Format 
    if current_format_box == 'xyxy':
        [x_min, y_min, x_max, y_max] = boxes 
        if output_format_box == 'yolo': 
            x_center, y_center, width, height = (x_max + x_min) / 2, (y_max + y_min) / 2, (x_max - x_min), (y_max - y_min)

            return [x_center, y_center, width, height]
        if output_format_box == 'xywh': 
            x_min, y_min, width, height = x_min, y_min, (x_max - x_min), (y_max - y_min)

            return [x_min, y_min, width, height]
    
    if current_format_box == 'xywh':
        [x_min, y_min, width, height] = boxes 
        if output_format_box == 'xyxy': 
            x_min, y_min, x_max, y_max = x_min, y_min, x_min + width, y_min + height

            return [x_min, y_min, x_max, y_max]
        if output_format_box == 'yolo': 
            x_center, y_center, width, height = x_min + (width / 2), y_min + (height / 2), width, height

            return [x_center, y_center, width, height]
        
def rescale_box(current_size, target_size, boxes=None):
    """
    Args: 
        current_size: (width, height)
        target_size: (width, height)
    Returns:
        boxes rescaled to original image
    """
    target_width, target_height = target_size
    current_width, current_height = current_size
    ratio = [target_width / current_width, target_height / current_height]

    if boxes is not None: 
        boxes = [np.round(box * ratio).astype(np.int32) for box in boxes] 

    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding: bool = True, xywh: bool = False):
    """ 
    Copied from the Ultralytics YOLO repository
    Source: https://github.com/ultralytics/ultralytics.git
    Copyright (c) Ultralytics
    Licensed under the GNU AGPL-3.0 License
    """

    """
    Rescale bounding boxes from one image shape to another.

    Rescales bounding boxes from img1_shape to img0_shape, accounting for padding and aspect ratio changes.
    Supports both xyxy and xywh box formats.

    Args:
        img1_shape (tuple): Shape of the source image (height, width).
        boxes (torch.Tensor): Bounding boxes to rescale in format (N, 4).
        img0_shape (tuple): Shape of the target image (height, width).
        ratio_pad (tuple, optional): Tuple of (ratio, pad) for scaling. If None, calculated from image shapes.
        padding (bool): Whether boxes are based on YOLO-style augmented images with padding.
        xywh (bool): Whether box format is xywh (True) or xyxy (False).

    Returns:
        (torch.Tensor): Rescaled bounding boxes in the same format as input.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    else:
        gain = ratio_pad[0][0]
        pad_x, pad_y = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        if not xywh:
            boxes[..., 2] -= pad_x  # x padding
            boxes[..., 3] -= pad_y  # y padding
    boxes[..., :4] /= gain
    return boxes

def filter_bboxes(outputs, current_size, target_size, threshold=0.25): 
    scores, labels, bboxes = outputs['scores'], outputs['labels'], outputs['boxes']
    keep = scores > threshold 

    scores_keep = scores[keep]
    labels_keep = labels[keep]
    bboxes = bboxes[keep]

    if target_size is not None: 
        assert isinstance(current_size, torch.Tensor) and isinstance(target_size, torch.Tensor)
        assert len(current_size) == 2 and len(target_size) == 2  # (H, W)
        # Rescale resized box to original box
        target_size = target_size.repeat(1, 2)
        current_size = current_size.repeat(1, 2)
        bboxes = bboxes * (target_size / current_size)

    return {'scores': scores_keep, 'labels': labels_keep, 'bboxes': bboxes}

def mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
    polygons = Mask(mask).polygons()
    return polygons.points

def check_and_refine_valid_box(box, image_size, min_width, min_height, box_format='xywh'):
    """
    if box contains the negative value, this function will convert the negative value to 0 value.
    Args: 
        box: [xmin, ymin, width, height]
        image_width: Image width
        image_height: Image height
    Returns: 
        [xmin, ymin, width, height] or None if invalid
    """
    assert box_format in ['xywh', 'xyxy']
    image_width, image_height = image_size 
    if box_format == 'xywh': 
        xmin, ymin, width, height = box
        xmax = xmin + width
        ymax = ymin + height
    else: 
        xmin, ymin, xmax, ymax = box

    intersect_x1 = max(0, xmin)
    intersect_y1 = max(0, ymin)
    intersect_x2 = min(image_width, xmax)
    intersect_y2 = min(image_height, ymax)

    new_width = intersect_x2 - intersect_x1
    new_height = intersect_y2 - intersect_y1
    
    # Check valid box
    if new_width < min_width or new_height < min_height:
        return None  # Invalid bbox

    return [intersect_x1, intersect_y1, new_width, new_height]

def check_and_refine_polygon(polygon, image_size, min_width=12, min_height=6):
    """Validate và refine polygon"""
    polygon = np.array(polygon, dtype=np.float32)
    image_width, image_height = image_size
    
    # Check NaN
    if np.isnan(polygon).any():
        return None
    
    # Clip image bounds
    polygon[:, 0] = np.clip(polygon[:, 0], 0, image_width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, image_height - 1)
    
    # Check size
    width = polygon[:, 0].max() - polygon[:, 0].min()
    height = polygon[:, 1].max() - polygon[:, 1].min()

    if width < min_width or height < min_height:
        return None
    
    return polygon