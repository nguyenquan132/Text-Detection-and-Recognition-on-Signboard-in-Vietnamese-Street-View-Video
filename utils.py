from collections import OrderedDict
import cv2
import numpy as np
import random
import torch
import os
from pathlib import Path
import zipfile 
import re
from tqdm import tqdm 
import yaml
import xml.etree.ElementTree as ET
class_name_to_id_mapping = {'signboard': 0}
class_id_to_name_mapping = {0: 'signboard'}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)               # CPU
    torch.cuda.manual_seed(seed)          # GPU
    torch.cuda.manual_seed_all(seed)      # Multi-GPU (nếu có)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_state_dict(state_dict, idx): 
    new_state_dict = OrderedDict()
    for k, v in state_dict.items(): 
        new_k = k[idx:] 
        new_state_dict[new_k] = v

    return new_state_dict

def parse_yaml(config_file): 
    with open(config_file, 'r') as f: 
        cfg = yaml.load(f, yaml.SafeLoader)

    return cfg

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

def draw_polygon(image, seg, thickness, color):
    seg = np.asarray(seg, dtype=np.int32)
    seg = seg.reshape((-1, 1, 2))
    cv2.polylines(image, pts=[seg], isClosed=True, color=color, thickness=thickness)
    return image

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

def draw_bounding_box(image, boxes, class_id=None, confidence=None, box_thickness=2, text_thickness=2, 
                      text_scale=0.7, format_box: str='yolo', mode: str='none', box_color=(0, 255, 0),
                      text_color=(255, 265, 0)):
    """
    Args: 
        format_box: 'yolo' or 'xywh'. If format_box is yolo, we need to convert to xyxy to plot annotated image.
        mode: 'normalize', 'denormalize', 'none'. 
            if mode == 'normalize': yolo boxes format need to normalize with range [0, 1]
            if mode == 'denormalize': yolo boxes format need to denormalize to plot annotated image
            if mode == 'none': only convert yolo to xyxy
        
        Note: format_box is xyxy, you don't need to call convert_box_format function.
    Return: 
        Annotated image
    """
    image_height, image_width = image.shape[:2]
    # Scale boxes 
    if mode == 'normalize': boxes = normalize(image_width, image_height, boxes)
    if mode == 'denormalize': boxes = denormalize(image_width, image_height, boxes)
    # Convert format boxes 
    if format_box == 'yolo': 
        boxes = convert_box_format(boxes, current_format_box=format_box,
                                                          output_format_box='xyxy')
    if format_box == 'xywh':
        boxes = convert_box_format(boxes, current_format_box=format_box, 
                                                          output_format_box='xyxy')
        
    [x_min, y_min, x_max, y_max] = [int(box) for box in boxes]
    # x_text_center = x_min + (x_max - x_min) // 4
    
    cv2.rectangle(img=image, pt1=(x_min, y_min), pt2=(x_max, y_max), 
                  color=box_color, thickness=box_thickness)
    if class_id is not None: 
        class_name = class_id_to_name_mapping[int(class_id)]
        if confidence is not None: text = f'{class_name} {confidence:.2f}'
        else: text = f'{class_name}'

        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)

        text_x, text_y = x_min + (x_max - x_min - text_width) // 2, y_min
        
        if y_min < text_height + 5:
            text_y = y_max + text_height + 5
        else:
            text_y = y_min - 5 
            
        cv2.putText(img=image, text=text, org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=text_scale, color=text_color, thickness=text_thickness, lineType=cv2.LINE_AA)
    return image

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

def order_points_counter_clockwise(polygon):
    """
    Orders the 4 points of a quadrilateral in counter-clockwise order,
    starting from the top-left point.
    
    Args:
        polygon: A numpy array of shape (4, 2) representing 4 (x, y) points.
    
    Returns:
        A numpy array of shape (4, 2) with points ordered counter-clockwise as:
        [top-left, bottom-left, bottom-right, top-right]
    """
    # Create a copy to avoid modifying the original input
    rect = polygon.copy()
    
    # Calculate the sum and difference of coordinates
    s = rect.sum(axis=1)
    diff = np.diff(rect, axis=1)
    
    # Identify the characteristic points
    top_left = rect[np.argmin(s)]       # Smallest (x+y) sum
    bottom_right = rect[np.argmax(s)]   # Largest (x+y) sum
    top_right = rect[np.argmin(diff)]   # Smallest (x-y) difference
    bottom_left = rect[np.argmax(diff)] # Largest (x-y) difference

    counter_clockwise = np.array([top_left, bottom_left, 
                                  bottom_right, top_right], dtype=np.float32)
    
    return counter_clockwise

def align_polygon_perspective(img, polygons): 
    # Points are ordered anti-clockwise
    pt_A, pt_B, pt_C, pt_D = order_points_counter_clockwise(polygons)
    
    width_AD = np.linalg.norm(pt_A - pt_D)
    width_BC = np.linalg.norm(pt_B - pt_C)
    maxWidth = max(int(width_AD), int(width_BC))
    
    height_AB = np.linalg.norm(pt_A - pt_B)
    height_CD = np.linalg.norm(pt_C - pt_D)
    maxHeight = max(int(height_AB), int(height_CD))
    

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts,output_pts)

    # Inverse Matrix: Aligned → Original 
    M_inv = cv2.getPerspectiveTransform(output_pts, input_pts)

    # Apply the perspective transformation to the entire input image to get the final transformed image.
    aligned = cv2.warpPerspective(img,M,(maxWidth, maxHeight),flags=cv2.INTER_CUBIC)

    return aligned, M_inv

def calculate_trapezoid_angles_positive(points):
    """
    Tính góc trong khoảng [0°, 180°]
    """
    p0, p1, p2, p3 = points

    angle = np.degrees(np.arctan2(p3[1] - p0[1], p3[0] - p0[0]))
    # Chuyển về [0°, 180°]
    angle = abs(angle) % 180
    if angle > 90:
        angle = 180 - angle
    
    return angle

def smart_crop_or_align(img, quadrilateral, polygon, angle_threshold=5):
    """
    Normalize góc về [0, 90] để dễ hiểu.
    0° = thẳng ngang hoặc thẳng dọc
    45° = xoay 45°
    """
    quadrilateral = order_points_counter_clockwise(quadrilateral)
    angle = calculate_trapezoid_angles_positive(quadrilateral)
    
    if angle <= angle_threshold:
        x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
        result = img[y:y+h, x:x+w]
        M_inv = np.array([[1, 0, x],
                        [0, 1, y],
                        [0, 0, 1]], dtype=np.float32)
    else:
        # Xoay nhiều → Align
        result, M_inv = align_polygon_perspective(img, quadrilateral)
    
    return result, M_inv

def transform_boxes_back(boxes, M_inv):
    """
    Transform boxes from aligned image back to original image space.
    
    Args:
        boxes: List of polygons, each with shape (n_points, 2)
        M_inv: Inverse perspective transformation matrix (3x3)
    
    Returns:
        List of transformed polygons in original image coordinates.
    """
    boxes_original = []
    for box in boxes:
        box_np = np.array(box, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(box_np, M_inv)
        boxes_original.append(transformed.reshape(-1, 2))
    return boxes_original

def calculate_font_scale(image, base_scale=9e-4):
    height, width = image.shape[:2]
    diagonal = (width**2 + height**2)**0.5
    return max(diagonal * base_scale, 0.5) 

def create_folder(folder): 
    if not Path(folder).is_dir():
        Path(folder).mkdir(parents=True, exist_ok=True)

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

def save_output_txt(file_name, polygons, save_folder,
                    pred_transcriptions=None, boxes=None): 
    sub_folder, file = Path(file_name).parent, Path(file_name).with_suffix('.txt').name
    pred_folder = os.path.join(save_folder, sub_folder)

    create_folder(pred_folder)
    pred_file = os.path.join(pred_folder, file)
    if len(polygons) > 0: 
        if boxes is not None: 
            if pred_transcriptions is not None: 
                with open(pred_file, 'w', encoding='utf-8') as f: 
                    for box, transcription in zip(boxes, pred_transcriptions): 
                        line = box.flatten()
                        f.write(','.join(map(str, line)) + f',{transcription}' + '\n')
            else: 
                with open(pred_file, 'w') as f: 
                    for box in boxes: 
                        line = box.flatten()
                        f.write(','.join(map(str, line)) + '\n')

def custom_key(x):
    text = x.split('-')[0]
    number = x.split('-')[-1].split('.')[0]
    try:
        int_number = int(number)
        overlap_number = None
    except ValueError: 
        overlap_number = int(number.split('(')[0])

    if overlap_number is None: return (text, int_number)
    else: return (text, overlap_number)
        
def prepare_file_to_evaluate(file_zip, folder_path): 
    with zipfile.ZipFile(file_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        list_file = [f for f in os.listdir(folder_path)if f.endswith(".txt")]
        
        if Path(folder_path).name == 'vietsignboard': 
            list_file = sorted(list_file, key=custom_key)
        if Path(folder_path).name in ['english', 'vin']: 
            list_file = sorted(list_file, key=lambda x: int(re.findall(r'\d+', x)[0]))
    
        for idx, filename in enumerate(tqdm(list_file)):
            file_path = os.path.join(folder_path, filename)
            new_name = f"res_img_{idx+1}.txt"
            arcname = new_name
    
            zipf.write(file_path, arcname)

def write_txt_yolo(file_path, class_id, x_center, y_center, width, height):
    with open(file_path, 'a') as f: 
        line = f"{class_id} {x_center} {y_center} {width} {height}"
        f.write(line + '\n')

def extract_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Get filename
    file_name = root.find('filename').text
    info_dict['file_name'] = file_name
    # Get image size, ex: [224, 224, 3]
    image_size = [int(size.text) for size in root.find('size')]
    # Convert list to tuple, ex: [224, 224, 3] -> (224, 224, 3)
    image_size = tuple(image_size)
    info_dict['image_size'] = image_size
    # Loop object to extract object's boxes 
    for object in root.iter('object'):
        bbox = {}
        class_name = object.find('name').text
        bbox['class'] = class_name
        for box in object.find('bndbox'):
            bbox[box.tag] = int(box.text)
        info_dict['bboxes'].append(bbox)

    return info_dict

def prepare_yolo_format(info_dict, folder=None):
    # image_width, image_height = info_dict['image_size'][:2]
    image_width, image_height = 1000, 600
    for bbox in info_dict['bboxes']:
        try:
            class_id = class_name_to_id_mapping[bbox['class']]
        except KeyError:
            print(f"Invalid Class. Must be one from {class_name_to_id_mapping.keys()}")

        # Get boxes from info_dict
        boxes = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
        # Convert xyxy format to x_center, y_center, width, height
        boxes = convert_box_format(boxes, current_format_box='xyxy',
                                   output_format_box='yolo')
        # Normalize boxes range [0, 1]
        n_x_center, n_y_center, n_width, n_height = normalize(image_width, image_height, boxes)

        file_path = os.path.join(folder, f"{info_dict['file_name'].split('.')[0]}.txt")
        # Write file txt
        write_txt_yolo(file_path, class_id, n_x_center, n_y_center, n_width, n_height)