from collections import OrderedDict
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
from .transforms import convert_box_format, normalize
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
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    try:
        with open(config_file, "r") as f:
            cfg = yaml.load(f, Loader=yaml.SafeLoader)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {config_file}: {e}")
    except Exception as e:
        raise RuntimeError(f"Cannot read config file {config_file}: {e}")

    return cfg

def calculate_font_scale(image, base_scale=9e-4):
    height, width = image.shape[:2]
    diagonal = (width**2 + height**2)**0.5
    return max(diagonal * base_scale, 0.5) 

def create_folder(folder): 
    if not Path(folder).is_dir():
        Path(folder).mkdir(parents=True, exist_ok=True)

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