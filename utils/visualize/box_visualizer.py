import numpy as np
import cv2
from .transforms import convert_box_format, normalize, denormalize
class_name_to_id_mapping = {'signboard': 0}
class_id_to_name_mapping = {0: 'signboard'}

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