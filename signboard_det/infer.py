from utils import draw_bounding_box, calculate_font_scale
import cv2

def plot_image(show_image, results):
    for obj in range(len(results['labels'])):
        boxes = results['bboxes'][obj].detach().cpu().numpy()
        class_id = results['labels'][obj].item() 
        conf = results['scores'][obj].item() 
        font_scale = round(calculate_font_scale(show_image), 2)
        draw_bounding_box(show_image, boxes, class_id, conf, box_thickness=3, 
                          text_thickness=3, text_scale=font_scale, format_box='xyxy', text_color=(0, 165, 255))
    return show_image

def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def inference(model, image_path, transform, device='cpu', threshold=0.5, seed=42):
    image = read_image(image_path)
    transformed_image, _ = transform(image, target=None)
    results = model.predict_one_image(transformed_image.unsqueeze(dim=0), 
                                      original_size=image.shape[:2], 
                                      threshold=threshold)
    show_image = image.copy()
    plot_image(show_image, results)
    cv2.imwrite('image_prediction.jpg', show_image)
