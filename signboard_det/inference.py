from utils.visualize import overlay, draw_bounding_box
from utils.utility import set_seed, calculate_font_scale
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

def inference(model, image_path, transforms, feature_extractor=None, mask=False, device='cpu', threshold=0.5, seed=42):
    set_seed(seed)
    image = read_image(image_path)
    pixel_values, _ = transforms(image, target=None)
    if feature_extractor is not None: 
        encoded_inputs = feature_extractor(pixel_values, return_tensors='pt')
        for k, v in encoded_inputs.items(): 
            encoded_inputs[k].squeeze_() 

        pixel_values = encoded_inputs['pixel_values']

    # Prediction
    model.eval() 
    model = model.to(device)
    if mask: 
        predicted_mask = model.predict_one_image(pixel_values.unsqueeze(dim=0),
                                                 target_size=image.shape[:2])
        show_image = overlay(image, predicted_mask.squeeze(), (0, 255, 0), alpha=0.7)
    else: 
        results = model.predict_one_image(pixel_values.unsqueeze(dim=0), 
                                        original_size=image.shape[:2], 
                                        threshold=threshold)
        show_image = image.copy()
        plot_image(show_image, results)
    # Save annotated image
    cv2.imwrite('image_prediction.jpg', show_image)
