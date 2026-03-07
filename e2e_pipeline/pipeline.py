import numpy as np
from PIL import Image
import cv2
import supervision as sv
from collections import defaultdict
from quadrilateral_fitter import QuadrilateralFitter
from utils.visualize import TextVisualizer, draw_bounding_box, draw_polygon
from utils.transforms import check_and_refine_valid_box
from utils.geometry import smart_crop_or_align, transform_boxes_back

class TextSpottingonSignboardVideo(object): 
    def __init__(self, model_det_signboard, model_det_text, model_rec_text):
        self.model_det_signboard = model_det_signboard
        self.model_det_text = model_det_text
        self.model_rec_text = model_rec_text
        self.tracker = sv.ByteTrack()
        self.signboard_data = defaultdict(dict)

    def polygon_to_xyxy(self, polygon):
        x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
        x1, y1, x2, y2 = x, y, x + w, y + h
        return np.array([x1, y1, x2, y2])
    
    def filter_and_refine_box(self, bboxes, confidences, image_size, min_width, min_height, box_format='xywh'): 
        assert box_format in ['xywh', 'xyxy']
        image_width, image_height = image_size 
        new_bboxes, new_conf = [], []
        for box, conf in zip(bboxes, confidences): 
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
                continue  # Invalid bbox

            new_bboxes.append([intersect_x1, intersect_y1, intersect_x2, intersect_y2])
            new_conf.append(conf)
        
        return new_bboxes, new_conf

    def detect_signboard(self, image):
        # RTDETRv2
        results = self.model_det_signboard(image)
        bboxes = results['bboxes'].cpu().numpy() 
        confidences = results['scores'].cpu().numpy() 
        return bboxes, confidences
    def detect_text(self, image):
        ctrl_points, boxes = self.model_det_text(image)
        return ctrl_points, boxes

    def recognize_text(self, image): 
        text, score = self.model_rec_text(image)
        return text, score
    
    def __call__(self, image):
        assert isinstance(image, np.ndarray)
        orig_size = Image.fromarray(image).size
        signboard_bboxes, signboard_conf = self.detect_signboard(image)

        new_signboard_bboxes, new_signboard_conf = self.filter_and_refine_box(signboard_bboxes, signboard_conf, orig_size, 
                                                                              min_width=12, min_height=6, box_format='xyxy')
        
        if len(new_signboard_bboxes) == 0:
            signboard_detections = sv.Detections.empty()
        else: 
            # Tracking
            signboard_detections = sv.Detections(
                xyxy=np.array(new_signboard_bboxes),
                confidence=np.array(new_signboard_conf)
            )

        tracked_signboards = self.tracker.update_with_detections(signboard_detections)

        for i in range(len(tracked_signboards)): 
            signboard_id = tracked_signboards.tracker_id[i]
            x1, y1, x2, y2 = tracked_signboards.xyxy[i].astype(int)
            text_infos = {
                int(signboard_id): {
                    'polygons': [],
                    'ctrl_pnts': [],
                    'text': [],
                    'scores': []
                }
            }

            signboard_image = image[y1:y2, x1:x2]
            ctrl_points, polygons = self.detect_text(signboard_image)


            for i, (ctrl_pnt, poly) in enumerate(zip(ctrl_points, polygons)): 
                x1_text, y1_text, x2_text, y2_text = self.polygon_to_xyxy(poly)
                new_bbox = check_and_refine_valid_box([x1_text, y1_text, x2_text, y2_text], Image.fromarray(signboard_image).size, 
                                                      min_width=5, min_height=3, box_format='xyxy')
                if new_bbox is None: 
                    text, score = '###', 1.00
                else:  
                    x_text, y_text, w_text, h_text = new_bbox
                    word_image = signboard_image[y_text:y_text+h_text:1, x_text:x_text+w_text:1]
                    # word_image = aligned[y_text:y_text+h_text:1, x_text:x_text+w_text:1]
                    text, score = self.recognize_text(word_image)

                # convert poly to orig image
                poly[:, 0] = poly[:, 0] + x1
                poly[:, 1] = poly[:, 1] + y1
                
                text_infos[signboard_id]['polygons'].append(poly.astype(np.int32))
                text_infos[signboard_id]['ctrl_pnts'].append(ctrl_pnt)
                text_infos[signboard_id]['text'].append(text)
                text_infos[signboard_id]['scores'].append(score)

            tracked_signboards.data.update(text_infos)

        return tracked_signboards
    
    def visualize(self, image, tracked_signboards=None, vis_threshold=0.5): 
        visualizer = TextVisualizer()
        visualized_image = image.copy()
        if len(tracked_signboards) > 0: 
            for i in range(len(tracked_signboards)):
                signboard_conf = tracked_signboards.confidence[i]
                if signboard_conf >= vis_threshold:
                    signboard_box = tracked_signboards.xyxy[i]
                    signboard_id = tracked_signboards.tracker_id[i]
                    if signboard_id in tracked_signboards.data.keys():
                        results = tracked_signboards.data[signboard_id]
                        draw_bounding_box(visualized_image, signboard_box, box_thickness=3, format_box='xyxy', box_color=(0, 255, 0))
                        vis = visualizer.draw_instance_predictions(visualized_image, results)
                        visualized_image = vis.get_image()
                    else: 
                        draw_bounding_box(visualized_image, signboard_box, box_thickness=3, format_box='xyxy', box_color=(0, 255, 255), class_id=0)

        return visualized_image
    
# inference with image
class TextSpottingonSignboard(object): 
    def __init__(self, model_det_signboard, model_det_text, model_rec_text):
        self.model_det_signboard = model_det_signboard
        self.model_det_text = model_det_text
        self.model_rec_text = model_rec_text

    def detect_signboard(self, image):
        # RTDETRv2
        results = self.model_det_signboard(image)
        return results['bboxes'].cpu().numpy()
        # SegFormer
        # results = self.model_det_signboard(image)
        # boxes = results['polygons']
        # return boxes

    def detect_text(self, image):
        ctrl_points, boxes = self.model_det_text(image)
        return ctrl_points, boxes

    def recognize_text(self, image): 
        text, score = self.model_rec_text(image)
        return text, score
        
    def __call__(self, image):
        assert isinstance(image, np.ndarray)
        orig_size = Image.fromarray(image).size
        signboard_bboxes = self.detect_signboard(image)
        results = {
            'signboard_boxes': [],
            'ctrl_pnts': [],
            'polygons': [],
            'text': [],
            'scores': []
        }
        for signboard_box in signboard_bboxes: 
            # Segformer or YOLOv8-OBB
            # x, y, w, h = cv2.boundingRect(signboard_box.astype(np.int32))
            # RTDETRv2
            new_signboard_box = check_and_refine_valid_box(signboard_box.astype(np.int32), orig_size, 
                                                           min_width=12, min_height=6, box_format='xyxy')
            # YOLOv8-OBB
            # new_signboard_box = check_and_refine_valid_box(cv2.boundingRect(signboard_box.astype(np.int32)), orig_size, 
            #                                                min_width=12, min_height=6, box_format='xywh')
            # new_signboard_box = check_and_refine_polygon(signboard_box, orig_size, 
            #                                             min_width=12, min_height=6)
            if new_signboard_box is None:
                continue

            # Segformer or YOLOv8-OBB
            # x, y, w, h =  cv2.boundingRect(signboard_box.astype(np.int32))
            x, y, w, h = new_signboard_box
            signboard_image = image[y:y+h:1, x:x+w:1]
            results['signboard_boxes'].append(signboard_box)
            # # Aligned Signboard for Segformer or YOLOv8-obb
            # quadrilateral = np.array(QuadrilateralFitter(polygon=new_signboard_box).fit(), dtype=np.float32)
            # aligned, M_inv = smart_crop_or_align(image, quadrilateral, new_signboard_box)
            # results['signboard_boxes'].append(quadrilateral)

            # Detect text on signboard
            ctrl_points, polygons = self.detect_text(signboard_image)
            # Segformer or YOLOv8-OBB
            # ctrl_points, polygons = self.detect_text(aligned)
            # new_poly = transform_boxes_back(polygons, M_inv)
            
            for i, (ctrl_pnt, poly) in enumerate(zip(ctrl_points, polygons)): 
                x_text, y_text, w_text, h_text = cv2.boundingRect(poly.astype(np.int32))
                new_bbox = check_and_refine_valid_box([x_text, y_text, w_text, h_text], Image.fromarray(signboard_image).size, 
                                                      min_width=5, min_height=3, box_format='xywh')
                # Segformer or YOLOv8-OBB
                # new_bbox = check_and_refine_valid_box([x_text, y_text, w_text, h_text], Image.fromarray(aligned).size, 
                #                                       min_width=5, min_height=3, box_format='xywh')
                if new_bbox is None: 
                    text, score = '###', 1.00
                else:  
                    x_text, y_text, w_text, h_text = new_bbox
                    word_image = signboard_image[y_text:y_text+h_text:1, x_text:x_text+w_text:1]
                    # Segformer or YOLOv8-OBB
                    # word_image = aligned[y_text:y_text+h_text:1, x_text:x_text+w_text:1]
                    text, score = self.recognize_text(word_image)

                results['ctrl_pnts'].append (ctrl_pnt)
                # convert poly to orig image
                poly[:, 0] = poly[:, 0] + x
                poly[:, 1] = poly[:, 1] + y
                results['polygons'].append(poly.astype(np.int32))
                # Segformer or YOLOv8-OBB
                # Transform polygons to original
                # results['polygons'].append(new_poly[i].astype(np.int32))
                results['text'].append(text)
                results['scores'].append(score)

        return results