from ultralytics import YOLO
import numpy as np

class YOLOv8_OBB(object): 
    def __init__(self, checkpoint_path, device='cpu', threshold=0.5):
        self.model = YOLO(checkpoint_path)
        self.model = self.model.to(device)
        self.threshold = threshold
    def __call__(self, image):
        assert isinstance(image, np.ndarray)
        preds = self.model(image, verbose=False)[0]
        xywhr = preds.obb.xywhr.cpu().numpy().astype(np.int32)
        ctrl_point = xywhr[:, :2]  # get ctrl point
        boxes = preds.obb.xyxyxyxy
        boxes = boxes.cpu().numpy().astype(np.int32)

        confidences = preds.obb.conf.cpu().numpy()
        indexes = np.where(confidences >= self.threshold)[0]
        new_boxes = boxes[indexes]
        new_ctrl_point = ctrl_point[indexes]

        return new_ctrl_point, new_boxes