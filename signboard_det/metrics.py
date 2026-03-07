from signboard_det.evaluation.BoundingBox import BoundingBox
from signboard_det.evaluation.Evaluator import *
from utils.transforms import convert_box_format, denormalize
import torch
import time
from fvcore.nn import FlopCountAnalysis, parameter_count

def bounding_boxes(allBoundingBoxes, image_id, gt, pred, current_gtbox_format, current_predbox_format, id2label, pixel_values): 
    file = f'image_{image_id}'
    for obj in range(len(gt['class_labels'])):
        class_name = id2label[int(gt['class_labels'][obj].item())]
        boxes = denormalize(pixel_values.shape[-1], pixel_values.shape[-2], gt['boxes'][obj])
        if current_gtbox_format != 'xywh':
            [x, y, w, h] = convert_box_format(boxes, current_format_box=current_gtbox_format, 
                                              output_format_box='xywh')
        else: 
            [x, y, w, h] = gt['boxes'][obj]
        
        bb = BoundingBox(file,
                         class_name,
                         float(x), float(y), float(w), float(h),
                         typeCoordinates=CoordinatesType.Absolute, 
                         bbType=BBType.GroundTruth,
                        format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)
    
    if isinstance(pred, torch.Tensor): 
        pred = {k: v.detach().cpu().numpy() for k, v in pred.items()}
    for obj in range(len(pred['labels'])):
        class_name, conf = id2label[int(pred['labels'][obj].item())], pred['scores'][obj].item()
        if current_predbox_format != 'xywh':
            [x, y, w, h] = convert_box_format(pred['bboxes'][obj].cpu().numpy(), 
                                              current_format_box=current_predbox_format, 
                                              output_format_box='xywh')
        else: 
            [x, y, w, h] = pred['bboxes'][obj].cpu().numpy()
        
        bb = BoundingBox(file,
                         class_name,
                         x, y, w, h,
                         typeCoordinates=CoordinatesType.Absolute,
                         bbType=BBType.Detected,
                         classConfidence=conf,
                        format=BBFormat.XYWH)
        allBoundingBoxes.addBoundingBox(bb)

    return allBoundingBoxes

def calculate_mAP(evaluator, allBoundingBoxes, iou_threshold=0.5): 
    detections = evaluator.GetPascalVOCMetrics(allBoundingBoxes,
                                               IOUThreshold=iou_threshold)
    acc_AP, validClasses = 0, 0
    total_TP, total_FP = 0, 0
    
    for metricsPerClass in detections:
        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        # print(precision)
        recall = metricsPerClass['recall']
        # print(recall)
        totalPositives = metricsPerClass['total positives']
        tp = metricsPerClass['total TP']
        fp = metricsPerClass['total FP']
        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            # print('AP: %s (%s)' % (ap_str, cl))
        total_TP += tp
        total_FP += fp
    
    mAP = acc_AP / validClasses
    return mAP

def measure_fps(model, image, num_runs=100):    
    model.eval()
    # Warm-up
    with torch.inference_mode():
        for _ in range(10):
            _ = model(image)
    
    # Measure FPS
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.inference_mode():
        for _ in range(num_runs):
            _ = model(image)
            
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs
    fps = 1 / avg_time
    
    print(f"Average time per frame: {avg_time*1000:.2f} ms")
    
    return fps

def measure_flops(model, image): 
    flops = FlopCountAnalysis(model, image).total() / 1e9
    return flops

def measure_params(model): 
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    total = trainable + non_trainable

    return total, trainable, non_trainable