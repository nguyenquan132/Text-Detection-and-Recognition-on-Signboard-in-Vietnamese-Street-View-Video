from e2e_pipeline.modeling import RTDETRV2, YOLOv8_OBB, PARSeq
from .pipeline import TextSpottingonSignboard, TextSpottingonSignboardVideo

def build_pipeline(device, checkpoint_signboard_det=None, checkpoint_text_det=None, 
                   checkpoint_text_rec=None, config_rec_file=None, charset_file=None, 
                   max_text_length=25, signboard_threshold=0.5, text_threshold=0.5): 
    id2label = {0: 'signboard'}
    rtdetr = RTDETRV2(id2label=id2label, checkpoint_path=checkpoint_signboard_det, device=device, threshold=signboard_threshold)
    yolov8_obb_text = YOLOv8_OBB(checkpoint_path=checkpoint_text_det, device=device, threshold=text_threshold)
    parseq = PARSeq(config_rec_file, checkpoint_path=checkpoint_text_rec, charset_file=charset_file, 
                    max_text_length=max_text_length, device=device)

    model = TextSpottingonSignboardVideo(model_det_signboard=rtdetr, model_det_text=yolov8_obb_text, model_rec_text=parseq)

    return model