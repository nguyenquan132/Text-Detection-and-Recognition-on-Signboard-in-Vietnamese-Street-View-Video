import cv2
import time
import argparse
import torch
from e2e_pipeline import build_pipeline
import warnings
warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated",
    category=UserWarning
)

def infer_video(video_path, pipeline): 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 30: 
        target_fps = 30
        skip = max(1, int(round(fps / target_fps)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_delay = int(1000 / fps)
    # Set time to save frame and label
    save_interval = 2.0 
    last_saved_time = -save_interval
    
    # Check if the video opened correctly
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Save video prediction
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('video_prediction.avi', fourcc, fps=30, frameSize=(width, height))
    frame_idx = 0
    while True: 
        start_time = time.time() 
        ret, frame = cap.read() 

        if not ret: 
            break # No more frame

        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Inference
        new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tracked_signboards = pipeline(new_frame)
        # Visualize
        annotated_frame = pipeline.visualize(frame, tracked_signboards)
        out.write(annotated_frame)  
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('-video_path', type=str, default=None, required=True, help='Path to the input video used for inference')
    parser.add_argument('-best_rtdetrv2_path', type=str, default=None, required=True, help='Path to the trained RT-DETRv2 model used for signboard detection')
    parser.add_argument('-best_yolov8_obb_path', type=str, default=None, required=True, help='Path to the trained YOLOv8-OBB model used for oriented text detection')
    parser.add_argument('-best_parseq_path', type=str, default=None, required=True, help='Path to the trained PARSeq model used for text recognition')
    parser.add_argument('-signboard_threshold', type=float, default=0.1, required=False, help='Confidence threshold used to filter signboard detections for tracking (default: 0.1)')
    args = parser.parse_args()

    # get video_path and device
    video_path = args.video_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # initialize end-to-end pipeline 
    checkpoint_signboard_det = args.best_rtdetrv2_path
    checkpoint_text_det = args.best_yolov8_obb_path
    checkpoint_text_rec = args.best_parseq_path
    signboard_threshold = args.signboard_threshold

    pipeline = build_pipeline(device=device, 
                              checkpoint_signboard_det=checkpoint_signboard_det, 
                              checkpoint_text_det=checkpoint_text_det, 
                              checkpoint_text_rec=checkpoint_text_rec, 
                              config_rec_file='src/text_rec/OpenOCR/configs/rec/parseq/vit_parseq.yml', 
                              charset_file='charset/vietnamese_charset.txt', 
                              signboard_threshold=signboard_threshold)
    # inference video
    infer_video(video_path, pipeline=pipeline)