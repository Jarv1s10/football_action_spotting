import os
import sys
import json
import importlib

import cv2
import torch
import numpy as np
from tqdm import tqdm

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

from data_preprocessing.video_feature_extractor import VideoFeatureExtractor
from models.netvlad_plusplus.code.dataset import feats2clip


def load_model(model_name: str = 'netvlad_plusplus', model_kwargs: dict = {}):
    print('loading model from checkpoing...')
    model = importlib.import_module(f'models.{model_name}.code.model').Model(**model_kwargs)
    checkpoint = torch.load(os.path.join(os.path.dirname(__file__), os.pardir, "models", model_name, "model.pth.tar"))
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model

MODEL = load_model()

def get_spot_from_nms(inpt, window, thresh):
    detections_tmp = np.copy(inpt)
    indexes = []
    max_values = []
    while np.max(detections_tmp) >= thresh:
        # Get the max remaining index and value
        max_value = np.max(detections_tmp)
        max_index = np.argmax(detections_tmp)
        max_values.append(max_value)
        indexes.append(max_index)

        nms_from = int(np.maximum(-(window / 2) + max_index, 0))
        nms_to = int(np.minimum(max_index + int(window / 2), len(detections_tmp)))
        detections_tmp[nms_from:nms_to] = -1

    return np.transpose([indexes, max_values])


def get_predictions(input_video_path: str) -> dict:
    clip_length_seconds = 15
    nms_window_seconds = 30
    framerate = 2
    
    features_output_path=os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'demo', 'features', os.path.basename(input_video_path).split('.')[0])+'.npy'
    
    feature_extractor = VideoFeatureExtractor(load_resnet = True)
    
    video_features = feature_extractor.extract_features(video_input_path=input_video_path, overwrite=True)
    
    clip_features = feats2clip(torch.from_numpy(video_features), stride=1, clip_length=clip_length_seconds*framerate, off=clip_length_seconds)
    output_long = []
    
    batch_size = 256
    
    for batch in range(int(np.ceil(len(clip_features) / batch_size))):
        start_time = batch_size * batch
        end_time = min(batch_size * (batch+1), len(clip_features))
        
        feat = clip_features[start_time:end_time].cuda()
        output_long.append(MODEL(feat).cpu().detach().numpy())
        
    output_long = np.concatenate(output_long)[:, 1:]
    
    predictions = []
    for class_idx, class_name in INVERSE_EVENT_DICTIONARY_V2.items():
        spots = get_spot_from_nms(output_long[:, class_idx], window=nms_window_seconds * framerate, thresh=0.5)
        for spot in spots:
            frame_idx, confidence = int(spot[0]), spot[1]
            
            seconds = int((frame_idx // framerate) % 60)
            minutes = int((frame_idx // framerate) // 60)
            
            predictions.append({
                "class": class_name,
                "seconds": frame_idx // framerate,
                "time": (minutes, seconds),
                "confidence": round(confidence, 4)
            })
    
    return predictions


def render_predictions_on_video(input_video_path: str, predictions: dict) -> str:
    print(f'{input_video_path = }')
    output_video_path = os.path.join(os.path.dirname(input_video_path), os.path.basename(input_video_path).split('.')[0]+'_output'+'.mp4')
    
    vid = cv2.VideoCapture(input_video_path)

    framerate = vid.get(cv2.CAP_PROP_FPS)
    frame_count = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(output_video_path, fourcc, framerate, (width, height))
    
    print('adding predictions to video...')
    
    for frame_idx in tqdm(list(range(frame_count))):
        ret, frame = vid.read()
        
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        seconds = frame_idx // framerate
        pred_frame = [pred for pred in predictions if 0 <= pred['seconds'] - seconds <= 10]

        for i, pred in enumerate(pred_frame, start=1):
            time = ':'.join(map(lambda x: str(x).zfill(2), pred['time']))
            text = f"{pred['class']}, time: {time}, confidence: {str(pred['confidence'])}"
            
            cv2.putText(frame, 
                        text, 
                        (50, height-50*i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 255), 
                        2, 
                        cv2.LINE_4)
        cv2.putText(frame, 
                    ':'.join(map(lambda x: str(int(x)).zfill(2), divmod(seconds, 60))),
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (100, 100, 100), 
                    2, 
                    cv2.LINE_4)
        cv2.imshow('Predictions', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        out.write(frame)
        
    vid.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_video_path
