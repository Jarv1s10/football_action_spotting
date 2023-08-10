import os
import numpy as np
import pickle as pkl

import tensorflow as tf
from SoccerNet.DataLoader import FrameCV


class VideoFeatureExtractor:
    def __init__(self, load_resnet=True) -> None:
        
        if load_resnet:
            print('loading ResNet152...')
            resnet152 = tf.keras.applications.resnet.ResNet152(include_top=True, weights='imagenet', 
                                                            input_tensor=None, input_shape=None,
                                                            pooling=None, classes=1000)
            
            self.model = tf.keras.models.Model(resnet152.input, 
                                            outputs=[resnet152.get_layer("avg_pool").output])
            self.model.trainable = False
        
        with open(os.path.join(os.path.dirname(__file__), 'pca_512_TF2.pkl'), "rb") as f:
            self.pca = pkl.load(f)
            
        with open(os.path.join(os.path.dirname(__file__), 'average_512_TF2.pkl'), "rb") as f:
            self.average = pkl.load(f)
        
        
    def resnet_features(self, video_input_path: str):
        print('subsampling frames...')
        video_loader = FrameCV(video_input_path, transform='crop', FPS=2)
        
        print('getting ResNet features...')
        frames = tf.keras.applications.resnet.preprocess_input(video_loader.frames)
        features = self.model.predict(frames, batch_size=64, verbose=1)
        
        return features
    
        
    def pca_reduction(self, input_features: np.array):
        return self.pca.transform(input_features - self.average)
    
        
    def extract_features(self, video_input_path, features_output_path=None, overwrite=False):
        if features_output_path and os.path.exists(features_output_path) and not overwrite:
            features = np.load(features_output_path)
        else:
            features = self.resnet_features(video_input_path)
            
        if features.shape[-1] != 512:
            features = self.pca_reduction(features)
            
        if features_output_path:
            np.save(features_output_path, features)      
            
        return features
    