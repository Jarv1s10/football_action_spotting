import os
import json

import torch
import numpy as np
from tqdm import tqdm

from metrics_visibility_fast import NMS
from preprocessing import batch2long, timestamps2long
from json_io import predictions2json
from SoccerNet.Downloader import getListGames


def model_inference_to_files(dataloader, model, games_ouput_path):
    spotting_grountruth = list()
    spotting_grountruth_visibility = list()
    spotting_predictions = list()
    segmentation_predictions = list()

    chunk_size = model.chunk_size
    receptive_field = model.receptive_field

    model.eval()

    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120) as t:
        for i, (feat_half1, feat_half2, label_half1, label_half2) in t:

            feat_half1 = feat_half1.cuda().squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.cuda().squeeze(0)
            label_half2 = label_half2.float().squeeze(0)

            feat_half1=feat_half1.unsqueeze(1)
            feat_half2=feat_half2.unsqueeze(1)

            # Compute the output
            output_segmentation_half_1, output_spotting_half_1 = model(feat_half1)
            output_segmentation_half_2, output_spotting_half_2 = model(feat_half2)

            timestamp_long_half_1 = timestamps2long(output_spotting_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            timestamp_long_half_2 = timestamps2long(output_spotting_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)
            segmentation_long_half_1 = batch2long(output_segmentation_half_1.cpu().detach(), label_half1.size()[0], chunk_size, receptive_field)
            segmentation_long_half_2 = batch2long(output_segmentation_half_2.cpu().detach(), label_half2.size()[0], chunk_size, receptive_field)

            spotting_grountruth.append(torch.abs(label_half1))
            spotting_grountruth.append(torch.abs(label_half2))
            spotting_grountruth_visibility.append(label_half1)
            spotting_grountruth_visibility.append(label_half2)
            spotting_predictions.append(timestamp_long_half_1)
            spotting_predictions.append(timestamp_long_half_2)
            segmentation_predictions.append(segmentation_long_half_1)
            segmentation_predictions.append(segmentation_long_half_2)

    # Transformation to numpy for evaluation
    targets_numpy = list()
    closests_numpy = list()
    detections_numpy = list()
    for target, detection in zip(spotting_grountruth_visibility,spotting_predictions):
        target_numpy = target.numpy()
        targets_numpy.append(target_numpy)
        detections_numpy.append(NMS(detection.numpy(), 20*model.framerate))
        closest_numpy = np.zeros(target_numpy.shape)-1
        #Get the closest action index
        for c in np.arange(target_numpy.shape[-1]):
            indexes = np.where(target_numpy[:,c] != 0)[0].tolist()
            if len(indexes) == 0 :
                continue
            indexes.insert(0,-indexes[0])
            indexes.append(2*closest_numpy.shape[0])
            for i in np.arange(len(indexes)-2)+1:
                start = max(0,(indexes[i-1]+indexes[i])//2)
                stop = min(closest_numpy.shape[0], (indexes[i]+indexes[i+1])//2)
                closest_numpy[start:stop,c] = target_numpy[indexes[i],c]
        closests_numpy.append(closest_numpy)

# Save the predictions to the json format
    list_game = getListGames(dataloader.dataset.split)
    for index in np.arange(len(list_game)):
        predictions2json(detections_numpy[index*2], detections_numpy[(index*2)+1], games_ouput_path, list_game[index], model.framerate)