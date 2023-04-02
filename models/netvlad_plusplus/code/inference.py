import os
import json
import time

import torch
import numpy as np

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2


def model_inference_to_files(dataloader, model, games_ouput_path, nms_window=30, nms_threshold=0.5):
    spotting_groundtruth = list()
    spotting_groundtruth_visibility = list()
    spotting_predictions = list()

    model.eval()
    print('Begin prediction of test data...')

    for game_id, feat_half1, feat_half2, label_half1, label_half2 in dataloader:
        game_id = game_id[0]

        current_game_ouput_path = os.path.join(games_ouput_path, game_id)
        os.makedirs(current_game_ouput_path, exist_ok=True)
        if os.path.exists(os.path.join(current_game_ouput_path, "results_spotting.json")):
            continue

        feat_half1 = feat_half1.squeeze(0)
        label_half1 = label_half1.float().squeeze(0)
        feat_half2 = feat_half2.squeeze(0)
        label_half2 = label_half2.float().squeeze(0)

        # Compute the output for batches of frames
        # start = time.perf_counter()
        batch_size = 256
        timestamp_long_half_1 = []
        for b in range(int(np.ceil(len(feat_half1) / batch_size))):
            start_frame = batch_size * b
            end_frame = batch_size * (b + 1) if batch_size * (b + 1) < len(feat_half1) else len(feat_half1)
            feat = feat_half1[start_frame:end_frame].cuda()
            output = model(feat).cpu().detach().numpy()
            timestamp_long_half_1.append(output)
        timestamp_long_half_1 = np.concatenate(timestamp_long_half_1)

        timestamp_long_half_2 = []
        for b in range(int(np.ceil(len(feat_half2) / batch_size))):
            start_frame = batch_size * b
            end_frame = batch_size * (b + 1) if batch_size * (b + 1) < len(feat_half2) else len(feat_half2)
            feat = feat_half2[start_frame:end_frame].cuda()
            output = model(feat).cpu().detach().numpy()
            timestamp_long_half_2.append(output)
        timestamp_long_half_2 = np.concatenate(timestamp_long_half_2)

        timestamp_long_half_1 = timestamp_long_half_1[:, 1:]
        timestamp_long_half_2 = timestamp_long_half_2[:, 1:]

        # end = time.perf_counter()
        # print(f'time for inference on one video: {end-start:.4f} sec')

        spotting_groundtruth.append(torch.abs(label_half1))
        spotting_groundtruth.append(torch.abs(label_half2))
        spotting_groundtruth_visibility.append(label_half1)
        spotting_groundtruth_visibility.append(label_half2)
        spotting_predictions.append(timestamp_long_half_1)
        spotting_predictions.append(timestamp_long_half_2)

        def get_spot_from_nms(inpt, window=60, thresh=0.0):
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

        framerate = dataloader.dataset.framerate

        json_data = dict()
        json_data["UrlLocal"] = game_id
        json_data["predictions"] = list()

        for half, timestamp in enumerate([timestamp_long_half_1, timestamp_long_half_2]):
            for l in range(dataloader.dataset.num_classes):
                spots = get_spot_from_nms(timestamp[:, l], window=nms_window * framerate, thresh=nms_threshold)
                for spot in spots:
                    frame_index, confidence = int(spot[0]), spot[1]

                    seconds = int((frame_index // framerate) % 60)
                    minutes = int((frame_index // framerate) // 60)

                    prediction_data = dict()
                    prediction_data["gameTime"] = str(half + 1) + " - " + str(minutes) + ":" + str(seconds)
                    prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[l]
                    prediction_data["position"] = str(int(frame_index / framerate * 1000))
                    prediction_data["half"] = str(half + 1)
                    prediction_data["confidence"] = str(confidence)
                    json_data["predictions"].append(prediction_data)

        with open(os.path.join(current_game_ouput_path, "results_spotting.json"), 'w') as output_file:
            json.dump(json_data, output_file, indent=4)