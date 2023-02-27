import os
import torch
import numpy as np
import json
import zipfile

from SoccerNet.Evaluation.ActionSpotting import evaluate
from SoccerNet.Evaluation.utils import AverageMeter, INVERSE_EVENT_DICTIONARY_V2


def test_spotting(dataloader, model, model_name, nms_window=30, nms_threshold=0.5):
    split = '_'.join(dataloader.dataset.split)
    output_results = os.path.join("models", model_name, f"results_spotting_{split}.zip")
    output_folder = f"outputs_{split}"

    if not os.path.exists(output_results):
        spotting_grountruth = list()
        spotting_grountruth_visibility = list()
        spotting_predictions = list()

        model.eval()
        print('Begin prediction of test data...')
        for game_id, feat_half1, feat_half2, label_half1, label_half2 in dataloader:

            # Batch size of 1
            game_id = game_id[0]
            feat_half1 = feat_half1.squeeze(0)
            label_half1 = label_half1.float().squeeze(0)
            feat_half2 = feat_half2.squeeze(0)
            label_half2 = label_half2.float().squeeze(0)

            # Compute the output for batches of frames
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

            spotting_grountruth.append(torch.abs(label_half1))
            spotting_grountruth.append(torch.abs(label_half2))
            spotting_grountruth_visibility.append(label_half1)
            spotting_grountruth_visibility.append(label_half2)
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
                        prediction_data["position"] = str(int((frame_index / framerate) * 1000))
                        prediction_data["half"] = str(half + 1)
                        prediction_data["confidence"] = str(confidence)
                        json_data["predictions"].append(prediction_data)

            os.makedirs(os.path.join("models", model_name, output_folder, game_id), exist_ok=True)
            with open(os.path.join("models", model_name, output_folder, game_id, "results_spotting.json"),
                      'w') as output_file:
                json.dump(json_data, output_file, indent=4)

        def zip_results(zip_path, target_dir, filename="results_spotting.json"):
            zipobj = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
            rootlen = len(target_dir) + 1
            for base, dirs, files in os.walk(target_dir):
                for file in files:
                    if file == filename:
                        fn = os.path.join(base, file)
                        zipobj.write(fn, fn[rootlen:])

        print('Zip json files with predicted actions...')
        # zip folder
        zip_results(zip_path=output_results, target_dir=os.path.join("models", model_name, output_folder),
                    filename="results_spotting.json")

    print('Calculate mAP metric for test predictions...')
    print(__file__)
    current_path = os.path.normpath(__file__)
    current_path = current_path.split(os.sep)[:-1]
    current_path.append(output_results)
    output_results_path = os.sep.join(current_path)

    results = evaluate(SoccerNet_path=dataloader.dataset.path,
                       Predictions_path=output_results_path,
                       split="test",
                       prediction_file="results_spotting.json")

    return results
