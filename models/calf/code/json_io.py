import os
import json
import numpy as np

from SoccerNet.Evaluation.utils import INVERSE_EVENT_DICTIONARY_V2


def predictions2json(predictions_half_1, predictions_half_2, games_ouput_path, game_id, framerate=2):

    current_game_ouput_path = os.path.join(games_ouput_path, game_id)
    os.makedirs(current_game_ouput_path, exist_ok=True)
    if os.path.exists(os.path.join(current_game_ouput_path, "results_spotting.json")):
        return

    frames_half_1, class_half_1 = np.where(predictions_half_1 >= 0)
    frames_half_2, class_half_2 = np.where(predictions_half_2 >= 0)

    json_data = dict()
    json_data["UrlLocal"] = game_id
    json_data["predictions"] = list()

    for frame_index, class_index in zip(frames_half_1, class_half_1):

        confidence = predictions_half_1[frame_index, class_index]

        seconds = int((frame_index//framerate)%60)
        minutes = int((frame_index//framerate)//60)

        prediction_data = dict()
        prediction_data["gameTime"] = str(1) + " - " + str(minutes) + ":" + str(seconds)
        prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_index]
        prediction_data["position"] = str(int((frame_index/framerate)*1000))
        prediction_data["half"] = str(1)
        prediction_data["confidence"] = str(confidence)

        json_data["predictions"].append(prediction_data)

    for frame_index, class_index in zip(frames_half_2, class_half_2):

        confidence = predictions_half_2[frame_index, class_index]

        seconds = int((frame_index//framerate)%60)
        minutes = int((frame_index//framerate)//60)

        prediction_data = dict()
        prediction_data["gameTime"] = str(2) + " - " + str(minutes) + ":" + str(seconds)
        prediction_data["label"] = INVERSE_EVENT_DICTIONARY_V2[class_index]
        prediction_data["position"] = str(int((frame_index/framerate)*1000))
        prediction_data["half"] = str(2)
        prediction_data["confidence"] = str(confidence)

        json_data["predictions"].append(prediction_data)

    with open(os.path.join(current_game_ouput_path, "results_spotting.json"), 'w') as output_file:
        json.dump(json_data, output_file, indent=4)