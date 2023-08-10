import os
import json
import sys

import torch

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2, INVERSE_EVENT_DICTIONARY_V2

EVENT_DICTIONARY_V2 = dict(EVENT_DICTIONARY_V2)
INVERSE_EVENT_DICTIONARY_V2 = dict(INVERSE_EVENT_DICTIONARY_V2)

EVENT_DICTIONARY_V2['Background'] = 17
INVERSE_EVENT_DICTIONARY_V2[17] = 'Background'


def nms(preds, nms_window_ms):
    preds_after_nms = {}
    for m, p in preds.items():
        if m not in preds_after_nms:
            preds_after_nms[m] = {"UrlLocal": m, "predictions": []}

        cur_label = "start"
        cur_time = "0"
        cur_score = "0"
        cur_gameTime = ""
        cur_half = ""

        for i, pred in enumerate(p["predictions"]):
            if pred["label"] == cur_label and (int(pred["position"]) - int(cur_time)) < nms_window_ms and float(pred["confidence"]) > float(cur_score) and i > 0:
                preds_after_nms[m]["predictions"].remove({"gameTime": cur_gameTime, "label": cur_label, "position": cur_time, "confidence": cur_score, "half": cur_half})
                cur_label = pred["label"]
                cur_time = pred["position"]
                cur_score = pred["confidence"]
                cur_gameTime = pred["gameTime"]
                cur_half = pred["half"]
                preds_after_nms[m]["predictions"].append(pred)
            elif pred["label"] == cur_label and (int(pred["position"]) - int(cur_time)) < nms_window_ms and float(pred["confidence"]) <= float(cur_score) and i > 0:
                continue
            else:
                cur_label = pred["label"]
                cur_time = pred["position"]
                cur_score = pred["confidence"]
                cur_gameTime = pred["gameTime"]
                cur_half = pred["half"]
                preds_after_nms[m]["predictions"].append(pred)

    return preds_after_nms


def standard_nms(preds, nms_window_ms):
    preds_after_nms = {}
    for m, p in preds.items():
        if m not in preds_after_nms:
            preds_after_nms[m] = {"UrlLocal": m, "predictions": []}

        p["predictions"].sort(key = lambda event: (event["label"], event["confidence"]), reverse=True)

        for i, pred in enumerate(p["predictions"]):
            if i == 0:
                preds_after_nms[m]["predictions"].append(pred)
            else:
                for stored in preds_after_nms[m]["predictions"]:  # all the stored events have an higher score than the current pred (see the sort function!)
                    if pred["half"] == stored["half"] and pred["label"] == stored["label"] and abs(int(pred["position"]) - int(stored["position"])) < nms_window_ms:
                        break
                else:
                    preds_after_nms[m]["predictions"].append(pred)

    return preds_after_nms


class JsonHandler(object):
    def __init__(self, out_dir):
        self.preds = dict()
        self.preds_after_nms = dict()
        self.out_dir = out_dir

    def update_preds(self, match, half,  classes, start_frame, scores, t_shift, frames_per_clip, fps=2):
        for m, h, c, f, s, t in zip(match, half, classes, start_frame, scores, t_shift):
            spot = f+t*frames_per_clip
            if m not in self.preds.keys():
                self.preds[m] = {"UrlLocal": m, "predictions": []}

            self.preds[m]["predictions"].append({"gameTime":str(h.item())+" - " + str(int(spot/60*fps)).zfill(2) + ":" + str(int((spot%60*fps)/2)).zfill(2),
                                                 "label":INVERSE_EVENT_DICTIONARY_V2[int(c)],
                                                 "position":str(int(spot/fps*1000)),
                                                 "half":str(h.item()),
                                                 "confidence":str(s.item())})
    def reset(self):
        self.preds = dict()
        self.preds_after_nms = dict()

    def apply_nms(self, nms_mode, nms_window_ms):
        if nms_mode=="new":
            self.preds_after_nms = nms(self.preds, nms_window_ms)
        elif nms_mode=="standard":
            self.preds_after_nms = standard_nms(self.preds, nms_window_ms)
        else:
            print("Error: invalid NMS mode specified, must be 'standard' or 'new'")

    def save_json(self):
        os.makedirs(self.out_dir, exist_ok=True)

        print("Saving prediction json...")
        for game_id, pred in self.preds_after_nms.items():
            os.makedirs(os.path.join(self.out_dir, game_id), exist_ok=True)

            with open(os.path.join(self.out_dir, game_id, "results_spotting.json"), "w") as outfile:
                json.dump(pred, outfile)


def model_inference_to_files(dataloader, model, games_ouput_path, nms_window_ms, device='cuda'):
    model.eval()

    json_handler = JsonHandler(games_ouput_path)

    with torch.no_grad():
        for features, label, rel_offset, game_id, half, start_frame in dataloader:
            features = features.to(device)
            label = label.to(device)
            rel_offset = rel_offset.to(device)

            out, pred_rel_offset = model(features)
            pred_rel_offset = pred_rel_offset.squeeze(1)
            score, cls = torch.max(out, dim=-1)

            non_background_indexes_predicted = (cls != EVENT_DICTIONARY_V2["Background"])
            score = score[non_background_indexes_predicted]
            half = [hm for i, hm in enumerate(half) if non_background_indexes_predicted[i]]
            start_frame = start_frame[non_background_indexes_predicted]
            cls = cls[non_background_indexes_predicted]
            game_id = [m for i, m in enumerate(game_id) if non_background_indexes_predicted[i]]

            if len(score) != 0:
                json_handler.update_preds(game_id, half, cls, start_frame, score, pred_rel_offset[non_background_indexes_predicted], dataloader.dataset.frames_per_clip)

        json_handler.apply_nms('standard', nms_window_ms)
        json_handler.save_json(dataloader.dataset.split)