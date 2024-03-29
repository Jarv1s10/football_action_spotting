import os
import json
import random

from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

LABELS_FILENAME = "Labels-v2.json"

def clips_iou(clip1, clip2):
    left_max = max(clip1[0], clip2[0])
    right_min = min(clip1[1], clip2[1])
    intersection = right_min - left_max

    left_min = min(clip1[0], clip2[0])
    right_max = max(clip1[1], clip2[1])
    union = right_max - left_min

    return intersection / union


def load_events(matches, labels_path, frames_per_clip, fps):
    events = []
    for m in matches:
        labels = json.load(open(os.path.join(labels_path, m, LABELS_FILENAME), "r"))
        for i, k in enumerate(labels["annotations"]):
            frame_indexes = int(k["gameTime"][-5:-3])*60 * fps + int(k["gameTime"][-2:]) * fps + 1
            events.append({"match": m, "half": int(k["gameTime"][0]), "frame_indexes": (frame_indexes-int(frames_per_clip/2), frame_indexes+int(frames_per_clip/2)+1),
                           "label": k["label"], "rel_offset": 0.5, "team": k["team"], "visibility": k["visibility"]})
    return events

######################################################### TRAIN ##########################################################

def load_background_train(n_frames_per_half_match, events, frames_per_clip):
    background_events = []

    for i in range(len(events) - 1):
        if events[i]["half"] != events[i+1]["half"]:  # sample background clips between two events of different halves
            for j in range(events[i]["frame_indexes"][1] + frames_per_clip, n_frames_per_half_match[events[i]["match"]+"_"+str(events[i]["half"])] - frames_per_clip, frames_per_clip):
                background_events.append({"match":events[i]["match"], "half":events[i]["half"], "frame_indexes":[j-int(frames_per_clip/2), j+int(frames_per_clip/2)+1], "label":"Background", "rel_offset":round(random.uniform(0.1, 0.9), 1), "team":"None", "visibility":"None"})
            for j in range(frames_per_clip, events[i+1]["frame_indexes"][0] - frames_per_clip, frames_per_clip):
                background_events.append({"match":events[i+1]["match"], "half":events[i+1]["half"], "frame_indexes":[j-int(frames_per_clip/2), j+int(frames_per_clip/2)+1], "label":"Background", "rel_offset":round(random.uniform(0.1, 0.9), 1), "team":"None", "visibility":"None"})

        if events[i]["match"] == events[i+1]["match"] and events[i]["half"] == events[i+1]["half"]:  # sample background clips between two events of the same half
            for j in range(events[i]["frame_indexes"][1] + frames_per_clip, events[i+1]["frame_indexes"][0] - frames_per_clip, frames_per_clip):
                background_events.append({"match":events[i]["match"], "half":events[i]["half"], "frame_indexes":[j-int(frames_per_clip/2), j+int(frames_per_clip/2)+1], "label":"Background", "rel_offset":round(random.uniform(0.1, 0.9), 1), "team":"None", "visibility":"None"})
    return background_events


def sample_training(events, how_many):
    new_events = []
    for k in events:
        new_events = new_events + random.sample(events[k], how_many if len(events[k])>=how_many else len(events[k]))
    return new_events


def augment_training(events, frames_per_clip):
    new_events = {label: list() for label in EVENT_DICTIONARY_V2}
    possible_offsets = range(-int((frames_per_clip-1)/2), int((frames_per_clip-1)/2)+1)

    for e in events:
        for of in possible_offsets:
            new_e = e.copy()
            new_e["frame_indexes"] = [new_e["frame_indexes"][0] + of, new_e["frame_indexes"][1] + of]
            new_e["rel_offset"] = new_e["rel_offset"] - (of / frames_per_clip)
            new_events[new_e["label"]].append(new_e)
    return new_events


def sample_background(events, how_many):
    return random.sample(events, how_many)

######################################################### VAL ##########################################################

def load_background_val(n_frames_per_half_match, events, frames_per_clip, overlap):
    background_events = []

    for half_match, n_frames in n_frames_per_half_match.items():
        for i in range(int(frames_per_clip/2), n_frames+1, int(frames_per_clip*(1-overlap))):
            background_events.append({"match": half_match[:-2], "half": int(half_match[-1]), "frame_indexes": [i-int(frames_per_clip/2), i+int(frames_per_clip/2)+1], "label": "Background", "rel_offset": 0.5, "team": "None", "visibility": "None"})

    e_index=0
    new_events = []
    for i, b in enumerate(background_events[:-1]):
        if e_index<len(events):
            e = events[e_index]
            if b["frame_indexes"][0] <= e["frame_indexes"][0]+int(frames_per_clip/2) <= b["frame_indexes"][1]:
                of = b["frame_indexes"][0] - e["frame_indexes"][0]
                new_e = {"match": e["match"], "half": e["half"], "frame_indexes": b["frame_indexes"], "label": e["label"], "rel_offset": 0.5 - (of / frames_per_clip), "team": e["team"], "visibility": e["visibility"]}
                new_events.append(new_e)
                e_index += 1
                if e_index<len(events):
                    e = events[e_index]
                    while b["frame_indexes"][0] <= e["frame_indexes"][0] + int(frames_per_clip / 2) <= b["frame_indexes"][1] and not background_events[i+1]["frame_indexes"][0] <= e["frame_indexes"][0] + int(frames_per_clip / 2) <= background_events[i+1]["frame_indexes"][1]:
                        e_index += 1
                        if e_index>=len(events):
                            break
                        e = events[e_index]
            else:
                new_events.append(b)
        else:
            new_events.append(b)

    return new_events
