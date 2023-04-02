import os
import random
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange, repeat

from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

EVENT_DICTIONARY_V2 = dict(EVENT_DICTIONARY_V2)
EVENT_DICTIONARY_V2['Background'] = 17

from dataset_utils import load_events, load_background_train, sample_training, sample_background, load_background_val, augment_training, LABELS_FILENAME


class SoccerNetFeatures(Dataset):
    def __init__(self, features_path, labels_path, features, split, frames_per_clip=41, framerate=2, class_samples_per_epoch=1000, test_overlap=0) -> None:
        self.features_path = features_path
        self.labels_path = labels_path
        self.features = features

        self.split = split
        self.list_games = getListGames(self.split)

        self.frames_per_clip = frames_per_clip
        self.framerate = framerate
        self.class_samples_per_epoch = class_samples_per_epoch
        self.test_overlap = test_overlap

        self.n_frames_per_half_match = {}
        for game in self.list_games:
            with open(os.path.join(self.labels_path, game, LABELS_FILENAME), 'r') as f:
                label = json.load(f)

                self.n_frames_per_half_match[game + "_1"] = label['n_frames_half_1']
                self.n_frames_per_half_match[game + "_2"] = label['n_frames_half_2']

        if "train" in self.split:
            self.interesting_events = load_events(self.list_games, self.labels_path, self.frames_per_clip, self.framerate)
            self.augmented_events = augment_training(self.interesting_events, self.frames_per_clip)
            self.all_background_events = load_background_train(self.n_frames_per_half_match, self.interesting_events, self.frames_per_clip)  # always the same for each epoch

            self.update_samples()

        elif "challenge" in self.split:  # challenge split without annotations
            self.sampled_events = load_background_val(self.n_frames_per_half_match, [], self.frames_per_clip, self.test_overlap)
            self.sampled_events.sort(key=lambda event: (event["match"], event["half"], event["frame_indexes"][0]))

        else:  # testing split with annotations
            self.interesting_events = load_events(self.list_games, self.labels_path, self.frames_per_clip, self.framerate)
            self.sampled_events = load_background_val(self.n_frames_per_half_match, self.interesting_events, self.frames_per_clip, self.test_overlap)
            self.sampled_events.sort(key=lambda event: (event["match"], event["half"], event["frame_indexes"][0]))

        self.sampled_events_features = self.get_sampled_events_features()

    def get_sampled_events_features(self):
        sampled_events_features = dict()
        for game_id, half in tqdm(set((event["match"], event["half"]) for event in self.sampled_events)):
            url = os.path.join(self.features_path, game_id, f'{half}_' + self.features)
            sampled_events_features[url] = np.load(url)
        return sampled_events_features

    def __getitem__(self, index):
        item = self.sampled_events[index]
        url = os.path.join(self.features_path, item["match"], f'{item["half"]}_' + self.features)
        frame_indexes = item["frame_indexes"]
        label = item["label"]
        rel_offset = item["rel_offset"]
        frame_features = torch.from_numpy(self.sampled_events_features[url][frame_indexes[0]:frame_indexes[1]])
        # frame_features = torch.from_numpy(np.load(url)[frame_indexes[0]:frame_indexes[1]])

        if len(frame_features) < self.frames_per_clip:  # if the center frame was at the beginning (or at the end) of the video, we loaded less frames than necessary
            if len(frame_features) > 0:
                frame_features = torch.cat((frame_features, repeat(frame_features[-1], 'feat -> fr feat', fr=self.frames_per_clip - len(frame_features))), dim=0)
            else:
                frame_features = torch.zeros((self.frames_per_clip, frame_features.shape[1]))

        return frame_features, EVENT_DICTIONARY_V2[label], rel_offset, item["match"], item["half"], frame_indexes[0]

    def __len__(self):
        return len(self.sampled_events)

    def update_samples(self):
        self.sampled_events = sample_training(self.augmented_events, how_many=self.class_samples_per_epoch)  # sample with different random offsets for each epoch
        self.sampled_background = sample_background(self.all_background_events, how_many=self.class_samples_per_epoch)
        self.sampled_events = self.sampled_events + self.sampled_background
        random.shuffle(self.sampled_events)

        self.sampled_events_features = self.get_sampled_events_features()