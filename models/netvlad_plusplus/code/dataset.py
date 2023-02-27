from torch.utils.data import Dataset

import numpy as np
import os

import torch

import logging
import json

from SoccerNet.Downloader import getListGames
from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2


def feats2clip(feats, stride, clip_length, padding="replicate_last", off=0):
    if padding == "zeropad":
        print("beforepadding", feats.shape)
        pad = feats.shape[0] - int(feats.shape[0] / stride) * stride
        print("pad need to be", clip_length - pad)
        m = torch.nn.ZeroPad2d((0, 0, clip_length - pad, 0))
        feats = m(feats)
        print("afterpadding", feats.shape)
        # nn.ZeroPad2d(2)

    idx = torch.arange(start=0, end=feats.shape[0] - 1, step=stride)
    idxs = []
    for i in torch.arange(-off, clip_length - off):
        idxs.append(idx + i)
    idx = torch.stack(idxs, dim=1)

    if padding == "replicate_last":
        idx = idx.clamp(0, feats.shape[0] - 1)

    # idx = torch.arange(feats.shape[0] * clip_length).view((-1, clip_length, feats.shape[-1]))
    return feats[idx, ...]


class SoccerNetClips(Dataset):
    def __init__(self, path, features, split, framerate=2, window_size=15):
        self.path = path
        self.listGames = getListGames(split)
        self.features = features
        self.window_size_frame = window_size * framerate
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels = "Labels-v2.json"

        self.game_feats = list()
        self.game_labels = list()

        print(f'Started loading features for {split[0]} split...')
        for game in self.listGames:
            # Load features
            feat_half1 = np.load(os.path.join(self.path, game, "1_" + self.features))
            feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
            feat_half2 = np.load(os.path.join(self.path, game, "2_" + self.features))
            feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

            feat_half1 = feats2clip(torch.from_numpy(feat_half1), stride=self.window_size_frame,
                                    clip_length=self.window_size_frame)
            feat_half2 = feats2clip(torch.from_numpy(feat_half2), stride=self.window_size_frame,
                                    clip_length=self.window_size_frame)

            # Load labels
            labels = json.load(open(os.path.join(self.path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes + 1))
            label_half1[:, 0] = 1  # those are BG classes
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes + 1))
            label_half2[:, 0] = 1  # those are BG classes

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * (seconds + 60 * minutes)

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                # if label outside temporal of view
                if half == 1 and frame // self.window_size_frame >= label_half1.shape[0]:
                    continue
                if half == 2 and frame // self.window_size_frame >= label_half2.shape[0]:
                    continue

                if half == 1:
                    label_half1[frame // self.window_size_frame][0] = 0  # not BG anymore
                    label_half1[frame // self.window_size_frame][label + 1] = 1  # that's my class

                if half == 2:
                    label_half2[frame // self.window_size_frame][0] = 0  # not BG anymore
                    label_half2[frame // self.window_size_frame][label + 1] = 1  # that's my class

            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(label_half1)
            self.game_labels.append(label_half2)

        self.game_feats = np.concatenate(self.game_feats)
        self.game_labels = np.concatenate(self.game_labels)

        print(f'{split[0]} features loaded.')

    def __getitem__(self, index):
        return self.game_feats[index, :, :], self.game_labels[index, :]

    def __len__(self):
        return len(self.game_feats)


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features, split=("test",), framerate=2, window_size=15):
        self.path = path
        self.listGames = getListGames(split)[:1]
        self.features = features
        self.window_size_frame = window_size * framerate
        self.framerate = framerate
        self.split = split
        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels = "Labels-v2.json"

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            feat_half1 (np.array): features for the 1st half.
            feat_half2 (np.array): features for the 2nd half.
            label_half1 (np.array): labels (one-hot) for the 1st half.
            label_half2 (np.array): labels (one-hot) for the 2nd half.
        """
        # Load features
        feat_half1 = np.load(os.path.join(os.pardir, os.pardir, self.path, self.listGames[index], "1_" + self.features))
        feat_half1 = feat_half1.reshape(-1, feat_half1.shape[-1])
        feat_half2 = np.load(os.path.join(os.pardir, os.pardir, self.path, self.listGames[index], "2_" + self.features))
        feat_half2 = feat_half2.reshape(-1, feat_half2.shape[-1])

        # Load labels
        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))

        # check if annoation exists
        labels_path = os.path.join(self.path, self.listGames[index], self.labels)
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                labels = json.load(f)

            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * (seconds + 60 * minutes)

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0] - 1)
                    label_half1[frame, label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0] - 1)
                    label_half2[frame, label] = value

        feat_half1 = feats2clip(torch.from_numpy(feat_half1),
                                stride=1, off=int(self.window_size_frame / 2),
                                clip_length=self.window_size_frame)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2),
                                stride=1, off=int(self.window_size_frame / 2),
                                clip_length=self.window_size_frame)

        return self.listGames[index], feat_half1, feat_half2, label_half1, label_half2

    def __len__(self):
        return len(self.listGames)
