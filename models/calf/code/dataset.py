import random
import os
import json

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

from SoccerNet.Downloader import getListGames
from SoccerNet.Evaluation.utils import EVENT_DICTIONARY_V2

from preprocessing import oneHotToShifts, getTimestampTargets, getChunks_anchors

K_V2 = torch.FloatTensor([[-100, -98, -20, -40, -96, -5, -8, -93, -99, -31, -75, -10, -97, -75, -20, -84, -18],
                          [-50, -49, -10, -20, -48, -3, -4, -46, -50, -15, -37, -5, -49, -38, -10, -42, -9],
                          [50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9],
                          [100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18]]).cuda()


class SoccerNetClips(Dataset):
    def __init__(self, features_path, labels_path, features, split, framerate=2, num_detections=15, chunk_size=240, receptive_field=80, chunks_per_epoch=6000):
        self.features_path = features_path
        self.labels_path = labels_path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.chunks_per_epoch = chunks_per_epoch

        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"
        self.K_parameters = K_V2*framerate
        self.num_detections = num_detections

        self.game_feats = list()
        self.game_labels = list()
        self.game_anchors = list()
        for _ in np.arange(self.num_classes+1):
            self.game_anchors.append(list())

        game_counter = 0
        for game in tqdm(self.listGames):
            # Load features
            feat_half1 = np.load(os.path.join(self.features_path, game, "1_" + self.features))
            feat_half2 = np.load(os.path.join(self.features_path, game, "2_" + self.features))

            # Load labels
            labels = json.load(open(os.path.join(self.labels_path, game, self.labels)))

            label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
            label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


            for annotation in labels["annotations"]:

                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = framerate * ( seconds + 60 * minutes )

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = 1

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = 1

            shift_half1 = oneHotToShifts(label_half1, self.K_parameters.cpu().numpy())
            shift_half2 = oneHotToShifts(label_half2, self.K_parameters.cpu().numpy())

            anchors_half1 = getChunks_anchors(shift_half1, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)
            game_counter = game_counter+1

            anchors_half2 = getChunks_anchors(shift_half2, game_counter, self.K_parameters.cpu().numpy(), self.chunk_size, self.receptive_field)
            game_counter = game_counter+1

            self.game_feats.append(feat_half1)
            self.game_feats.append(feat_half2)
            self.game_labels.append(shift_half1)
            self.game_labels.append(shift_half2)

            for anchor in anchors_half1:
                self.game_anchors[anchor[2]].append(anchor)
            for anchor in anchors_half2:
                self.game_anchors[anchor[2]].append(anchor)

    def __getitem__(self, index):
        # Retrieve the game index and the anchor
        class_selection = random.randint(0, self.num_classes)
        event_selection = random.randint(0, len(self.game_anchors[class_selection])-1)
        game_index = self.game_anchors[class_selection][event_selection][0]
        anchor = self.game_anchors[class_selection][event_selection][1]

        # Compute the shift for event chunks
        if class_selection < self.num_classes:
            shift = np.random.randint(-self.chunk_size+self.receptive_field, -self.receptive_field)
            start = anchor + shift
        # Compute the shift for non-event chunks
        else:
            start = random.randint(anchor[0], anchor[1]-self.chunk_size)
        if start < 0:
            start = 0
        if start+self.chunk_size >= self.game_feats[game_index].shape[0]:
            start = self.game_feats[game_index].shape[0]-self.chunk_size-1

        # Extract the clips
        clip_feat = self.game_feats[game_index][start:start+self.chunk_size]
        clip_labels = self.game_labels[game_index][start:start+self.chunk_size]

        # Put loss to zero outside receptive field
        clip_labels[0:int(np.ceil(self.receptive_field/2)),:] = -1
        clip_labels[-int(np.ceil(self.receptive_field/2)):,:] = -1

        # Get the spotting target
        clip_targets = getTimestampTargets(np.array([clip_labels]), self.num_detections)[0]

        return torch.from_numpy(clip_feat), torch.from_numpy(clip_labels), torch.from_numpy(clip_targets)

    def __len__(self):
        return self.chunks_per_epoch


class SoccerNetClipsTesting(Dataset):
    def __init__(self, features_path, labels_path, features, split="test", framerate=2, num_detections=15, chunk_size=240, receptive_field=80):
        self.features_path = features_path
        self.labels_path = labels_path
        self.listGames = getListGames(split)
        self.features = features
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate

        self.dict_event = EVENT_DICTIONARY_V2
        self.num_classes = 17
        self.labels="Labels-v2.json"
        self.K_parameters = K_V2*framerate
        self.num_detections = num_detections
        self.split=split

    def __getitem__(self, index):

        # Load features
        feat_half1 = np.load(os.path.join(self.features_path, self.listGames[index], "1_" + self.features))
        feat_half2 = np.load(os.path.join(self.features_path, self.listGames[index], "2_" + self.features))


        label_half1 = np.zeros((feat_half1.shape[0], self.num_classes))
        label_half2 = np.zeros((feat_half2.shape[0], self.num_classes))


        # Load labels
        if os.path.exists(os.path.join(self.labels_path, self.listGames[index], self.labels)):
            labels = json.load(open(os.path.join(self.labels_path, self.listGames[index], self.labels)))

            for annotation in labels["annotations"]:
                time = annotation["gameTime"]
                event = annotation["label"]

                half = int(time[0])

                minutes = int(time[-5:-3])
                seconds = int(time[-2::])
                frame = self.framerate * ( seconds + 60 * minutes )

                if event not in self.dict_event:
                    continue
                label = self.dict_event[event]

                value = 1
                if "visibility" in annotation.keys():
                    if annotation["visibility"] == "not shown":
                        value = -1

                if half == 1:
                    frame = min(frame, feat_half1.shape[0]-1)
                    label_half1[frame][label] = value

                if half == 2:
                    frame = min(frame, feat_half2.shape[0]-1)
                    label_half2[frame][label] = value

        def feats2clip(feats, stride, clip_length):
            idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            idx = idx.clamp(0, feats.shape[0]-1)
            idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

            return feats[idx,:]

        feat_half1 = feats2clip(torch.from_numpy(feat_half1),
                        stride=self.chunk_size-self.receptive_field,
                        clip_length=self.chunk_size)

        feat_half2 = feats2clip(torch.from_numpy(feat_half2),
                        stride=self.chunk_size-self.receptive_field,
                        clip_length=self.chunk_size)

        return feat_half1, feat_half2, torch.from_numpy(label_half1), torch.from_numpy(label_half2)

    def __len__(self):
        return len(self.listGames)