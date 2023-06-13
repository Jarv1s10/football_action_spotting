import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, labels, output):
        return torch.mean(torch.mean(labels * - torch.log(output) + (1 - labels) * - torch.log(1 - output)))


class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_batch_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1 / np.sqrt(feature_size)) * torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1 / np.sqrt(feature_size)) * torch.randn(1, feature_size, cluster_size))

        self.add_batch_norm = add_batch_norm
        self.out_dim = cluster_size * feature_size

    def forward(self, x):
        # x [BS, T, D]
        max_sample = x.size()[1]

        if self.add_batch_norm:
            x = F.normalize(x, p=2, dim=2)

        x = x.reshape(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad


class Model(nn.Module):
    def __init__(self, input_size=512, num_classes=17, vocab_size=64, window_size=15, framerate=2):
        """
        INPUT: a Tensor of shape (batch_size,window_size,feature_size)
        OUTPUTS: a Tensor of shape (batch_size,num_classes+1)
        """

        super(Model, self).__init__()

        self.window_size_frame = window_size * framerate
        self.input_size = input_size
        self.num_classes = num_classes
        self.framerate = framerate
        self.vlad_k = vocab_size

        if self.input_size != 512:
            self.feature_extractor = nn.Linear(self.input_size, 512)
            self.input_size = 512

        self.pool_layer_before = NetVLAD(cluster_size=self.vlad_k // 2, feature_size=self.input_size,
                                         add_batch_norm=True)
        self.pool_layer_after = NetVLAD(cluster_size=self.vlad_k // 2, feature_size=self.input_size,
                                        add_batch_norm=True)

        self.fc = nn.Linear(input_size * self.vlad_k, self.num_classes + 1)

        self.drop = nn.Dropout(p=0.4)
        self.sigm = nn.Sigmoid()

    def load_weights(self, weights=None):
        if weights is not None:
            print(f"=> loading checkpoint '{weights}'")
            checkpoint = torch.load(weights)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{weights}' (epoch {checkpoint['epoch']})")

    def forward(self, inputs):
        # `inputs` shape: (batch, frames, dim_features)
        if inputs.shape[-1] != self.input_size:
            inputs = rearrange(inputs, 'b fr d -> (b fr) d')
            inputs = self.feature_extractor(inputs)
            inputs = rearrange(inputs, '(b fr) d -> b fr d', fr=self.window_size_frame)

        nb_frames_50 = int(inputs.shape[1] / 2)
        inputs_before_pooled = self.pool_layer_before(inputs[:, :nb_frames_50, :])
        inputs_after_pooled = self.pool_layer_after(inputs[:, nb_frames_50:, :])
        inputs_pooled = torch.cat((inputs_before_pooled, inputs_after_pooled), dim=1)

        # Extra FC layer with dropout and sigmoid activation
        output = self.sigm(self.fc(self.drop(inputs_pooled)))

        return output
