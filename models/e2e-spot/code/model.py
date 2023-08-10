import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import FloatTensor as ftens

import einops
import torchvision
import timm


class GSM(nn.Module):
    def __init__(self, fPlane: int, num_segments: int = 3):
        super().__init__()

        self.conv3D = nn.Conv3d(fPlane, 2, (3, 3, 3), stride=1,
                                padding=(1, 1, 1), groups=2)
        nn.init.constant_(self.conv3D.weight, 0)
        nn.init.constant_(self.conv3D.bias, 0)
        self.fPlane = fPlane
        self.num_segments = num_segments
        self.bn = nn.BatchNorm3d(num_features=fPlane)

    def lshift_zeroPad(self, x):
        return torch.cat((x[:,:,1:], ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0)), dim=2)

    def rshift_zeroPad(self, x):
        return torch.cat((ftens(x.size(0), x.size(1), 1, x.size(3), x.size(4)).fill_(0), x[:,:,:-1]), dim=2)

    def forward(self, x):
        batchSize = x.size(0) // self.num_segments
        shape = x.size(1), x.size(2), x.size(3)
        assert  shape[0] == self.fPlane
        x = x.view(batchSize, self.num_segments, *shape).permute(0, 2, 1, 3, 4).contiguous()
        x_bn = self.bn(x)
        x_bn_relu = F.relu(x_bn)
        gate = F.tanh(self.conv3D(x_bn_relu))
        gate_group1 = gate[:, 0].unsqueeze(1)
        gate_group2 = gate[:, 1].unsqueeze(1)
        x_group1 = x[:, :self.fPlane // 2]
        x_group2 = x[:, self.fPlane // 2:]
        y_group1 = gate_group1 * x_group1
        y_group2 = gate_group2 * x_group2

        r_group1 = x_group1 - y_group1
        r_group2 = x_group2 - y_group2

        y_group1 = self.lshift_zeroPad(y_group1) + r_group1
        y_group2 = self.rshift_zeroPad(y_group2) + r_group2

        y_group1 = y_group1.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)
        y_group2 = y_group2.view(batchSize, 2, self.fPlane // 4, self.num_segments, *shape[1:]).permute(0, 2, 1, 3, 4, 5)

        y = torch.cat((y_group1.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:]),
                       y_group2.contiguous().view(batchSize, self.fPlane//2, self.num_segments, *shape[1:])), dim=1)

        return y.permute(0, 2, 1, 3, 4).contiguous().view(batchSize*self.num_segments, *shape)


class GatedShift(nn.Module):
    def __init__(self, net, n_segment, n_div):
        super(GatedShift, self).__init__()

        print(f'{type(net) = }')
        if isinstance(net, torchvision.models.resnet.BasicBlock):
            channels = net.conv1.in_channels
        elif isinstance(net, torchvision.ops.misc.ConvNormActivation):
            channels = net[0].in_channels
        elif isinstance(net, timm.models.layers.conv_bn_act.ConvBnAct):
            channels = net.conv.in_channels
        elif isinstance(net, nn.Conv2d):
            channels = net.in_channels
        else:
            raise NotImplementedError(type(net))

        self.fold_dim = math.ceil(channels // n_div / 4) * 4
        self.gsm = GSM(self.fold_dim, n_segment)
        self.net = net
        self.n_segment = n_segment
        print(f'=> Using GSM, fold dim: {self.fold_dim} / {channels}')

    def forward(self, x):
        y = torch.zeros_like(x)
        y[:, :self.fold_dim, :, :] = self.gsm(x[:, :self.fold_dim, :, :])
        y[:, self.fold_dim:, :, :] = x[:, self.fold_dim:, :, :]
        return self.net(y)


def make_temporal_shift(net: timm.models.regnet.RegNet, clip_len: int):

    def make_block_temporal(stage):
        blocks = list(stage.children())
        for i, b in enumerate(blocks):
            blocks[i].conv1 = GatedShift(b.conv1, n_segment=clip_len, n_div=4)

    make_block_temporal(net.s1)
    make_block_temporal(net.s2)
    make_block_temporal(net.s3)
    make_block_temporal(net.s4)


class GRUPrediction(nn.Module):

    def __init__(self, feat_dim: int, num_classes: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(feat_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout()
        self.fc_out = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, x):
        y, _ = self.gru(x)

        y = self.dropout(y)
        y = einops.rearrange(y, 'bs clip feat -> (bs clip) feat')
        y = self.fc_out(y)
        y = einops.rearrange(y, '(bs clip) out -> bs clip out')

        return y


class E2EModel(nn.Module):
    def __init__(self, num_classes: int, feature_arch: str, clip_len: int) -> None:
        super().__init__()

        assert feature_arch in ('regnet_002', 'regnet_008')
        self.features_model = timm.create_model(feature_arch, pretrained=True)
        self.feat_dim = self.features_model.head.fc.in_features
        self.require_clip_len = clip_len

        self.features_model.head.fc = nn.Identity()
        make_temporal_shift(self.features_model, clip_len)

        self.pred_fine = GRUPrediction(self.feat_dim, num_classes, hidden_dim=self.feat_dim, num_layers=1)

    def forward(self, x):
        batch_size, true_clip_len, channels, height, width = x.shape

        assert true_clip_len <= self.require_clip_len

        clip_len = true_clip_len
        if true_clip_len < self.require_clip_len:
            x = F.pad(x, (0,) * 7 + (self.require_clip_len - true_clip_len,))
            clip_len = self.require_clip_len

        im_feat = self.features_model(x.view(-1, channels, height, width)).reshape(batch_size, clip_len, self.feat_dim)

        # Undo padding
        if true_clip_len != clip_len:
            im_feat = im_feat[:, :true_clip_len, :]

        return self.pred_fine(im_feat)


def change_num_classes(model: E2EModel, new_num_classes: int):
    model.pred_fine.fc_out = nn.Linear(model.pred_fine.fc_out.in_features, new_num_classes)
    return model