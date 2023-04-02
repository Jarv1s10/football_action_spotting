import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RMSNet(nn.Module):
    def __init__(self, features_size, num_classes) -> None:
        super().__init__()
        self.features_size = features_size
        self.num_classes = num_classes
        self.temporal_conv1 = torch.nn.Conv1d(in_channels=self.features_size, out_channels=512, kernel_size=9, stride=1, padding=4)
        self.temporal_conv2 = torch.nn.Conv1d(in_channels=512, out_channels=256, kernel_size=9, stride=1, padding=4)

        self.fc = torch.nn.Linear(256, 128)
        self.fc_class = torch.nn.Linear(128, num_classes)
        self.fc_t_shift = torch.nn.Linear(128, 1)

    def load_weights(self, weights_path=None):
        if weights_path is not None:
            print(f"=> loading checkpoint '{weights_path}'")
            checkpoint = torch.load(weights_path)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{weights_path}' (epoch {checkpoint['epoch']})")

    def forward(self, inputs):
        x = rearrange(inputs, 'bs fr feat -> bs feat fr')
        x = F.relu(self.temporal_conv1(x))
        x = F.relu(self.temporal_conv2(x))
        x = F.dropout(x, p=0.1, training=self.training)

        x = rearrange(x, 'bs feat fr -> bs fr feat')
        x = x.contiguous()

        x_event = torch.max(x, dim=1)[0]
        x_event = F.relu(self.fc(x_event))

        out = F.softmax(self.fc_class(x_event), dim=-1)
        rel_offset = F.sigmoid(self.fc_t_shift(x_event))

        return out, rel_offset