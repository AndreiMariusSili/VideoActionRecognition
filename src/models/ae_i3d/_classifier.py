# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
from typing import Tuple

import torch as th
from torch import nn

from models import common as cm
from options import model_options as mo


class I3DClassifier(nn.Module):
    num_classes: int
    latent_channels: int

    def __init__(self, latent_channels: int, dropout_prob: float, num_classes: int):
        super(I3DClassifier, self).__init__()

        self.latent_channels = latent_channels
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout3d(dropout_prob)
        opts = mo.Unit3DOptions(in_channels=latent_channels, out_channels=self.num_classes, kernel_size=(1, 1, 1),
                                stride=(1, 1, 1), activation='none', use_bias=False, use_bn=False, padding='VALID')

        self.classifier = cm.Unit3D(opts)

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        b = _in.shape[0]

        _embed = self.avg_pool(_in)
        _out = self.dropout(_embed)
        _out = self.classifier(_out)

        return _out.reshape(b, self.num_classes), _embed.reshape(b, self.latent_channels)
