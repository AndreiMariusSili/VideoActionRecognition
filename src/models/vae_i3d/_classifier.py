# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
from typing import Tuple

import torch as th
from torch import nn

from models import common as cm
from options import model_options as mo


class I3DClassifier(nn.Module):
    num_classes: int
    latent_size: int

    def __init__(self, latent_size: int, dropout_prob: float, num_classes: int):
        super(I3DClassifier, self).__init__()

        self.latent_size = latent_size
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.aggregation = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Dropout3d(dropout_prob)
        )
        opts = mo.Unit3DOptions(in_channels=latent_size, out_channels=self.num_classes, kernel_size=(1, 1, 1),
                                stride=(1, 1, 1), activation='none', use_bias=False, use_bn=False, padding='VALID')
        self.classifier = cm.Unit3D(opts)

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        b, s, c, t, h, w = _in.shape
        _in = _in.reshape(b * s, c, t, h, w)
        _embed = self.aggregation(_in)
        _out = self.classifier(_embed)

        return _out.reshape(b, s, self.num_classes), _embed.reshape(b, s, self.latent_size)
