# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import typing as tp

import torch as th
from torch import nn

import models.i3d.common.blocks as ib
from options import model_options as mo


class I3DClassifier(nn.Module):
    def __init__(self, embed_planes: int, dropout_prob: float, num_classes: int):
        super(I3DClassifier, self).__init__()
        self.embed_planes = embed_planes
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

        self.avg_pool = nn.AdaptiveAvgPool3d([1, 1, 1])
        self.dropout = nn.Dropout3d(dropout_prob)
        opts = mo.Unit3DOptions(in_channels=embed_planes, out_channels=self.num_classes, kernel_size=[1, 1, 1],
                                stride=[1, 1, 1], activation='none', use_bias=False, use_bn=False, padding='SAME')

        self.classifier = ib.Unit3D(opts)

    def forward(self, _in: th.Tensor) -> tp.Tuple[th.Tensor, th.Tensor]:
        b = _in.shape[0]

        _embed = self.avg_pool(_in)
        _out = self.dropout(_embed)
        _out = self.classifier(_out)

        return _out.reshape(b, self.num_classes), _embed.reshape(b, self.embed_planes)
