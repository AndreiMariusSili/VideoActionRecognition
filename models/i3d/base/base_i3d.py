# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import typing as tp

import torch as th
from torch import nn

import models.helpers as hp
from models.i3d.common import classifier as cls, encoder as en


class I3D(nn.Module):
    NAME = 'I3D'

    def __init__(self, time_steps: int, num_classes: int, dropout_prob: float = 0.0):
        super(I3D, self).__init__()
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        self.encoder = en.I3DEncoder(None)
        self.classifier = cls.I3DClassifier(1024, dropout_prob, num_classes)

        hp.he_init(self)

    def forward(self, _in: th.Tensor) -> tp.Tuple[th.Tensor, th.Tensor, th.Tensor]:
        _temporal_embeds = self.encoder(_in)
        _pred, _class_embed = self.classifier(_temporal_embeds)

        return _pred, _temporal_embeds, _class_embed
