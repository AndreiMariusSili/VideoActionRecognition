import typing as tp

import torch as th
from torch import nn

import models.helpers as hp
from models.i3d.common import classifier as cls, decoder as de, encoder as en

AE_FORWARD = tp.Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]


class AEI3D(nn.Module):
    def __init__(self, time_steps: int, embed_planes: int, dropout_prob: float, num_classes: int, flow: bool):
        super(AEI3D, self).__init__()
        self.time_steps = time_steps
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embed_planes = embed_planes
        self.flow = flow

        self.encoder = en.I3DEncoder(self.embed_planes, flow=self.flow)
        self.decoder = de.I3DDecoder(self.embed_planes, self.time_steps, self.flow)
        self.classifier = cls.I3DClassifier(self.embed_planes, self.dropout_prob, self.num_classes)

        hp.he_init(self)

    def forward(self, _in: th.Tensor) -> AE_FORWARD:
        _temporal_embeds, _encoder_mid_outs = self.encoder(_in)
        _recon = self.decoder(_temporal_embeds, _encoder_mid_outs)
        _pred, _class_embed = self.classifier(_temporal_embeds)

        return _recon, _pred, _temporal_embeds, _class_embed
