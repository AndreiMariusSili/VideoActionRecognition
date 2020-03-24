from typing import Tuple

import torch as th
from torch import nn

import models.helpers as hp
from models.tarn.common import classifier as cls, spatial_decoder as sd, spatial_encoder as se, temporal_encoder as te


class AETimeAlignedResNet(nn.Module):
    def __init__(self, time_steps: int,
                 spatial_encoder_planes: Tuple[int, ...], bottleneck_planes: int,
                 spatial_decoder_planes: Tuple[int, ...],
                 classifier_drop_rate: float, class_embed_planes: int, num_classes: int, flow: bool):
        super(AETimeAlignedResNet, self).__init__()

        self.spatial_encoder_planes = spatial_encoder_planes
        self.spatial_bottleneck_planes = bottleneck_planes
        self.spatial_decoder_planes = spatial_decoder_planes

        self.time_steps = time_steps
        self.drop_rate = classifier_drop_rate
        self.class_embed_planes = class_embed_planes
        self.num_classes = num_classes
        self.flow = flow

        self.class_in_planes = self.spatial_bottleneck_planes * self.time_steps
        self.spatial_encoder = se.SpatialResNetEncoder(self.spatial_encoder_planes, self.spatial_bottleneck_planes)
        self.temporal_encoder = te.TemporalResNetEncoder(self.time_steps,
                                                         self.spatial_bottleneck_planes)
        self.decoder = sd.SpatialResNetDecoder(self.spatial_bottleneck_planes, self.spatial_decoder_planes, self.flow)
        self.classifier = cls.TimeAlignedResNetClassifier(self.class_in_planes,
                                                          self.class_embed_planes,
                                                          self.drop_rate,
                                                          self.num_classes)

        hp.he_init(self)

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        _spatial_out, _spatial_mid_outs = self.spatial_encoder(_in)
        _temporal_embed, _temporal_embeds = self.temporal_encoder(_spatial_out)
        _pred, _class_embed = self.classifier(_temporal_embeds)
        _recon = self.decoder(_temporal_embeds, _spatial_mid_outs)

        return _recon, _pred, _temporal_embeds, _class_embed
