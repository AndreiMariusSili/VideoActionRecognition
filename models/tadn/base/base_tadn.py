from typing import Tuple

import torch as th
from torch import nn

from models.tadn.base._classifier import TimeAlignedDenseNetClassifier
from models.tadn.base._spatial_encoder import SpatialInceptionV1Encoder
from models.tadn.base._temporal_encoder import TemporalDenseNetEncoder


class TimeAlignedDenseNet(nn.Module):
    NAME = 'TADN'

    def __init__(self, time_steps: int, temporal_in_planes: int, growth_rate: int, temporal_drop_rate: float,
                 classifier_drop_rate: float, class_embed_planes: int, num_classes: int):
        super(TimeAlignedDenseNet, self).__init__()

        self.time_steps = time_steps
        self.temporal_in_planes = temporal_in_planes
        self.growth_rate = growth_rate
        self.temporal_drop_rate = temporal_drop_rate
        self.aggregation_drop_rate = classifier_drop_rate
        self.num_classes = num_classes
        self.class_embed_planes = class_embed_planes

        self.temporal_out_planes = self.time_steps * self.growth_rate

        self.spatial_encoder = SpatialInceptionV1Encoder(self.temporal_drop_rate, self.temporal_in_planes)
        self.temporal_encoder = TemporalDenseNetEncoder(self.temporal_in_planes, self.time_steps,
                                                        self.growth_rate, self.temporal_drop_rate)
        self.classifier = TimeAlignedDenseNetClassifier(self.temporal_out_planes, self.class_embed_planes,
                                                        self.aggregation_drop_rate, self.num_classes)

        self._he_init()

    def _he_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                th.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        _spatial_embeds = self.spatial_encoder(_in)
        _temporal_embeds = self.temporal_encoder(_spatial_embeds)
        _pred, _class_embed = self.classifier(_temporal_embeds)

        return _pred, _temporal_embeds, _class_embed
