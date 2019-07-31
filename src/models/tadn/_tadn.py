from typing import Tuple

import torch as th
from torch import nn

from models.tadn._classifier import TimeAlignedDenseNetClassifier
from models.tadn._encoder import SpatialInceptionV1Encoder, TemporalDenseNetEncoder


class TimeAlignedDenseNet(nn.Module):
    def __init__(self, time_steps: int, temporal_in_planes: int, growth_rate: int, temporal_drop_rate: float,
                 classifier_drop_rate: float, num_classes: int, class_embed_planes: int = None):
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

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        _spatial_out = self.spatial_encoder(_in)
        _temporal_out = self.temporal_encoder(_spatial_out)
        _pred, _embed = self.classifier(_temporal_out)

        return _pred, _embed


if __name__ == "__main__":
    import os

    os.chdir(f'{os.getenv("MT_ROOT")}/src')
    import helpers as hp

    _input_var = th.randn(2, 4, 3, 224, 224)
    model = TimeAlignedDenseNet(4, 256, 64, 0.5, 0.5, 30, 512)
    print(model)
    output, embeds = model(_input_var)
    print(output.shape, embeds.shape)
    print(f'{hp.count_parameters(model):,}')
    print(f'{hp.count_parameters(model.spatial_encoder):,}')
    print(f'{hp.count_parameters(model.temporal_encoder):,}')
    print(f'{hp.count_parameters(model.classifier):,}')
