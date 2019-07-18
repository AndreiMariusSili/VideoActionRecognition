from typing import Tuple

import torch as th
from torch import nn

from models.tarn import _classifier as cls, _encoder as en


class TimeAlignedResNet(nn.Module):
    spatial: nn.Module
    temporal: nn.ModuleList
    aggregation: nn.Sequential
    classifier: nn.Sequential

    in_planes: int
    out_planes: int
    time_steps: int
    drop_rate: float
    num_classes: int

    def __init__(self, time_steps: int, drop_rate: float, num_classes: int, encoder_planes: Tuple[int, ...]):
        super(TimeAlignedResNet, self).__init__()

        self.spatial_encoder_planes = encoder_planes
        self.temporal_in_planes = self.temporal_out_planes = encoder_planes[-1]

        self.time_steps = time_steps
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        self.spatial_encoder = en.SpatialResNetEncoder(self.spatial_encoder_planes)
        self.temporal_encoder = en.TemporalResNetEncoder(self.time_steps,
                                                         self.temporal_in_planes,
                                                         self.temporal_out_planes)
        self.classifier = cls.TimeAlignedResNetClassifier(self.temporal_out_planes * self.time_steps,
                                                          self.drop_rate,
                                                          self.num_classes)

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
        b, t, c, h, w = _in.shape

        _out = self.spatial_encoder(_in)
        _out, _outs = self.temporal_encoder(_out)
        _out, _embed = self.classifier(_outs)

        return _out.reshape(b, self.num_classes), _embed.reshape(b, -1)


if __name__ == "__main__":
    import models.helpers

    _input_var = th.randn(2, 4, 3, 224, 224)
    model = TimeAlignedResNet(4, 0.5, 30, (16, 32, 64, 128, 256))
    output, embeds = model(_input_var)
    print(output.shape, f'{models.helpers.count_parameters(model):,}')
