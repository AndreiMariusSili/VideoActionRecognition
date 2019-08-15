from typing import Optional, Tuple

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

    def __init__(self, time_steps: int, classifier_drop_rate: float, num_classes: int, encoder_planes: Tuple[int, ...],
                 temporal_out_planes: int, class_embed_planes: Optional[int] = None):
        super(TimeAlignedResNet, self).__init__()

        self.spatial_encoder_planes = encoder_planes
        self.temporal_in_planes = encoder_planes[-1]
        self.temporal_out_planes = temporal_out_planes

        self.time_steps = time_steps
        self.drop_rate = classifier_drop_rate
        self.num_classes = num_classes
        self.class_embed_planes = class_embed_planes

        self.class_in_planes = self.temporal_out_planes * self.time_steps

        self.spatial_encoder = en.SpatialResNetEncoder(self.spatial_encoder_planes)
        self.temporal_encoder = en.TemporalResNetEncoder(self.time_steps,
                                                         self.temporal_in_planes,
                                                         self.temporal_out_planes)
        self.classifier = cls.TimeAlignedResNetClassifier(self.class_in_planes,
                                                          self.class_embed_planes,
                                                          self.drop_rate,
                                                          self.num_classes)

        self._he_init()

    # noinspection DuplicatedCode
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
        _temporal_out, _temporal_outs = self.temporal_encoder(_spatial_out)
        _pred, _embed = self.classifier(_temporal_outs)

        return _pred, _embed


if __name__ == "__main__":
    import models.helpers as hp

    _input_var = th.randn(2, 4, 3, 224, 224)
    model = TimeAlignedResNet(4, 0.0, 30, (16, 32, 64, 128, 256), 128, 512)
    output, embeds = model(_input_var)
    print(f'{hp.count_parameters(model):,}')
    print(f'{hp.count_parameters(model.spatial_encoder):,}')
    print(f'{hp.count_parameters(model.temporal_encoder):,}')
    print(f'{hp.count_parameters(model.classifier):,}')
