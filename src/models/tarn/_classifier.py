from typing import Tuple

import torch as th
from torch import nn

SPATIAL_OUT_C, SPATIAL_OUT_H, SPATIAL_OUT_W = 256, 7, 7


class TimeAlignedResNetClassifier(nn.Module):
    in_planes: int
    num_classes: int
    drop_rate: float

    def __init__(self, embed_planes: int, drop_rate: float, num_classes: int):
        super(TimeAlignedResNetClassifier, self).__init__()

        self.in_planes = embed_planes
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.aggregation = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(self.drop_rate),
        )
        self.classifier = nn.Conv2d(self.in_planes, self.num_classes, kernel_size=1, bias=True)

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        b, t, c, h, w = _in.shape
        _in = _in.reshape(b, t * c, h, w)

        _embed = self.aggregation(_in)
        _out = self.classifier(_embed)

        return _out.reshape(b, self.num_classes), _embed.reshape(b, self.in_planes)
