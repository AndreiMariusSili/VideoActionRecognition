from typing import Optional, Tuple

import torch as th
from torch import nn


class TimeAlignedDenseNetClassifier(nn.Module):
    def __init__(self, in_planes: int, embed_planes: Optional[int], drop_rate: float, num_classes: int):
        super(TimeAlignedDenseNetClassifier, self).__init__()

        self.in_planes = in_planes
        self.embed_planes = embed_planes
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        if self.embed_planes:
            self.aggregation = nn.Sequential(
                nn.Conv2d(self.in_planes, self.embed_planes, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(self.embed_planes),
                nn.Dropout2d(self.drop_rate),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            self.embed_planes = self.in_planes
            self.aggregation = nn.Sequential(
                nn.Dropout2d(self.drop_rate),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        self.classifier = nn.Conv2d(self.embed_planes, self.num_classes, kernel_size=1)

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        b, c, h, w = _in.shape

        _embed = self.aggregation(_in)
        _out = self.classifier(_embed)

        return _out.reshape(b, self.num_classes), _embed.reshape(b, self.embed_planes)
