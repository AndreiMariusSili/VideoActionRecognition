import typing as tp

import torch as th
from torch import nn


class VarTimeAlignedResNetClassifier(nn.Module):
    in_planes: int
    num_classes: int
    drop_rate: float

    def __init__(self, in_planes: int, embed_planes: tp.Optional[int], drop_rate: float, num_classes: int):
        super(VarTimeAlignedResNetClassifier, self).__init__()

        self.in_planes = in_planes
        self.embed_planes = embed_planes
        self.num_classes = num_classes
        self.drop_rate = drop_rate

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
        self.classifier = nn.Conv2d(self.embed_planes, self.num_classes, kernel_size=1, bias=True)

    def forward(self, _in: th.Tensor) -> tp.Tuple[th.Tensor, th.Tensor]:
        b, s, t, c, h, w = _in.shape
        _in = _in.reshape(b * s, t * c, h, w)

        _embed = self.aggregation(_in)
        _out = self.classifier(_embed)

        return _out.reshape(b, s, self.num_classes), _embed.reshape(b, s, self.embed_planes)
