from typing import Tuple

import torch as th
from torch import nn

import models.tarn.common as tc


class VarSpatialResNetEncoder(nn.Module):
    def __init__(self, out_planes: Tuple[int, ...], bottleneck_planes: int):
        super(VarSpatialResNetEncoder, self).__init__()
        self.in_planes = out_planes[0]
        self.out_planes = out_planes
        self.bottleneck_planes = bottleneck_planes

        self.conv1 = nn.Conv2d(3, self.out_planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(self.out_planes[1], 2))  # noqa
        for out_plane in self.out_planes[2:]:
            self.layers.append(self._make_layer(out_plane, 2, stride=2))  # noqa
        self.bottleneck = nn.Sequential(
            tc.conv1x1(self.in_planes, self.bottleneck_planes),
            nn.BatchNorm2d(self.bottleneck_planes, eps=0.001),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, out_planes: int, blocks: int, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * tc.SpatialResidualBlock.expansion:
            downsample = nn.Sequential(
                tc.conv1x1(self.in_planes, out_planes * tc.SpatialResidualBlock.expansion, stride),
                nn.BatchNorm2d(out_planes * tc.SpatialResidualBlock.expansion),
            )
        layers = [
            tc.SpatialResidualBlock(self.in_planes, out_planes, stride, downsample)
        ]
        self.in_planes = out_planes * tc.SpatialResidualBlock.expansion
        for _ in range(1, blocks):
            layers.append(tc.SpatialResidualBlock(self.in_planes, out_planes, 1))

        return nn.Sequential(*layers)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        b, t, c, h, w = _in.shape
        _out = _in.reshape((b * t, c, h, w))
        _out = self.conv1(_out)
        _out = self.bn1(_out)
        _out = self.relu(_out)
        _out = self.max_pool(_out)

        for layer in self.layers:  # noqa
            _out = layer(_out)
        _out = self.bottleneck(_out)

        _, c, h, w = _out.shape
        # add 1 for the sampling dimension
        return _out.reshape(b, 1, t, c, h, w)
