import typing as tp

import torch as th
from torch import nn

import models.tarn.common as tc


class SpatialResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, downsample: tp.Optional[nn.Module] = None):
        super(SpatialResidualBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = tc.conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = tc.conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, _in: th.Tensor) -> th.Tensor:
        identity = _in

        _out = self.conv1(_in)
        _out = self.bn1(_out)
        _out = self.relu(_out)

        _out = self.conv2(_out)
        _out = self.bn2(_out)

        if self.downsample is not None:
            identity = self.downsample(_in)

        _out += identity
        _out = self.relu(_out)

        return _out


class SpatialResNetEncoder(nn.Module):
    def __init__(self, out_planes: tp.Tuple[int, ...], bottleneck_planes: int):
        super(SpatialResNetEncoder, self).__init__()
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

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.in_planes != planes * SpatialResidualBlock.expansion:
            downsample = nn.Sequential(
                tc.conv1x1(self.in_planes, planes * SpatialResidualBlock.expansion, stride),
                nn.BatchNorm2d(planes * SpatialResidualBlock.expansion),
            )
        layers = [
            SpatialResidualBlock(self.in_planes, planes, stride, downsample)
        ]
        self.in_planes = planes * SpatialResidualBlock.expansion
        for _ in range(1, blocks):
            layers.append(SpatialResidualBlock(self.in_planes, planes, 1))

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
        return _out.reshape(b, t, c, h, w)
