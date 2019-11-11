import typing as tp

import torch as th
from torch import nn

import models.tarn.common as tc


class TransSpatialResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, upsample: tp.Optional[nn.Module] = None):
        super(TransSpatialResidualBlock, self).__init__()
        self.conv1 = tc.conv3x3(in_planes, in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.conv2 = tc.conv3x3(in_planes, out_planes, stride)

        if upsample is not None and stride != 1:
            self.conv2 = tc.t_conv3x3(in_planes, out_planes, stride=stride, padding=1)
        else:
            self.conv2 = tc.conv3x3(in_planes, out_planes, stride)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, _in: th.Tensor) -> th.Tensor:
        identity = _in

        _out = self.conv1(_in)
        _out = self.bn1(_out)
        _out = self.relu(_out)

        _out = self.conv2(_out)
        _out = self.bn2(_out)

        if self.upsample is not None:
            identity = self.upsample(_in)

        _out += identity
        _out = self.relu(_out)

        return _out


class SpatialResNetDecoder(nn.Module):
    def __init__(self, in_planes: int, out_planes: tp.Tuple[int, ...]):
        super(SpatialResNetDecoder, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes

        self.layers = nn.ModuleList()
        for out_plane in self.out_planes[0:-2]:
            self.layers.append(self._make_transpose(out_plane, 2, stride=2))
        self.layers.append(self._make_transpose(self.out_planes[-2], 1, stride=1))

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.out_planes[-2], self.out_planes[-1], kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(self.out_planes[-1]),
            nn.ReLU(inplace=True),
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(self.out_planes[-1], 3, kernel_size=7, stride=2, padding=3, bias=False,
                               output_padding=1),
            nn.Sigmoid()
        )

    def _make_transpose(self, out_planes: int, blocks: int, stride: int = 1) -> nn.Module:
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                tc.t_conv3x3(self.in_planes, out_planes * TransSpatialResidualBlock.expansion, stride),
                nn.BatchNorm2d(out_planes),
            )
        elif self.in_planes != out_planes:
            upsample = nn.Sequential(
                tc.conv1x1(self.in_planes, out_planes * TransSpatialResidualBlock.expansion, stride),
                nn.BatchNorm2d(out_planes),
            )

        layers = []
        for i in range(1, blocks):
            layers.append(TransSpatialResidualBlock(self.in_planes, self.in_planes))

        layers.append(TransSpatialResidualBlock(self.in_planes, out_planes, stride, upsample))
        self.in_planes = out_planes * TransSpatialResidualBlock.expansion

        return nn.Sequential(*layers)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        b, t, c, h, w = _in.shape
        _out = _in.reshape(b * t, c, h, w)

        for layer in self.layers:
            _out = layer(_out)

        _out = self.upsample(_out)
        _out = self.final(_out)

        _, c, h, w = _out.shape
        return _out.reshape(b, t, c, h, w)
