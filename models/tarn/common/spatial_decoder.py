import typing as t

import torch as th
from torch import nn

import models.tarn.common as tc


class TransSpatialResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, upsample: t.Optional[nn.Module] = None):
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
    def __init__(self, in_planes: int, out_planes: t.Tuple[int, ...], flow: bool):
        super(SpatialResNetDecoder, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.flow = flow

        self.debottleneck = nn.Sequential(
            tc.conv1x1(self.in_planes, self.out_planes[0]),
            nn.BatchNorm2d(self.out_planes[0]),
            nn.ReLU(inplace=True),
        )
        self.in_planes = self.out_planes[0]
        if self.flow:
            self.in_planes *= 2

        self.layers = nn.ModuleList()
        for out_plane in self.out_planes[1:-1]:
            self.layers.append(self._make_transpose(out_plane, 2, stride=2))
        self.layers.append(self._make_transpose(self.out_planes[-1], 2, stride=1))

        if self.flow:
            self.final = nn.ConvTranspose2d(self.out_planes[-1], 2, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.final = nn.ConvTranspose2d(self.out_planes[-1], 3, kernel_size=3, stride=1, padding=1, bias=True)
            self.final_act = nn.Sigmoid()

    def _make_transpose(self, out_planes: int, blocks: int, stride: int = 1) -> nn.Module:
        if stride != 1:
            upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
                tc.conv3x3(self.in_planes, out_planes * TransSpatialResidualBlock.expansion, 1),
                nn.BatchNorm2d(out_planes),
            )
        elif self.in_planes != out_planes:
            upsample = nn.Sequential(
                tc.conv1x1(self.in_planes, out_planes * TransSpatialResidualBlock.expansion, stride),
                nn.BatchNorm2d(out_planes),
            )
        else:
            upsample = None
        layers = []
        for _ in range(0, blocks - 1):
            layers.append(TransSpatialResidualBlock(self.in_planes, self.in_planes))
        layers.append(TransSpatialResidualBlock(self.in_planes, out_planes, stride, upsample))
        self.in_planes = out_planes * TransSpatialResidualBlock.expansion
        if self.flow:
            self.in_planes *= 2

        return nn.Sequential(*layers)

    def forward(self, _in: th.Tensor, _spatial_mid_outs: t.List[th.Tensor]) -> th.Tensor:
        b, _t, c, h, w = _in.shape
        _out = _in.reshape(b * _t, c, h, w)

        _spatial_mid_outs = list(reversed(_spatial_mid_outs))

        _out = self.debottleneck(_out)

        for layer, _spatial_mid_out in zip(self.layers, _spatial_mid_outs):
            if self.flow:
                _out = th.cat([_out, _spatial_mid_out], dim=1)
            _out = layer(_out)

        _out = self.final(_out)
        _, c, h, w = _out.shape
        _out = _out.reshape(b, _t, c, h, w)
        if self.flow:
            _out = _out[:, 1:, :, :]
        else:
            _out = self.final_act(_out)

        return _out
