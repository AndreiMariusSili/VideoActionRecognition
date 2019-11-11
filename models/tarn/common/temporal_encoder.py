import typing as tp

import torch as th
from torch import nn

import models.tarn.common as tc


class TemporalResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super(TemporalResidualBlock, self).__init__()

        if in_planes != out_planes:
            self.bottleneck = tc.conv1x1(in_planes, out_planes)

        self.conv1 = tc.conv3x3(out_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()
        self.conv2 = tc.conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride

    def forward(self, _in: th.Tensor, _prev: tp.Optional[th.Tensor] = None) -> th.Tensor:
        if self.bottleneck is not None:
            _in = self.bottleneck(_in)

        _out = _in

        if _prev is not None:
            _out = _in.add(_prev)

        _out = self.conv1(_out)
        _out = self.bn1(_out)
        _out = self.relu(_out)

        _out = self.conv2(_out)
        _out = self.bn2(_out)

        _out = _out.add(_in)
        _out = self.relu(_out)

        return _out


class TemporalResNetEncoder(nn.Module):
    def __init__(self, time_steps: int, in_planes: int, out_planes: int):
        super(TemporalResNetEncoder, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.time_steps = time_steps

        self.temporal = nn.ModuleList()
        for step in range(self.time_steps):
            layer = TemporalResidualBlock(self.in_planes, self.out_planes)
            self.temporal.append(layer)

    def forward(self, _in: th.Tensor) -> tp.Tuple[th.Tensor, th.Tensor]:
        _outs = []
        _out = None

        for _t in range(self.time_steps):
            _out_prev = _out
            _in_t = _in[:, _t, :, :, :]
            _out = self.temporal[_t](_in_t, _out_prev)
            b, c, h, w = _out.shape
            _outs.append(_out.reshape(b, 1, c, h, w))
        _outs = th.cat(_outs, dim=1)

        return _out, _outs
