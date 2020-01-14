import typing as tp

import torch as th
from torch import nn

import models.tarn.common as tc


class TemporalResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, stride: int = 1):
        super(TemporalResidualBlock, self).__init__()
        self.in_planes = in_planes
        self.stride = stride

        self.conv1 = tc.conv3x3(in_planes, in_planes, stride)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = tc.conv3x3(in_planes, in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)

    def forward(self, in_spatial: th.Tensor, in_temporal: tp.Optional[th.Tensor] = None) -> th.Tensor:
        _out = in_spatial

        if in_temporal is not None:
            _out = in_spatial.add(in_temporal)  # noqa

        _out = self.conv1(_out)
        _out = self.bn1(_out)
        _out = self.relu(_out)

        _out = self.conv2(_out)
        _out = self.bn2(_out)

        _out = _out.add(in_spatial)  # noqa
        _out = self.relu(_out)

        return _out


class TemporalResNetEncoder(nn.Module):
    def __init__(self, time_steps: int, in_planes: int):
        super(TemporalResNetEncoder, self).__init__()

        self.in_planes = in_planes
        self.time_steps = time_steps

        self.temporal = nn.ModuleList()
        for _ in range(self.time_steps):
            layer = TemporalResidualBlock(self.in_planes, self.in_planes)
            self.temporal.append(layer)  # noqa

    def forward(self, _in: th.Tensor) -> tp.Tuple[th.Tensor, th.Tensor]:
        _outs = []
        _out = None

        for _t in range(self.time_steps):
            _out_prev = _out
            _in_t = _in[:, _t, :, :, :]
            _out = self.temporal[_t](_in_t, _out_prev)  # noqa
            b, c, h, w = _out.shape
            _outs.append(_out.reshape(b, 1, c, h, w))
        _outs = th.cat(_outs, dim=1)

        return _out, _outs
