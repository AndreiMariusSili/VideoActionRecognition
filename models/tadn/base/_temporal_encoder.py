import typing as tp

import torch as th
from torch import nn


class TemporalDenseBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, drop_rate: float, step: int):
        super(TemporalDenseBlock, self).__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.drop_rate = drop_rate
        self.step = step

        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_planes, self.out_planes, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(self.drop_rate)
        )

    def forward(self, _in: th.Tensor, _prev: tp.Optional[th.Tensor]) -> th.Tensor:
        _out = _in
        if _prev is not None:
            # cat across channels
            _out = th.cat((_in, _prev), dim=1)

        _out = self.layers(_out)

        if self.step > 0:
            # concatenate outputs of other time-steps
            _out = th.cat((_prev, _out), dim=1)

        return _out


# noinspection PyUnresolvedReferences
class TemporalDenseNetEncoder(nn.Module):
    # n_channels, step, self.growth_rate, self.n_input_plane, self.drop_rate

    def __init__(self, in_planes: int, time_steps: int, growth_rate: int, drop_rate: float):
        super(TemporalDenseNetEncoder, self).__init__()

        self.in_planes = in_planes
        self.time_steps = time_steps
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

        self.temporal = nn.ModuleList()
        temporal_in_planes = self.in_planes
        for step in range(self.time_steps):
            layer = TemporalDenseBlock(temporal_in_planes, self.growth_rate, self.drop_rate, step)
            temporal_in_planes += growth_rate
            self.temporal.append(layer)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        _out = None

        for _t in range(self.time_steps):
            _out_prev = _out
            _in_t = _in[:, _t, :, :, :]
            _out = self.temporal[_t](_in_t, _out_prev)

        return _out
