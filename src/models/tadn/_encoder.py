from typing import Optional

import torch as th
from torch import nn


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class Inception(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        return th.cat([branch1, branch2, branch3, branch4], 1)


class SpatialInceptionV1Encoder(nn.Module):

    def __init__(self, dropout_prob: float, out_planes: int):
        super(SpatialInceptionV1Encoder, self).__init__()

        self.dropout_prob = dropout_prob
        self.out_planes = out_planes

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.max_pool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.max_pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.bottleneck = BasicConv2d(1024, self.out_planes, kernel_size=1)

    def forward(self, _in: th.Tensor):
        b, t, c, h, w = _in.shape
        _in = _in.reshape(b * t, c, h, w)

        # N x 3 x 224 x 224
        _in = self.conv1(_in)
        # N x 64 x 112 x 112
        _in = self.max_pool1(_in)
        # N x 64 x 56 x 56
        _in = self.conv2(_in)
        # N x 64 x 56 x 56
        _in = self.conv3(_in)
        # N x 192 x 56 x 56
        _in = self.max_pool2(_in)

        # N x 192 x 28 x 28
        _in = self.inception3a(_in)
        # N x 256 x 28 x 28
        _in = self.inception3b(_in)
        # N x 480 x 28 x 28
        _in = self.max_pool3(_in)
        # N x 480 x 14 x 14
        _in = self.inception4a(_in)
        # N x 512 x 14 x 14

        _in = self.inception4b(_in)
        # N x 512 x 14 x 14
        _in = self.inception4c(_in)
        # N x 512 x 14 x 14
        _in = self.inception4d(_in)
        # N x 528 x 14 x 14

        _in = self.inception4e(_in)
        # N x 832 x 14 x 14
        _in = self.max_pool4(_in)
        # N x 832 x 7 x 7
        _in = self.inception5a(_in)
        # N x 832 x 7 x 7
        _in = self.inception5b(_in)
        # N x 1024 x 7 x 7
        _in = self.bottleneck(_in)
        # N x 256 x 7 x 7

        _, c, h, w = _in.shape
        return _in.reshape(b, t, c, h, w)


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

    def forward(self, _in: th.Tensor, _prev: Optional[th.Tensor]) -> th.Tensor:
        _out = _in
        if _prev is not None:
            # cat across channels
            _out = th.cat((_in, _prev), dim=1)

        _out = self.layers(_out)

        if self.step > 0:
            # concatenate outputs of other time-steps
            _out = th.cat((_prev, _out), dim=1)

        return _out


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
