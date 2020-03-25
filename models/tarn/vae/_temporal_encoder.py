from typing import Tuple

import torch as th
from torch import nn

import models.common
from models.tarn import common as tc

TEMPORAL_ENCODER_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
TEMPORAL_BLOCK_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor]


class VarTemporalResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, stride: int = 1):
        super(VarTemporalResidualBlock, self).__init__()
        self.in_planes = in_planes
        self.stride = stride

        self.conv1 = tc.conv3x3(in_planes, in_planes, stride)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = tc.conv3x3(in_planes, in_planes)
        self.bn2 = nn.BatchNorm2d(in_planes)

        self.mean = tc.conv1x1(in_planes, in_planes)
        self.var = nn.Sequential(
            tc.conv1x1(in_planes, in_planes),
            nn.Softplus()
        )

        self.rsample = models.common.ReparameterizedSample()

    def forward(self, in_spatial: th.Tensor, in_temporal: th.Tensor, num_samples: int) -> TEMPORAL_BLOCK_FORWARD:
        b, s, c, h, w = in_spatial.shape

        out = in_spatial

        # fuse output of previous time step.
        if in_temporal is not None:
            _, s, _, _, _ = in_temporal.shape
            in_spatial = in_spatial.repeat([1, s, 1, 1, 1])
            out = out.repeat([1, s, 1, 1, 1])
            out = out.add(in_temporal)  # noqa
        in_spatial = in_spatial.reshape(b * s, c, h, w)
        out = out.reshape(b * s, c, h, w)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out.add(in_spatial)  # noqa
        out = self.relu(out)

        mean = self.mean(out)
        var = self.var(out) + 1e-5  # Lower bound variance of posterior to prevent infinite density.

        # num = 0 means 1 ML estimate
        out, mean, var = out.reshape(b, s, c, h, w), mean.reshape(b, s, c, h, w), var.reshape(b, s, c, h, w)
        if num_samples:
            z = self.rsample(mean, var, num_samples)
        else:
            z = mean
        z = z.reshape(b, max(num_samples, s), c, h, w)

        return z, mean, var


class VarTemporalResNetEncoder(nn.Module):
    def __init__(self, time_steps: int, in_planes: int):
        super(VarTemporalResNetEncoder, self).__init__()

        self.time_steps = time_steps
        self.in_planes = in_planes

        self.temporal = nn.ModuleList()
        for _ in range(self.time_steps):
            layer = VarTemporalResidualBlock(self.in_planes)
            self.temporal.append(layer)

    def forward(self, _in: th.Tensor, num_samples: int) -> TEMPORAL_ENCODER_FORWARD:
        b, s, t, c, h, w = _in.shape

        # because the first step is deterministic,
        # we can consider it the start of the chain, sample num_samples from there,
        # and then propagate each through the temporal encoder.
        if num_samples:
            num_samples_per_step = [1] * self.time_steps
            num_samples_per_step[0] = num_samples
        else:
            num_samples_per_step = [0] * self.time_steps

        _z = None
        _zs = []
        _means = []
        _vars = []
        for _t in range(self.time_steps):
            _z_prev = _z
            _in_t = _in[:, :, _t, :, :, :]
            _z, _mean, _var = self.temporal[_t](_in_t, _z_prev, num_samples_per_step[_t])  # noqa

            b, s, c, h, w = _z.shape
            _zs.append(_z.reshape(b, s, 1, c, h, w))

            b, s, c, h, w = _mean.shape
            _means.append(_mean.reshape(b, s, 1, c, h, w))
            _vars.append(_var.reshape(b, s, 1, c, h, w))

            b, s, c, h, w = _z.shape

        _means[0] = _means[0].repeat([1, s, 1, 1, 1, 1])
        _vars[0] = _vars[0].repeat([1, s, 1, 1, 1, 1])

        _zs = th.cat(_zs, dim=2)
        _means = th.cat(_means, dim=2)
        _vars = th.cat(_vars, dim=2)

        return _z, _zs, _means, _vars
