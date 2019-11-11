from typing import Tuple

import torch as th
from torch import nn

import models.common
import models.i3d.common.blocks
import models.tarn.common as tcm

TEMPORAL_ENCODER_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
TEMPORAL_BLOCK_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor]


class VariationalSpatialResNetEncoder(nn.Module):

    def __init__(self, out_planes: Tuple[int, ...]):
        super(VariationalSpatialResNetEncoder, self).__init__()
        self.in_planes = out_planes[0]
        self.out_planes = out_planes
        self.conv1 = nn.Conv2d(3, self.out_planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_planes[0])
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(self.out_planes[1], 2))
        for out_plane in self.out_planes[2:]:
            self.layers.append(self._make_layer(out_plane, 2, stride=2))

    def _make_layer(self, out_planes: int, blocks: int, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != out_planes * tcm.SpatialResidualBlock.expansion:
            downsample = nn.Sequential(
                tcm.conv1x1(self.in_planes, out_planes * tcm.SpatialResidualBlock.expansion, stride),
                nn.BatchNorm2d(out_planes * tcm.SpatialResidualBlock.expansion),
            )
        layers = [
            tcm.SpatialResidualBlock(self.in_planes, out_planes, stride, downsample)
        ]
        self.in_planes = out_planes * tcm.SpatialResidualBlock.expansion
        for _ in range(1, blocks):
            layers.append(tcm.SpatialResidualBlock(self.in_planes, out_planes, 1))

        return nn.Sequential(*layers)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        b, t, c, h, w = _in.shape
        _out = _in.reshape((b * t, c, h, w))
        _out = self.conv1(_out)
        _out = self.bn1(_out)
        _out = self.relu(_out)
        _out = self.max_pool(_out)

        for layer in self.layers:
            _out = layer(_out)

        _, c, h, w = _out.shape
        # add 1 for the sampling dimension
        return _out.reshape(b, 1, t, c, h, w)


class VariationalTemporalResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super(VariationalTemporalResidualBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.conv1 = tcm.conv3x3(out_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()
        self.conv2 = tcm.conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.mean = tcm.conv1x1(out_planes, out_planes)
        self.var = nn.Sequential(
            tcm.conv1x1(out_planes, out_planes),
            nn.Softplus()
        )

        self.rsample = models.common.ReparameterizedSample()

        self.stride = stride

    def forward(self, _in: th.Tensor, _prev: th.Tensor, num_samples: int) -> TEMPORAL_BLOCK_FORWARD:
        b, s, c, h, w = _in.shape

        # fuse output of previous time step.
        if _prev is not None:
            b, s, c, h, w = _prev.shape
            _in = _in.repeat([1, s, 1, 1, 1])
            _out = _in.add(_prev)

        _out = _in.reshape(b * s, c, h, w)

        _out = self.conv1(_out)
        _out = self.bn1(_out)
        _out = self.relu(_out)

        _mean = self.mean(_out)
        _var = self.var(_out) + 1e-5  # Lower bound variance of posterior to prevent infinite density.

        # num = 0 means 1 ML estimate
        _mean, _var = _mean.reshape(b, s, c, h, w), _var.reshape(b, s, c, h, w)
        if num_samples:
            _z = self.rsample(_mean, _var, num_samples).reshape(b * max(num_samples, s), c, h, w)
        else:
            _z = _mean.reshape(b * s, c, h, w)

        _z = self.bn2(_z).reshape(b, max(num_samples, s), c, h, w)
        _z = _z.add(_in)
        _z = self.relu(_z)

        return _z, _mean, _var


class VariationalTemporalResNetEncoder(nn.Module):
    def __init__(self, time_steps: int, in_planes: int, out_planes: int):
        super(VariationalTemporalResNetEncoder, self).__init__()

        self.time_steps = time_steps
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.temporal = nn.ModuleList()
        for step in range(self.time_steps):
            layer = VariationalTemporalResidualBlock(self.in_planes, self.out_planes)
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
            _z, _mean, _var = self.temporal[_t](_in_t, _z_prev, num_samples_per_step[_t])

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


if __name__ == '__main__':
    import models.helpers
    import torch as th

    _input_var = th.randn(3, 4, 3, 224, 224)

    spatial_model = VariationalSpatialResNetEncoder((16, 32, 64, 128, 256))
    __f = spatial_model(_input_var)

    print(f'Spatial feature maps:\t{__f.shape}')
    print(f'Spatial model size:\t{models.helpers.count_parameters(spatial_model):,}')

    temporal_model = VariationalTemporalResNetEncoder(4, 256, 256)
    __z, _, _, _ = temporal_model(__f, 10)

    print(f'Temporal feature maps:\t{__z.shape}')
    print(f'Temporal model size:\t{models.helpers.count_parameters(temporal_model):,}')
