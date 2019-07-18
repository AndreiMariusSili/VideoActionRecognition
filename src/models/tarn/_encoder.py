from typing import Optional, Tuple

import torch as th
from torch import nn

from models.tarn import _helpers as hp


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super(ResidualBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = hp.conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = hp.conv3x3(out_planes, out_planes)
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

    def __init__(self, out_planes: Tuple[int, ...]):
        super(SpatialResNetEncoder, self).__init__()
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

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.in_planes != planes * ResidualBlock.expansion:
            downsample = nn.Sequential(
                hp.conv1x1(self.in_planes, planes * ResidualBlock.expansion, stride),
                nn.BatchNorm2d(planes * ResidualBlock.expansion),
            )
        layers = [
            ResidualBlock(self.in_planes, planes, stride, downsample)
        ]
        self.in_planes = planes * ResidualBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_planes, planes, 1))

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
        return _out.reshape(b, t, c, h, w)


class TemporalResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1):
        super(TemporalResidualBlock, self).__init__()

        self.conv1 = hp.conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()
        self.conv2 = hp.conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride

    def forward(self, _in: th.Tensor, _prev: Optional[th.Tensor] = None) -> th.Tensor:
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

        self.out_planes = out_planes
        self.in_planes = in_planes
        self.time_steps = time_steps

        self.temporal = nn.ModuleList()
        for step in range(self.time_steps):
            layer = TemporalResidualBlock(self.in_planes, self.out_planes)
            self.temporal.append(layer)

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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


if __name__ == '__main__':
    import models.helpers

    _input_var = th.randn(2, 4, 3, 224, 224)

    spatial_model = SpatialResNetEncoder((16, 32, 64, 128, 256))
    temporal_model = TemporalResNetEncoder(4, 256, 256)

    spatial_feature_maps = spatial_model(_input_var)
    temporal_feature_map, temporal_feature_maps = temporal_model(spatial_feature_maps)

    print(spatial_feature_maps.shape)
    print(temporal_feature_maps.shape)
    print(f'{models.helpers.count_parameters(spatial_model):,}')
    print(f'{models.helpers.count_parameters(temporal_model):,}')
