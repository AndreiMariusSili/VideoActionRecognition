# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
from typing import List, Optional, Tuple

import torch as th
from torch import nn
from torch.nn import functional

from . import _helpers as hp
from .. import options as mo


class Unit3D(nn.Module):
    conv3d: nn.Conv3d
    pad: Optional[nn.ConstantPad3d]
    batch3d: Optional[nn.BatchNorm3d]
    activation: Optional[nn.ReLU]

    def __init__(self, opts: mo.Unit3DOptions):
        super(Unit3D, self).__init__()
        self.pad = None
        self.batch3d = None
        self.activation = None

        if opts.padding not in ['SAME', 'VALID']:
            raise ValueError(f'padding should be in [VALID, SAME] but got {opts.padding}.')

        padding_shape = hp.get_padding_shape(opts.kernel_size, opts.stride)
        if opts.padding == 'SAME':
            simplify_pad, pad_size = hp.simplify_padding(padding_shape)
            if simplify_pad:
                self.conv3d = nn.Conv3d(opts.in_channels, opts.out_channels, opts.kernel_size, stride=opts.stride,
                                        padding=pad_size, bias=opts.use_bias)
            else:
                pad = nn.ConstantPad3d(padding_shape, 0)
                self.pad = pad
                self.conv3d = nn.Conv3d(opts.in_channels, opts.out_channels, opts.kernel_size, stride=opts.stride,
                                        bias=opts.use_bias)
        elif opts.padding == 'VALID':
            self.conv3d = nn.Conv3d(opts.in_channels, opts.out_channels, opts.kernel_size,
                                    padding=padding_shape, stride=opts.stride, bias=opts.use_bias)

        if opts.use_bn:
            self.batch3d = nn.BatchNorm3d(opts.out_channels)
        if opts.activation == 'relu':
            self.activation = nn.functional.relu

    def forward(self, _in: th.Tensor) -> th.Tensor:
        if self.pad is not None:
            _in = self.pad(_in)
        _out = self.conv3d(_in)
        if self.batch3d is not None:
            _out = self.batch3d(_out)
        if self.activation is not None:
            _out = self.activation(_out)

        return _out


class MaxPool3dTFPadding(nn.Module):
    pad: nn.ConstantPad3d
    pool: nn.MaxPool3d

    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = hp.get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = nn.ConstantPad3d(padding_shape, 0)
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        _out = self.pad(_in)
        _out = self.pool(_out)

        return _out


class Mixed(nn.Module):
    branch_0: Unit3D
    branch_1: nn.Sequential
    branch_2: nn.Sequential
    branch_3: nn.Sequential

    def __init__(self, in_channels: int, out_channels: List[int]):
        super(Mixed, self).__init__()

        # Branch 0
        self.branch_0 = Unit3D(mo.Unit3DOptions(in_channels, out_channels[0], kernel_size=(1, 1, 1)))

        # Branch 1
        branch_1_conv1 = Unit3D(mo.Unit3DOptions(in_channels, out_channels[1], kernel_size=(1, 1, 1)))
        branch_1_conv2 = Unit3D(mo.Unit3DOptions(out_channels[1], out_channels[2], kernel_size=(3, 3, 3)))
        self.branch_1 = nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3D(mo.Unit3DOptions(in_channels, out_channels[3], kernel_size=(1, 1, 1)))
        branch_2_conv2 = Unit3D(mo.Unit3DOptions(out_channels[3], out_channels[4], kernel_size=(3, 3, 3)))
        self.branch_2 = nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3D(mo.Unit3DOptions(in_channels, out_channels[5], kernel_size=(1, 1, 1)))
        self.branch_3 = nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        _out_0 = self.branch_0(_in)
        _out_1 = self.branch_1(_in)
        _out_2 = self.branch_2(_in)
        _out_3 = self.branch_3(_in)
        _out = th.cat((_out_0, _out_1, _out_2, _out_3), 1)

        return _out


class Upsample(nn.Module):
    shape: Tuple[int, int, int]

    def __init__(self, shape: Tuple[int, int, int]):
        super(Upsample, self).__init__()

        self.shape = shape

    def forward(self, _in: th.Tensor):
        return functional.interpolate(_in, self.shape)


class Flatten(nn.Module):
    num_channels: int

    def __init__(self, num_channels: int):
        super(Flatten, self).__init__()

        self.num_channels = num_channels

    def forward(self, _in: th.Tensor):
        bs = _in.shape[0]
        return _in.view((bs, self.num_channels))


class Unflatten(nn.Module):
    num_channels: int

    def __init__(self, num_channels: int):
        super(Unflatten, self).__init__()

        self.num_channels = num_channels

    def forward(self, _in: th.Tensor):
        bs = _in.shape[0]
        return _in.view((bs, self.num_channels, 1, 1, 1))


class ReparameterizedSample(nn.Module):
    latent_size: int

    def __init__(self, latent_size: int):
        super(ReparameterizedSample, self).__init__()

        self.latent_size = latent_size

    def forward(self, mean: th.Tensor, log_var: th.Tensor):
        bs = mean.shape[0]
        std = th.exp(th.tensor(0.5) * log_var)
        eps = th.randn((bs, self.latent_size))

        return eps * std + mean
