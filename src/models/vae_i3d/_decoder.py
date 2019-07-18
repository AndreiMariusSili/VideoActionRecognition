# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
from typing import List

import torch as th
from torch import nn

import constants as ct
from models import common as cm
from options import model_options as mo


class Normalize(nn.Module):
    def __init__(self, by: int):
        super(Normalize, self).__init__()
        self.by = nn.Parameter(th.tensor(by, dtype=th.float), requires_grad=False)

    def forward(self, _in: th.Tensor):
        return _in.div(self.by)


class Standardize(nn.Module):
    def __init__(self, means: List[float], stds: List[float]):
        super(Standardize, self).__init__()
        self.means = nn.Parameter(th.tensor(means, dtype=th.float).reshape((1, 3, 1, 1, 1)), requires_grad=False)
        self.stds = nn.Parameter(th.tensor(stds, dtype=th.float).reshape((1, 3, 1, 1, 1)), requires_grad=False)

    def forward(self, _in: th.Tensor):
        return _in.sub(self.means).div(self.stds)


class I3DDecoder(nn.Module):
    def __init__(self, latent_size: int, name: str = 'i3d_encoder'):
        super(I3DDecoder, self).__init__()
        self.latent_size = latent_size
        self.name = name

        # latent_size x 1 x 1 x 1
        self.up_1x7x7 = cm.Upsample((1, 7, 7))
        # latent_size x 1 x 7 x 7
        chs = [256, 160, 320, 32, 128, 128]
        self.mixed_1a = cm.Mixed(self.latent_size, chs, [1, 1, 1, 1, 1, 1, 1], False)
        # 832 x 1 x 7 x 7
        chs = [256, 160, 320, 32, 128, 128]
        self.mixed_1b = cm.Mixed(832, chs, [1, 1, 1, 1, 1, 1, 1], False)
        # 832 x 1 x 7 x 7
        self.up_1x14x14 = cm.Upsample((1, 14, 14))
        # 832 x 1 x 14 x 14
        chs = [112, 144, 288, 32, 64, 64]
        self.mixed_2a = cm.Mixed(832, chs, [1, 1, 1, 1, 1, 1, 1], False)
        # 528 x 1 x 14 x 14
        chs = [128, 128, 256, 24, 64, 64]
        self.mixed_2b = cm.Mixed(528, chs, [1, 1, 1, 1, 1, 1, 1], False)
        # 512 x 1 x 14 x 14
        chs = [160, 112, 224, 24, 64, 64]
        self.mixed_2c = cm.Mixed(512, chs, [1, 1, 1, 1, 1, 1, 1], False)
        # 512 x 1 x 14 x 14
        chs = [192, 96, 208, 16, 48, 64]
        self.mixed_2d = cm.Mixed(512, chs, [1, 1, 1, 1, 1, 1, 1], False)
        # 512 x 1 x 14 x 14
        chs = [128, 128, 192, 32, 96, 64]
        self.mixed_2e = cm.Mixed(512, chs, [1, 1, 1, 1, 1, 1, 1], False)
        # 480 x 2 x 14 x 14
        self.up_2x28x28 = cm.Upsample((2, 28, 28))
        # 480 x 2 x 28 x 28
        chs = [64, 96, 128, 16, 32, 32]
        self.mixed_3a = cm.Mixed(480, chs, max_pool=False)
        # 256 x 2 x 28 x 28
        opts = mo.Unit3DOptions(in_channels=256, out_channels=152, kernel_size=(3, 3, 3))
        self.mixed_3b = cm.Unit3D(opts)
        # 192 x 2 x 28 x 28
        self.up_2x56x56 = cm.Upsample((2, 56, 56))
        # 192 x 2 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=152, out_channels=64, kernel_size=(3, 3, 3))
        self.conv3d_4a = cm.Unit3D(opts)
        # 64 x 2 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=64, out_channels=64, kernel_size=(3, 3, 3))
        self.conv3d_4b = cm.Unit3D(opts)
        # 64 x 2 x 56 x 56
        self.up_4x112x112 = cm.Upsample((4, 112, 112))
        # 64 x 4 x 112 x 112
        opts = mo.Unit3DOptions(in_channels=64, out_channels=3, kernel_size=(3, 5, 5))
        self.conv3d_5a = cm.Unit3D(opts)
        # 3 x 4 x 112 x 112
        opts = mo.Unit3DOptions(in_channels=3, out_channels=3, kernel_size=(3, 3, 3))
        self.conv3d_5b = cm.Unit3D(opts)
        # 3 x 4 x 112 x 112
        self.up_4x224x224 = cm.Upsample((4, 224, 224))
        # 3 x 4 x 224 x 224
        opts = mo.Unit3DOptions(in_channels=3, out_channels=3, kernel_size=(3, 5, 5))
        self.conv3d_6a = cm.Unit3D(opts)
        # 3 x 4 x 224 x 224
        opts = mo.Unit3DOptions(in_channels=3, out_channels=3, kernel_size=(3, 3, 3),
                                use_bn=False, use_bias=True, activation='none')
        self.conv3d_6b = cm.Unit3D(opts)
        # 3 x 4 x 224 x 224
        self.sigmoid = nn.Sigmoid()
        self.standardize = Standardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS)

    def forward(self, _in: th.Tensor) -> th.tensor:
        b, s, c, t, h, w = _in.shape

        # Just get the decoding of the average latent variable for efficiency.
        _in = _in.mean(dim=1).reshape(b, c, t, h, w)

        # print(f'{"decoder input":20s}:\t{_in.shape}')
        for name, module in self.named_children():
            _in = module(_in)
            # print(f'{name:20s}:\t{_in.shape}')

        return _in.transpose(1, 2).contiguous()


if __name__ == '__main__':
    import models.helpers

    decoder = I3DDecoder(1024)
    print(decoder)
    x = th.randn((1, 2, 1024, 1, 7, 7), dtype=th.float)
    _out = decoder(x)
    print(f'Reconstruction: {_out.shape}')
    print(f'Model Size: {models.helpers.count_parameters(decoder):,}')
