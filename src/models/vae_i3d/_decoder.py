# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import torch as th
from torch import nn

from . import _common as cm
from .. import options as mo


class I3DDecoder(nn.Module):
    def __init__(self, latent_size: int, name: str = 'i3d_encoder'):
        super(I3DDecoder, self).__init__()
        self.latent_size = latent_size
        self.name = name

        # 1024 x 1 x1 x 1
        self.up_2x7x7 = cm.Upsample((1, 7, 7))
        # 1024 x 2 x 7 x 7
        self.mixed_1a = cm.Mixed(self.latent_size, [256, 160, 320, 32, 128, 128])
        # 832 x 2 x 7 x 7
        self.mixed_1b = cm.Mixed(832, [256, 160, 320, 32, 128, 128])
        # 832 x 2 x 7 x 7
        self.up_4x14x14 = cm.Upsample((1, 14, 14))
        # 832 x 4 x 14 x 14
        self.mixed_2a = cm.Mixed(832, [112, 144, 288, 32, 64, 64])
        # 528 x 4 x 14 x 14
        self.mixed_2b = cm.Mixed(528, [128, 128, 256, 24, 64, 64])
        # 512 x 4 x 14 x 14
        self.mixed_2c = cm.Mixed(512, [160, 112, 224, 24, 64, 64])
        # 512 x 4 x 14 x 14
        self.mixed_2d = cm.Mixed(512, [192, 96, 208, 16, 48, 64])
        # 512 x 4 x 14 x 14
        self.mixed_2e = cm.Mixed(512, [128, 128, 192, 32, 96, 64])
        # 480 x 4 x 14 x 14
        self.up_8x28x28 = cm.Upsample((2, 28, 28))
        # 480 x 8 x 28 x 28
        self.mixed_3a = cm.Mixed(480, [64, 96, 128, 16, 32, 32])
        # 256 x 8 x 28 x 28
        opts = mo.Unit3DOptions(in_channels=256, out_channels=192, kernel_size=(3, 3, 3))
        self.mixed_3b = cm.Unit3D(opts)
        # 192 x 8 x 28 x 28
        self.up_8x56x56 = cm.Upsample((2, 56, 56))
        # 192 x 8 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=192, out_channels=64, kernel_size=(3, 3, 3))
        self.conv3d_4a = cm.Unit3D(opts)
        # 64 x 8 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=64, out_channels=64, kernel_size=(3, 3, 3))
        self.conv3d_4b = cm.Unit3D(opts)
        # 64 x 8 x 56 x 56
        self.up_16x112x112 = cm.Upsample((4, 112, 112))
        # 64 x 16 x 112 x 112
        opts = mo.Unit3DOptions(in_channels=64, out_channels=3, kernel_size=(5, 5, 5))
        self.conv3d_5a = cm.Unit3D(opts)
        # 3 x 16 x 112 x 112
        opts = mo.Unit3DOptions(in_channels=3, out_channels=3, kernel_size=(3, 3, 3))
        self.conv3d_5b = cm.Unit3D(opts)
        # 3 x 16 x 112 x 112
        self.up_16x224x224 = cm.Upsample((4, 224, 224))
        # 3 x 16 x 224 x 224
        opts = mo.Unit3DOptions(in_channels=3, out_channels=3, kernel_size=(7, 7, 7))
        self.conv3d_6a = cm.Unit3D(opts)
        # 3 x 16 x 224 x 224
        opts = mo.Unit3DOptions(in_channels=3, out_channels=3, kernel_size=(5, 5, 5))
        self.conv3d_6b = cm.Unit3D(opts)
        # 3 x 16 x 224 x 224
        self.sigmoid = nn.Sigmoid()

    def forward(self, _in: th.Tensor) -> th.tensor:
        # print(f'{"decoder input":20s}:\t{_in.shape}')
        for name, module in self.named_children():
            _in = module(_in)
            # print(f'{name:20s}:\t{_in.shape}')

        return _in.transpose(1, 2)


if __name__ == '__main__':
    import os

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    decoder = I3DDecoder(1024)
    print(decoder)
    _in = th.randn((1, 1024, 1, 1, 1), dtype=th.float)
    _out = decoder(_in)
