# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import typing as tp

import torch as th
from torch import nn

import models.i3d.common.blocks as ib
from options import model_options as mo


class I3DEncoder(nn.Module):
    def __init__(self, embed_size: tp.Optional[int], name: str = 'i3d_encoder'):
        super(I3DEncoder, self).__init__()
        self.embed_size = embed_size
        self.name = name

        # 3 x 4 x 224 x 224
        opts = mo.Unit3DOptions(out_channels=64, in_channels=3, kernel_size=[7, 7, 7], stride=[2, 2, 2], padding='SAME')
        self.conv3d_1a_7x7 = ib.Unit3D(opts)
        # 64 x 2 x 112 x 112
        self.maxPool3d_2a_3x3 = ib.MaxPool3dTFPadding(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding='SAME')
        # 64 x 2 x 56 x 56
        opts = mo.Unit3DOptions(out_channels=64, in_channels=64, kernel_size=[1, 1, 1], padding='SAME')
        self.conv3d_2b_1x1 = ib.Unit3D(opts)
        # 64 x 2 x 56 x 56
        opts = mo.Unit3DOptions(out_channels=192, in_channels=64, kernel_size=[3, 3, 3], padding='SAME')
        self.conv3d_2c_3x3 = ib.Unit3D(opts)
        # 192 x 2 x 56 x 56
        self.maxPool3d_3a_3x3 = ib.MaxPool3dTFPadding(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding='SAME')
        # 192 x 2 x 28 x 28
        self.mixed_3b = ib.Mixed(192, [64, 96, 128, 16, 32, 32])
        # 256 x 2 x 28 x 28
        self.mixed_3c = ib.Mixed(256, [128, 128, 192, 32, 96, 64])
        # 480 x 2 x 28 x 28
        self.maxPool3d_4a_3x3 = ib.MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')
        # 480 x 1 x 14 x 14
        self.mixed_4b = ib.Mixed(480, [192, 96, 208, 16, 48, 64])
        # 512 x 1 x 14 x 14
        self.mixed_4c = ib.Mixed(512, [160, 112, 224, 24, 64, 64])
        # 512 x 1 x 14 x 14
        self.mixed_4d = ib.Mixed(512, [128, 128, 256, 24, 64, 64])
        # 512 x 1 x 14 x 14
        self.mixed_4e = ib.Mixed(512, [112, 144, 288, 32, 64, 64])
        # 528 x 1 x 14 x 14
        self.mixed_4f = ib.Mixed(528, [256, 160, 320, 32, 128, 128])
        # 832 x 1 x 14 x 14
        self.maxPool3d_5a_2x2 = ib.MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 832 x 1 x 7 x 7
        self.mixed_5b = ib.Mixed(832, [256, 160, 320, 32, 128, 128])
        # 832 x 1 x 7 x 7
        self.mixed_5c = ib.Mixed(832, [384, 192, 384, 48, 128, 128])
        # 1024 x 1 x 7 x 7
        if self.embed_size:
            opts = mo.Unit3DOptions(out_channels=self.embed_size, in_channels=1024, kernel_size=[1, 1, 1],
                                    stride=[1, 1, 1], activation='none', padding='SAME', use_bias=False, use_bn=False)
            self.embed = ib.Unit3D(opts)

    def forward(self, _in: th.Tensor) -> th.tensor:
        _out = _in.transpose(1, 2)
        # print(f'{"encoder input":20s}:\t{_out.shape}')
        for name, module in list(self.named_children()):
            _out = module(_out)
            # print(f'{name:20s}:\t{_out.shape}')

        return _out
