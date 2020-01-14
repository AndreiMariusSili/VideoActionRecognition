# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import torch as th
from torch import nn

import models.common as mc
import models.i3d.common.blocks as ib
from options import model_options as mo


class I3DDecoder(nn.Module):
    def __init__(self, latent_size: int, time_steps: int, name: str = 'i3d_encoder'):
        super(I3DDecoder, self).__init__()
        self.latent_size = latent_size
        self.time_steps = time_steps
        self.name = name

        # latent_size x 1 x 7 x 7
        self.mixed_1a = ib.Mixed(self.latent_size, [256, 160, 320, 32, 128, 128], max_pool=False)
        # 832 x 1 x 7 x 7
        self.mixed_1b = ib.Mixed(832, [256, 160, 320, 32, 128, 128], max_pool=False)
        # 832 x 1 x 7 x 7
        frames = max(1, self.time_steps // 4)
        self.up_4x14x14 = mc.Upsample((frames, 14, 14))
        # 832 x 1 x 14 x 14
        self.mixed_2a = ib.Mixed(832, [112, 144, 288, 32, 64, 64], max_pool=False)
        # 528 x 1 x 14 x 14
        self.mixed_2b = ib.Mixed(528, [128, 128, 256, 24, 64, 64], max_pool=False)
        # 512 x 1 x 14 x 14
        self.mixed_2c = ib.Mixed(512, [160, 112, 224, 24, 64, 64], max_pool=False)
        # 512 x 1 x 14 x 14
        self.mixed_2d = ib.Mixed(512, [192, 96, 208, 16, 48, 64], max_pool=False)
        # 512 x 1 x 14 x 14
        self.mixed_2e = ib.Mixed(512, [128, 128, 192, 32, 96, 64], max_pool=False)
        # 480 x 2 x 14 x 14
        frames = max(1, self.time_steps // 2)
        self.up_8x28x28 = mc.Upsample((frames, 28, 28))
        # 480 x 2 x 28 x 28
        self.mixed_3a = ib.Mixed(480, [64, 96, 128, 16, 32, 32], max_pool=False)
        # 256 x 2 x 28 x 28
        opts = mo.Unit3DOptions(in_channels=256, out_channels=192, kernel_size=[3, 3, 3])
        self.mixed_3b = ib.Unit3D(opts)
        # 192 x 2 x 28 x 28
        self.up_8x56x56 = mc.Upsample((frames, 56, 56))
        # 192 x 2 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=192, out_channels=64, kernel_size=[3, 3, 3])
        self.conv3d_4a = ib.Unit3D(opts)
        # 64 x 2 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=64, out_channels=64, kernel_size=[3, 3, 3])
        self.conv3d_4b = ib.Unit3D(opts)
        # 64 x 2 x 56 x 56
        frames = self.time_steps
        self.up_16x112x112 = mc.Upsample((frames, 112, 112))
        # 64 x 4 x 112 x 112
        opts = mo.Unit3DOptions(in_channels=64, out_channels=32, kernel_size=[3, 5, 5])
        self.conv3d_5a = ib.Unit3D(opts)
        # 3 x 4 x 112 x 112
        opts = mo.Unit3DOptions(in_channels=32, out_channels=32, kernel_size=[3, 3, 3])
        self.conv3d_5b = ib.Unit3D(opts)
        # 3 x 4 x 112 x 112
        self.up_16x224x224 = mc.Upsample((frames, 224, 224))
        # 3 x 4 x 224 x 224
        opts = mo.Unit3DOptions(in_channels=32, out_channels=3, kernel_size=[3, 5, 5])
        self.conv3d_6a = ib.Unit3D(opts)
        # 3 x 4 x 224 x 224
        opts = mo.Unit3DOptions(in_channels=3, out_channels=3, kernel_size=[3, 3, 3],
                                use_bn=False, use_bias=True, activation='none')
        self.conv3d_6b = ib.Unit3D(opts)
        # 3 x 4 x 224 x 224
        self.sigmoid = nn.Sigmoid()

    def forward(self, _in: th.Tensor) -> th.tensor:
        _out = _in
        # print(f'{"decoder input":20s}:\t{_out.shape}')
        for name, module in list(self.named_children())[:-1]:
            _out = module(_out)
            # print(f'{name:20s}:\t{_out.shape}')
        if self.training:
            _out = self.sigmoid(_out)

        return _out.transpose(1, 2)
