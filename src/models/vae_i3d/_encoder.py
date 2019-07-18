# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
from typing import Tuple

import torch as th
from torch import nn

from models import common as cm
from options import model_options as mo


class I3DEncoder(nn.Module):

    def __init__(self, latent_planes: int, name: str = 'i3d_encoder'):
        super(I3DEncoder, self).__init__()
        self.latent_planes = latent_planes
        self.name = name

        # 3 x 4 x 224 x 224
        opts = mo.Unit3DOptions(out_channels=64, in_channels=3, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                                padding='TEMPORAL_VALID')
        self.conv3d_1a_7x7 = cm.Unit3D(opts)
        # 64 x 2 x 112 x 112
        self.maxPool3d_2a_3x3 = cm.MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # 64 x 2 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=64, out_channels=64, kernel_size=(1, 1, 1))
        self.conv3d_2b_1x1 = cm.Unit3D(opts)
        # 64 x 2 x 56 x 56
        opts = mo.Unit3DOptions(in_channels=64, out_channels=192, kernel_size=(3, 3, 3))
        self.conv3d_2c_3x3 = cm.Unit3D(opts)
        # 192 x 2 x 56 x 56
        self.maxPool3d_3a_3x3 = cm.MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # 192 x 2 x 28 x 28
        self.mixed_3b = cm.Mixed(192, [64, 96, 128, 16, 32, 32])
        # 256 x 2 x 28 x 28
        self.mixed_3c = cm.Mixed(256, [128, 128, 192, 32, 96, 64])
        # 480 x 2 x 28 x 28
        self.maxPool3d_4a_3x3 = cm.MaxPool3dTFPadding(kernel_size=(2, 3, 3), stride=(2, 2, 2))
        # 480 x 1 x 14 x 14
        self.mixed_4b = cm.Mixed(480, [192, 96, 208, 16, 48, 64], [1, 1, 1, 1, 1, 1, 1])
        # 512 x 1 x 14 x 14
        self.mixed_4c = cm.Mixed(512, [160, 112, 224, 24, 64, 64], [1, 1, 1, 1, 1, 1, 1])
        # 512 x 1 x 14 x 14
        self.mixed_4d = cm.Mixed(512, [128, 128, 256, 24, 64, 64], [1, 1, 1, 1, 1, 1, 1])
        # 512 x 1 x 14 x 14
        self.mixed_4e = cm.Mixed(512, [112, 144, 288, 32, 64, 64], [1, 1, 1, 1, 1, 1, 1])
        # 528 x 1 x 14 x 14
        self.mixed_4f = cm.Mixed(528, [256, 160, 320, 32, 128, 128], [1, 1, 1, 1, 1, 1, 1])
        # 832 x 1 x 14 x 14
        self.maxPool3d_5a_2x2 = cm.MaxPool3dTFPadding(kernel_size=(1, 2, 2), stride=(2, 2, 2))
        # 832 x 1 x 7 x 7
        self.mixed_5b = cm.Mixed(832, [256, 160, 320, 32, 128, 128], [1, 1, 1, 1, 1, 1, 1])
        # 832 x 1 x 7 x 7
        self.mixed_5c = cm.Mixed(832, [384, 192, 384, 48, 128, 128], [1, 1, 1, 1, 1, 1, 1])
        # 1024 x 1 x 7 x 7
        opts = mo.Unit3DOptions(in_channels=1024, out_channels=self.latent_planes, kernel_size=(1, 1, 1),
                                activation='none', use_bias=False, use_bn=False)
        self.mean = cm.Unit3D(opts)
        # latent_size x 1 x 7 x 7
        self.var = nn.Sequential(
            cm.Unit3D(opts),
            nn.Softplus()
        )
        # latent_size x 1 x 7 x 7
        self.rsample = cm.ReparameterizedSample()
        # latent_size x 1 x 7 x 7

    def forward(self, _in: th.Tensor, num_samples: int) -> Tuple[th.tensor, th.Tensor, th.Tensor]:
        _in = _in.transpose(1, 2).contiguous()
        # print(f'{"encoder input":20s}:\t{_in.shape}')
        for name, module in list(self.named_children())[:-3]:
            _in = module(_in)
            # print(f'{name:20s}:\t{_in.shape}')

        _mean = self.mean(_in)
        _var = self.var(_in) + 1e-5  # Lower bound variance of posterior to prevent infinite density.

        # num = 0 means 1 ML estimate
        b, c, t, h, w = _mean.shape
        if num_samples:
            _z = self.rsample(_mean, _var, num_samples)
        else:
            _z = _mean.reshape(b, 1, c, t, h, w)
        # print(f'{"encoder log_var":20s}:\t{_var.shape}\n{"encoder mean":20s}:\t{_mean.shape}')

        return _z, _mean, _var


if __name__ == '__main__':
    import models.helpers

    x = th.randn((1, 4, 3, 224, 224), dtype=th.float)

    encoder = I3DEncoder(1024, 'i3d_encoder')
    print(encoder)

    z, mean, var = encoder(x, 2)
    print(f'Z:\t{z.shape}')
    print(f'Mean:\t{mean.shape}')
    print(f'Var:\t{var.shape}')
    print(f'Model size:\t{models.helpers.count_parameters(encoder):,}')
