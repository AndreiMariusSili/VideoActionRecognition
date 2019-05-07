# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import torch as th
from torch import nn

from . import _common as cm
from .. import options as mo


class I3DEncoder(nn.Module):
    def __init__(self, latent_size: int, dropout_prob: float = 0.0, name: str = 'i3d_encoder'):
        super(I3DEncoder, self).__init__()
        self.latent_size = latent_size
        self.name = name

        # 1st conv-pool
        opts = mo.Unit3DOptions(out_channels=64, in_channels=3, kernel_size=(7, 7, 7), stride=(2, 2, 2))
        self.conv3d_1a_7x7 = cm.Unit3D(opts)
        self.maxPool3d_2a_3x3 = cm.MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # conv conv
        opts = mo.Unit3DOptions(out_channels=64, in_channels=64, kernel_size=(1, 1, 1))
        self.conv3d_2b_1x1 = cm.Unit3D(opts)
        opts = mo.Unit3DOptions(out_channels=192, in_channels=64, kernel_size=(3, 3, 3))
        self.conv3d_2c_3x3 = cm.Unit3D(opts)
        self.maxPool3d_3a_3x3 = cm.MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        # Mixed_3b
        self.mixed_3b = cm.Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = cm.Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = cm.MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))

        # Mixed 4
        self.mixed_4b = cm.Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = cm.Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = cm.Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = cm.Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = cm.Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = cm.MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Mixed 5
        self.mixed_5b = cm.Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = cm.Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        opts = mo.Unit3DOptions(in_channels=1024, out_channels=self.latent_size, kernel_size=(1, 1, 1),
                                activation='none', use_bias=True, use_bn=False)
        self.mean = cm.Unit3D(opts)
        self.log_var = cm.Unit3D(opts)

    def forward(self, _in: th.Tensor) -> th.tensor:
        _in = _in.transpose(1, 2)
        # print(f'{"encoder input":20s}:\t{_in.shape}')
        for name, module in list(self.named_children())[:-2]:
            _in = module(_in)
            # print(f'{name:20s}:\t{_in.shape}')
        mean = self.mean(_in)
        log_var = self.log_var(_in)
        # print(f'{"encoder mean":20s}:\t{mean.shape}')
        # print(f'{"encoder log_var":20s}:\t{log_var.shape}')

        return mean, log_var


if __name__ == '__main__':
    import os

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    encoder = I3DEncoder(1024, 0.0, 'i3d_encoder')
    print(encoder)
    _in = th.randn((1, 4, 3, 224, 224), dtype=th.float)
    _mean, _log_var = encoder(_in)
