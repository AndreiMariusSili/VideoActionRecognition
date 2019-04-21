# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
from typing import Tuple, List, Any
from torch import nn
import torch as th

from models.i3d import _helpers as hp
import constants as ct


class Unit3D(nn.Module):
    conv3d: nn.Conv3d
    pad: nn.ConstantPad3d
    batch3d: nn.BatchNorm3d
    activation: nn.ReLU

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (1, 1, 1),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 activation: str = 'relu',
                 padding: str = 'SAME',
                 use_bias: bool = False,
                 use_bn: bool = True):
        super(Unit3D, self).__init__()
        self.pad = None
        self.batch3d = None
        self.activation = None

        if padding not in ['SAME', 'VALID']:
            raise ValueError(f'padding should be in [VALID, SAME] but got {padding}.')

        padding_shape = hp.get_padding_shape(kernel_size, stride)
        if padding == 'SAME':
            simplify_pad, pad_size = hp.simplify_padding(padding_shape)
            if simplify_pad:
                self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                                        padding=pad_size, bias=use_bias)
            else:
                pad = nn.ConstantPad3d(padding_shape, 0)
                self.pad = pad
                self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, bias=use_bias)
        elif padding == 'VALID':
            self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size,
                                    padding=padding_shape, stride=stride, bias=use_bias)

        if use_bn:
            self.batch3d = nn.BatchNorm3d(out_channels)
        if activation == 'relu':
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
    branch2: nn.Sequential
    branch3: nn.Sequential

    def __init__(self, in_channels: int, out_channels: List[int]):
        super(Mixed, self).__init__()

        # Branch 0
        self.branch_0 = Unit3D(in_channels, out_channels[0], kernel_size=(1, 1, 1))

        # Branch 1
        branch_1_conv1 = Unit3D(in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3D(out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = nn.Sequential(branch_1_conv1, branch_1_conv2)

        # Branch 2
        branch_2_conv1 = Unit3D(in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3D(out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = nn.Sequential(branch_2_conv1, branch_2_conv2)

        # Branch3
        branch_3_pool = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='SAME')
        branch_3_conv2 = Unit3D(in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        _out_0 = self.branch_0(_in)
        _out_1 = self.branch_1(_in)
        _out_2 = self.branch_2(_in)
        _out_3 = self.branch_3(_in)
        _out = th.cat((_out_0, _out_1, _out_2, _out_3), 1)

        return _out


class I3D(nn.Module):
    def __init__(self, num_classes: int, modality: str = 'rgb', dropout_prob: float = 0.0, name: str = 'inception'):
        super(I3D, self).__init__()
        if modality not in ['rgb', 'flow']:
            raise ValueError(f'Unknown modality {modality}. Possible values: [rgb, flow].')

        self.name = name
        self.num_classes = num_classes
        if modality == 'rgb':
            in_channels = 3
        else:
            in_channels = 2
        self.modality = modality

        # 1st conv-pool
        self.conv3d_1a_7x7 = Unit3D(out_channels=64, in_channels=in_channels, kernel_size=(7, 7, 7), stride=(2, 2, 2))
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        # conv conv
        self.conv3d_2b_1x1 = Unit3D(out_channels=64, in_channels=64, kernel_size=(1, 1, 1))
        self.conv3d_2c_3x3 = Unit3D(out_channels=192, in_channels=64, kernel_size=(3, 3, 3))
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2))

        # Mixed_3b
        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2))

        # Mixed 4
        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Mixed 5
        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.avg_pool = nn.AvgPool3d((2, 7, 7), (1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.conv3d_0c_1x1 = Unit3D(in_channels=1024, out_channels=400, kernel_size=(1, 1, 1),
                                    activation='none', use_bias=True, use_bn=False)
        self.softmax = nn.Softmax(dim=1)

        if ct.I3D_PT_RGB_CHECKPOINT.exists() and self.modality == 'rgb':
            self.load_state_dict(th.load(ct.I3D_PT_RGB_CHECKPOINT))

        self.conv3d_0c_1x1 = Unit3D(in_channels=1024, out_channels=self.num_classes, kernel_size=(1, 1, 1),
                                    activation='none', use_bias=True, use_bn=False)

    def forward(self, _in: th.Tensor) -> th.tensor:
        _in = _in.transpose(1, 2)
        out = self.conv3d_1a_7x7(_in)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)
        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)
        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)
        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)
        out = self.avg_pool(out)
        out = self.dropout(out)
        out = self.conv3d_0c_1x1(out)
        out = out.squeeze(3)
        out = out.squeeze(3)
        out = out.mean(2)
        out_logits = out

        return out_logits

    def load_tf_weights(self, sess: Any) -> None:
        if self.modality not in ['rgb', 'flow']:
            raise ValueError(f'Unknown modality {self.modality}')

        state_dict = {}
        if self.modality == 'rgb':
            prefix = 'RGB/inception_i3d'
        else:
            prefix = 'Flow/inception_i3d'

        hp.load_conv3d(state_dict, 'conv3d_1a_7x7', sess, f'{prefix}/Conv3d_1a_7x7')
        hp.load_conv3d(state_dict, 'conv3d_2b_1x1', sess, f'{prefix}/Conv3d_2b_1x1')
        hp.load_conv3d(state_dict, 'conv3d_2c_3x3', sess, f'{prefix}/Conv3d_2c_3x3')

        hp.load_mixed(state_dict, 'mixed_3b', sess, f'{prefix}/Mixed_3b')
        hp.load_mixed(state_dict, 'mixed_3c', sess, f'{prefix}/Mixed_3c')
        hp.load_mixed(state_dict, 'mixed_4b', sess, f'{prefix}/Mixed_4b')
        hp.load_mixed(state_dict, 'mixed_4c', sess, f'{prefix}/Mixed_4c')
        hp.load_mixed(state_dict, 'mixed_4d', sess, f'{prefix}/Mixed_4d')
        hp.load_mixed(state_dict, 'mixed_4e', sess, f'{prefix}/Mixed_4e')
        hp.load_mixed(state_dict, 'mixed_4f', sess, f'{prefix}/Mixed_4f')

        hp.load_mixed(state_dict, 'mixed_5b', sess, f'{prefix}/Mixed_5b', fix_typo=True)
        hp.load_mixed(state_dict, 'mixed_5c', sess, f'{prefix}/Mixed_5c')
        hp.load_conv3d(state_dict, 'conv3d_0c_1x1', sess, f'{prefix}/Logits/Conv3d_0c_1x1', bias=True, bn=False)

        self.load_state_dict(state_dict)
