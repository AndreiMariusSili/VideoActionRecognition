from torch import nn


def t_conv3x3(in_planes: int, out_planes: int, stride: int = 1, padding: int = 1, groups: int = 1,
              dilation: int = 1, output_padding: int = 1) -> nn.ConvTranspose2d:
    """3x3 transpose convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups,
                              dilation=dilation, output_padding=output_padding, bias=False)


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
