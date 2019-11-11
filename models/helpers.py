# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import typing as tp

import torch as th
from torch import nn


def _pad_top_bottom(filter_dim: int, stride_val: int) -> tp.Tuple[int, int]:
    pad_along = max(filter_dim - stride_val, 0)
    pad_top = pad_along // 2
    pad_bottom = pad_along - pad_top

    return pad_top, pad_bottom


def get_padding_shape(filter_shape: tp.Tuple[int, int, int], stride: tp.Tuple[int, int, int]) -> tp.Tuple[
    int, ...]:
    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes: tp.Tuple[int, ...]) -> tp.Tuple[bool, int]:
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False

    return all_same, padding_init


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def he_init(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            th.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
