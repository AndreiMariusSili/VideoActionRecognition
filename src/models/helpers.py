# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
from typing import Tuple

from torch import nn


def _pad_top_bottom(filter_dim: int, stride_val: int) -> Tuple[int, int]:
    pad_along = max(filter_dim - stride_val, 0)
    pad_top = pad_along // 2
    pad_bottom = pad_along - pad_top

    return pad_top, pad_bottom


def get_padding_shape(filter_shape: Tuple[int, int, int], stride: Tuple[int, int, int]) -> Tuple[int, ...]:
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


def simplify_padding(padding_shapes: Tuple[int, ...]) -> Tuple[bool, int]:
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False

    return all_same, padding_init


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
