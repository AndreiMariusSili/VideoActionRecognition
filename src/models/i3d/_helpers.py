# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch

import math
from typing import List, Tuple


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


def _get_padding(padding_name: bytes, conv_shape: Tuple[int, int, int]) -> List[int]:
    padding_name = padding_name.decode("utf-8")
    if padding_name not in ['SAME', 'VALID']:
        raise ValueError(f'Unknown padding {padding_name}.')

    if padding_name == 'SAME':
        return [
            math.floor(int(conv_shape[0]) / 2),
            math.floor(int(conv_shape[1]) / 2),
            math.floor(int(conv_shape[2]) / 2)
        ]
    else:
        return [0, 0]

