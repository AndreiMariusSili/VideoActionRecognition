# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch

from typing import Tuple, List, Dict, Any
import tensorflow as tf
import torch as th
import numpy as np
import math
import os

from env import logging


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


def get_conv_params(sess: tf.Session, name, bias=False) -> List[tf.Tensor]:
    # Get conv weights
    conv_weights_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'w:0'))
    conv_weights = sess.run(conv_weights_tensor)

    conv_shape = conv_weights.shape
    kernel_shape = conv_shape[0:3]
    in_channels = conv_shape[3]
    out_channels = conv_shape[4]

    conv_op = sess.graph.get_operation_by_name(os.path.join(name, 'convolution'))
    padding_name = conv_op.get_attr('padding')
    padding = _get_padding(padding_name, kernel_shape)
    all_strides = conv_op.get_attr('strides')
    strides = all_strides[1:4]

    conv_params = [conv_weights, kernel_shape, in_channels, out_channels, strides, padding]

    if bias:
        conv_bias_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'b:0'))
        conv_bias = sess.run(conv_bias_tensor)
        conv_params.append(conv_bias)

    return conv_params


def get_bn_params(sess: tf.Session, name: str) -> Tuple[tf.Tensor, ...]:
    moving_mean_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'moving_mean:0'))
    moving_var_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'moving_variance:0'))
    beta_tensor = sess.graph.get_tensor_by_name(os.path.join(name, 'beta:0'))
    moving_mean = sess.run(moving_mean_tensor)
    moving_var = sess.run(moving_var_tensor)
    beta = sess.run(beta_tensor)

    return moving_mean, moving_var, beta


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


def load_conv3d(state_dict: Dict[str, Any], name_pt: str, sess: tf.Session, name_tf: str,
                bias: bool = False, bn: bool = True) -> None:
    # Transfer convolution params
    conv_name_tf = os.path.join(name_tf, 'conv_3d')
    conv_params = get_conv_params(sess, conv_name_tf, bias=bias)
    if bias:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding, conv_bias = conv_params
        state_dict[name_pt + '.conv3d.bias'] = th.from_numpy(conv_bias)
    else:
        conv_weights, kernel_shape, in_channels, out_channels, strides, padding = conv_params

    # convert to pt format (out_c, in_c, depth, height, width)
    conv_weights_rs = np.transpose(conv_weights, (4, 3, 0, 1, 2))
    state_dict[name_pt + '.conv3d.weight'] = th.from_numpy(conv_weights_rs)

    # Transfer batch norm params
    if bn:
        conv_tf_name = os.path.join(name_tf, 'batch_norm')
        moving_mean, moving_var, beta = get_bn_params(sess, conv_tf_name)

        out_planes = conv_weights_rs.shape[0]
        state_dict[name_pt + '.batch3d.weight'] = th.ones(out_planes)
        state_dict[name_pt + '.batch3d.bias'] = th.from_numpy(beta.squeeze())
        state_dict[name_pt + '.batch3d.running_mean'] = th.from_numpy(moving_mean.squeeze())
        state_dict[name_pt + '.batch3d.running_var'] = th.from_numpy(moving_var.squeeze())


def load_mixed(state_dict: Dict[str, Any], name_pt: str, sess: tf.Session, name_tf: str, fix_typo=False) -> None:
    # Branch 0
    load_conv3d(state_dict, name_pt + '.branch_0', sess, os.path.join(name_tf, 'Branch_0/Conv3d_0a_1x1'))

    # Branch .1
    load_conv3d(state_dict, name_pt + '.branch_1.0', sess, os.path.join(name_tf, 'Branch_1/Conv3d_0a_1x1'))
    load_conv3d(state_dict, name_pt + '.branch_1.1', sess, os.path.join(name_tf, 'Branch_1/Conv3d_0b_3x3'))

    # Branch 2
    load_conv3d(state_dict, name_pt + '.branch_2.0', sess, os.path.join(name_tf, 'Branch_2/Conv3d_0a_1x1'))
    if fix_typo:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess, os.path.join(name_tf, 'Branch_2/Conv3d_0a_3x3'))
    else:
        load_conv3d(state_dict, name_pt + '.branch_2.1', sess, os.path.join(name_tf, 'Branch_2/Conv3d_0b_3x3'))

    # Branch 3
    load_conv3d(state_dict, name_pt + '.branch_3.1', sess, os.path.join(name_tf, 'Branch_3/Conv3d_0b_1x1'))


def compare_outputs(tf_out: np.ndarray, pt_out: np.ndarray) -> None:
    out_diff = np.abs(pt_out - tf_out)
    mean_diff = out_diff.mean()
    max_diff = out_diff.max()
    logging.info('===============')
    logging.info(f'max diff : {max_diff}, mean diff : {mean_diff}')
    logging.info(f'mean val: tf {tf_out.mean()} pt {pt_out.mean()}')
    logging.info(f'max vals: tf {tf_out.max()} pt {pt_out.max()}')
    logging.info(f'max relative diff: tf {(out_diff / np.abs(tf_out)).max()} pt {(out_diff / np.abs(pt_out)).max()}')
    logging.info('===============')
