import abc
import random
from typing import Tuple

import torch as th

__all__ = ['TensorTransform', 'TensorNormalize', 'TensorStandardize', 'SpatialRandomCrop']


class TensorTransform(abc.ABC):
    """Base video transform class."""

    @abc.abstractmethod
    def __call__(self, video: th.Tensor) -> th.Tensor:
        """Video is always in format Mx(HxWxC)."""
        pass


class TensorNormalize(TensorTransform):
    """Normalize a np.ndarray video to [0.0, 1.0] range"""
    by: th.Tensor

    def __init__(self, by: int):
        self.by = th.tensor(by, dtype=th.float)

    def __call__(self, video: th.Tensor) -> th.Tensor:
        device = video.device
        return video / self.by.to(device)


class TensorStandardize(TensorTransform):
    """Normalize a np.ndarray video with mean and standard deviation."""

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.mean = th.tensor(mean, dtype=th.float)
        self.std = th.tensor(std, dtype=th.float)

    def __call__(self, video: th.Tensor) -> th.Tensor:
        device = video.device
        if len(video.shape) == 4:
            return video.sub_(self.mean.to(device)).div_(self.std.to(device))
        elif len(video.shape) == 5:
            _, _, c1, _, c2 = video.shape
            if c1 == 3:
                return (video - self.mean.reshape((1, 3, 1, 1)).to(device)) / self.std.reshape((1, 3, 1, 1)).to(device)
            elif c2 == 3:
                return (video - self.mean.reshape((1, 1, 1, 3)).to(device)) / self.std.reshape((1, 1, 1, 3)).to(device)
            else:
                raise ValueError(f'Unsupported video format: {video.shape}')
        else:
            raise ValueError(f'Unsupported video format: {video.shape}')


class SpatialRandomCrop(TensorTransform):
    """Crops a random spatial crop in a spatio-temporal
    tensor input [Channel, Time, Height, Width] at a given size."""

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, tensor: th.Tensor) -> th.Tensor:
        h, w = self.size
        _, _, tensor_h, tensor_w = tensor.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)
        x1 = random.randint(0, tensor_w - w)
        y1 = random.randint(0, tensor_h - h)

        return tensor[:, :, y1:y1 + h, x1:x1 + h]
