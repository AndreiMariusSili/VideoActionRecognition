import abc
from typing import Tuple
import torch as th
import random

__all__ = ['TensorTransform', 'TensorNormalize', 'TensorStandardize', 'SpatialRandomCrop']


class TensorTransform(abc.ABC):
    """Base video transform class."""

    @abc.abstractmethod
    def __call__(self, video: th.Tensor) -> th.Tensor:
        """Video is always in format Mx(HxWxC)."""
        pass


class TensorNormalize(TensorTransform):
    """Normalize a torch.Tensor video to [0.0, 1.0] range"""
    by: int

    def __init__(self, by: int):
        self.by = th.tensor(by, dtype=th.float)

    def __call__(self, video: th.Tensor) -> th.Tensor:
        return video.div_(self.by)


class TensorStandardize(TensorTransform):
    """Normalize a torch.Tensor video with mean and standard deviation."""

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.mean = th.tensor(mean)
        self.std = th.tensor(std)

    def __call__(self, tensor: th.Tensor) -> th.Tensor:
        return tensor.sub_(self.mean).div_(self.std)


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
