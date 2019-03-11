from typing import Tuple
import numpy as np
import abc

__all__ = ['ArrayTransform', 'ArrayNormalize', 'ArrayStandardize', 'FramePad', 'VideoPad']


class ArrayTransform(abc.ABC):
    """Base video transform class."""

    @abc.abstractmethod
    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Video is always in format Mx(HxWxC)."""
        pass


class ArrayNormalize(ArrayTransform):
    """Normalize a np.ndarray video to [0.0, 1.0] range"""
    by: int

    def __init__(self, by: int):
        self.by = np.array(by, dtype=np.float32)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        return (video / self.by).astype(dtype=np.float32)


class ArrayStandardize(ArrayTransform):
    """Normalize a np.ndarray video with mean and standard deviation."""

    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        _, c1, _, c2 = video.shape

        if c1 == 3:
            return (video - self.mean.reshape((1, 3, 1, 1))) / self.std.reshape((1, 3, 1, 1))
        elif c2 == 3:
            return (video - self.mean.reshape((1, 1, 1, 3))) / self.std.reshape((1, 1, 1, 3))
        else:
            raise ValueError(f'Unsupported video format: {video.shape}')


class FramePad(ArrayTransform):
    """Pad every frame of a video to bring it to dimension std_height X std_width."""
    std_height: int
    std_width: int
    channel_first: bool

    def __init__(self, std_height: int, std_width: int, channel_first: bool = False):
        self.std_height = std_height
        self.std_width = std_width
        self.channel_first = channel_first

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Pad every frame of a video to bring it to dimension std_height X std_width."""
        if self.channel_first:
            l, c, h, w, = video.shape
        else:
            l, h, w, c = video.shape
        left, top, right, bottom = self.__padding_values(h, w)
        if self.channel_first:
            return np.pad(video, ((0, 0), (0, 0), (top, bottom), (left, right)), mode='constant')
        else:
            return np.pad(video, ((0, 0), (top, bottom), (left, right), (0, 0)), mode='constant')

    def __padding_values(self, height: int, width: int) -> Tuple[int, int, int, int]:
        """Get padding values based on current size and desired size."""
        top_pad = int((self.std_height - height) // 2)
        bottom_pad = int((self.std_height - height) // 2 + (self.std_height - height) % 2)

        left_pad = int((self.std_width - width) // 2)
        right_pad = int((self.std_width - width) // 2 + (self.std_width - width) % 2)

        return left_pad, top_pad, right_pad, bottom_pad


class VideoPad(ArrayTransform):
    """Pad the video with the last frame up to some length"""
    std_length: int

    def __init__(self, std_length: int):
        self.std_length = std_length

    def __call__(self, video: np.ndarray) -> np.ndarray:
        l, _, _, _ = video.shape
        diff = self.std_length - l
        return np.pad(video, ((0, diff), (0, 0), (0, 0), (0, 0)), mode='edge')
