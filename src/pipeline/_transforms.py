
from typing import Tuple, List, Union, Dict, Any
import skimage.transform
import numpy as np
import cv2
import abc


class VideoTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class TransformComposition(VideoTransform):
    transforms: List[VideoTransform]

    def __init__(self, transforms: List[VideoTransform]):
        self.transforms = transforms

    def __call__(self, data: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            data = transform(data)

        return data


class Rescale(VideoTransform):
    """Rescaled an image by a factor of scale."""
    scale: float
    interpolation: str
    args: Dict[str, Any]

    def __init__(self, scale: float):
        assert 0.0 < scale <= 1.0, f'Scale should be in range (0.0, 1]. Received {scale}.'
        self.scale = scale
        self.args = dict(anti_aliasing=False, preserve_range=True, multichannel=True)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        l, h, w, c = video.shape
        nh, nw = round(h * self.scale), round(w * self.scale)

        scaled = np.empty((l, nh, nw, c), dtype=np.float)
        for i, frame in enumerate(video):
            scaled[i, :, :, :] = skimage.transform.rescale(frame, self.scale, **self.args)

        return scaled


class Resize(VideoTransform):
    size: Union[int, Tuple[int, int]]
    interpolation: str

    def __init__(self, size: Union[int, Tuple[int, int]], interpolation: str):
        self.size = size
        if interpolation == 'inter_area':
            self.interpolation = cv2.INTER_AREA
        elif interpolation == 'inter_linear':
            self.interpolation = cv2.INTER_LINEAR
        else:
            raise ValueError(f'Unknown interpolation: {interpolation}.')

    def __call__(self, video: np.ndarray) -> np.ndarray:
        l, h, w, c = video.shape
        nh, nw = self._get_dimensions(h, w)

        scaled = np.empty((l, nh, nw, c), dtype=np.float)
        for i, frame in enumerate(video):
            scaled[i, :, :, :] = cv2.resize(frame, (nw, nh), interpolation=self.interpolation)

        return scaled

    def _get_dimensions(self, h: int, w: int):
        if isinstance(self.size, int):
            if w < h:
                nw = self.size
                nh = int(self.size * h / w)
            else:
                nh = self.size
                nw = int(self.size * w / h)
        elif isinstance(self.size, tuple):
            nh, nw = self.size
        else:
            raise TypeError(f'Bad size type: {type(self.size)}')

        return nh, nw


class Normalize(VideoTransform):
    by: int

    def __init__(self, by: int):
        self.by = by

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Normalize a video from [0,255] to [0.0, 1.0]"""

        return video / self.by


class FramePad(VideoTransform):
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
            l, c, h, w,  = video.shape
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


class VideoPad(VideoTransform):
    std_length: int

    def __init__(self, std_length: int):
        self.std_length = std_length

    def __call__(self, video: np.ndarray) -> np.ndarray:
        l, _, _, _ = video.shape
        diff = self.std_length - l
        return np.pad(video, ((0, diff), (0, 0), (0, 0), (0, 0)), mode='edge')


class Standardize(VideoTransform):
    """Standardize every frame to 0 mean and 1 standard deviation."""
    means: np.ndarray
    stds: np.ndarray

    def __init__(self, means: Tuple[float, float, float], stds: Tuple[float, float, float]):
        self.means = np.array(means)
        self.stds = np.array(stds)

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Standardize every frame to 0 mean and unit standard deviation."""

        return (video - self.means.squeeze()) / self.stds.squeeze()


class ToVolumeArray(VideoTransform):
    """Transform a np.ndarray of size (L x H x W x C) to a torch.Tensor (L x C x H x W)"""

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Transform a np.ndarray of size (L x H x W x C) to a torch.Tensor (L x C x H x W)"""
        return video.transpose((0, 3, 1, 2))


class ToStackedArray(VideoTransform):
    """Transform an np.ndarray of size (L x H x W x C) to a torch.Tensor (LC x H x W)"""

    def __call__(self, video: np.ndarray) -> np.ndarray:
        """Transform a np.ndarray of size (L x H x W x C) to a torch.Tensor (L x C x H x W)"""
        l, h, w, c = video.shape
        return video.transpose((0, 3, 1, 2)).reshape((l * c, h, w))
