from typing import List
from PIL import Image
import numpy as np
import torch as th

from pipeline.transforms.video_transforms import VideoTransform
from pipeline.transforms.functional import convert_img_to_channel_first

__all__ = ['ClipToStackedTensor', 'ClipToStackedArray']


class ClipToStackedTensor(VideoTransform):
    """Converts a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    or PIL Images to a torch.FloatTensor of shape (m*C x H x W)
    in the range [0, 1.0]
    """

    channels: int

    def __init__(self, channels=3):
        self.channels = channels

    def __call__(self, video: List[Image.Image]) -> th.Tensor:
        w, h = video[0].size

        np_video = np.zeros([self.channels * len(video), int(h), int(w)], dtype=np.float32)
        for img_idx, img in enumerate(video):
            # noinspection PyTypeChecker
            img = np.array(img, copy=False)
            img = convert_img_to_channel_first(img)
            np_video[img_idx * self.channels:(img_idx + 1) * self.channels, :, :] = img

        return th.from_numpy(np_video)


class ClipToStackedArray(VideoTransform):
    """Converts a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    or PIL Images to a numpy.ndarray of shape (m*C x H x W) in the range [0, 1.0]
    """

    channels: int

    def __init__(self, channels=3):
        self.channels = channels

    def __call__(self, video: List[Image.Image]) -> np.ndarray:
        w, h = video[0].size

        np_video = np.zeros([self.channels * len(video), int(h), int(w)], dtype=np.float32)
        for img_idx, img in enumerate(video):
            # noinspection PyTypeChecker
            img = np.array(img, copy=False)
            img = convert_img_to_channel_first(img)
            np_video[img_idx * self.channels:(img_idx + 1) * self.channels, :, :] = img

        return np_video
