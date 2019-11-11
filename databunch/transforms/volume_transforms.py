from typing import List

import numpy as np
import torch as th
from PIL import Image

from databunch.transforms.functional import convert_img_to_channel_first
from databunch.transforms.video_transforms import VideoTransform

__all__ = ['ToVolumeTensor', 'ToVolumeArray']


class ToVolumeTensor(VideoTransform):
    """Convert a list of M x (HxWxC) numpy.ndarrays in the range [0, 255]
    to a torch.FloatTensor of shape (C x m x H x W) in the range [0, 1.0]"""

    channels: int
    channel_first: bool

    def __init__(self, channels=3, channel_first: bool = True):
        self.channels = channels
        self.channel_first = channel_first

    def __call__(self, video: List[Image.Image]) -> th.Tensor:
        h, w = video[0].size

        if self.channel_first:
            np_video = np.zeros([len(video), self.channels, int(h), int(w)], dtype=np.float32)
        else:
            np_video = np.zeros([len(video), int(h), int(w), self.channels], dtype=np.float32)
        for img_idx, img in enumerate(video):
            # noinspection PyTypeChecker
            img = np.array(img, copy=False)
            if self.channel_first:
                img = convert_img_to_channel_first(img)
            np_video[img_idx, :, :, :] = img

        return th.from_numpy(np_video)


class ToVolumeArray(VideoTransform):
    """Convert a list of M x (HxWxC) numpy.ndarrays in the range [0, 255]
    to a numpy.ndarray of shape (C x m x H x W) in the range [0, 1.0]"""

    channels: int
    channel_first: bool

    def __init__(self, channels=3, channel_first: bool = True):
        self.channels = channels
        self.channel_first = channel_first

    def __call__(self, video: List[Image.Image]) -> np.ndarray:
        h, w = video[0].size

        if self.channel_first:
            np_video = np.zeros([len(video), self.channels, int(h), int(w)], dtype=np.float32)
        else:
            np_video = np.zeros([len(video), int(h), int(w), self.channels], dtype=np.float32)
        for img_idx, img in enumerate(video):
            # noinspection PyTypeChecker
            img = np.array(img, copy=False)
            if self.channel_first:
                img = convert_img_to_channel_first(img)
            np_video[img_idx, :, :, :] = img

        return np_video
