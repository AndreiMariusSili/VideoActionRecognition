import abc
import random
from typing import Callable, List, Tuple, Union

import numpy as np
import torch as th
import torchvision
from PIL import Image

from databunch.transforms import functional as func

__all__ = [
    'VideoTransform', 'VideoCompose',
    'RandomHorizontalFlip', 'RandomResize', 'Resize', 'RandomCrop', 'CenterCrop',
    'RandomRotation', 'ColorJitter',
    'Pad',
]


class VideoTransform(abc.ABC):
    """Base video transform class."""

    @abc.abstractmethod
    def __call__(self, video: List[Image.Image]) -> Union[List[Image.Image], th.Tensor, np.ndarray]:
        """Video is always in format Mx(HxWxC)."""
        pass


class VideoCompose(VideoTransform):
    """Compose multiple video transforms sequentially."""
    transforms: List[VideoTransform]

    def __init__(self, transforms: List[VideoTransform]):
        self.transforms = transforms

    def __call__(self, video: List[Image.Image]) -> Union[List[Image.Image], np.ndarray, th.Tensor]:
        for transform in self.transforms:
            video = transform(video)

        return video


class RandomHorizontalFlip(VideoTransform):
    """Horizontally flip the list of given images randomly with a probability 0.5"""

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in video]

        return video


class RandomResize(VideoTransform):
    """Resize a video to a final random size, maintaining aspect ratio.
    The larger the original image is, the more times it takes to interpolate.
    Interpolation can be one of: 'nearest', 'bilinear'."""

    def __init__(self, ratio=(3. / 4., 4. / 3.), interpolation='nearest'):
        self.ratio = ratio
        self.interpolation = interpolation

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        scaling_factor = random.uniform(self.ratio[0], self.ratio[1])
        im_w, im_h = video[0].size

        new_w = int(im_w * scaling_factor)
        new_h = int(im_h * scaling_factor)
        new_size = (new_w, new_h)

        return func.resize_video(video, new_size, interpolation=self.interpolation)


class Resize(VideoTransform):
    """Resize a list of to a final specified size.
    The larger the original image is, the more times it takes to interpolate.
    Interpolation can be one of: 'nearest', 'bilinear'."""

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        return func.resize_video(video, self.size, interpolation=self.interpolation)


class RandomCrop(VideoTransform):
    """Extract random crop at the same location for a list of images."""

    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)

        self.size = size

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        h, w = self.size
        im_w, im_h = video[0].size
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h, w=w, h=h)
            )
            raise ValueError(error_msg)

        x1 = random.randint(0, im_w - w)
        y1 = random.randint(0, im_h - h)

        return func.crop_video(video, y1, x1, h, w)


class CenterCrop(VideoTransform):
    """Extract center crop at the same location for a list of images."""
    size: Tuple[int, int]

    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)

        self.size = size

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        h, w = self.size
        im_w, im_h = video[0].size
        if w > im_w or h > im_h:
            error_msg = (
                'Initial image size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial image is ({im_w}, {im_h})'.format(im_w=im_w, im_h=im_h, w=w, h=h))
            raise ValueError(error_msg)

        x1 = int(round((im_w - w) / 2.))
        y1 = int(round((im_h - h) / 2.))

        return func.crop_video(video, y1, x1, h, w)


class RandomRotation(VideoTransform):
    """Rotate entire clip randomly by a random angle within given bounds, specified by 'degrees'.
    If degrees is a number instead of sequence like (min, max), the range of degrees, will be (-degrees, +degrees). """

    def __init__(self, degrees: Union[int, float, Tuple[int, int]]):
        if isinstance(degrees, float) or isinstance(degrees, int):
            if degrees < 0:
                raise ValueError('If degrees is a single number, must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence, it must be of len 2.')

        self.degrees = degrees

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        angle = random.uniform(self.degrees[0], self.degrees[1])

        return [img.rotate(angle) for img in video]


class ColorJitter(VideoTransform):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    * brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    * contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    * saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    * hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        b, c, s, h = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        transforms = self.__create_color_jitter_transform(b, c, s, h)

        new_video = []
        for frame in video:
            for transform in transforms:
                frame = transform(frame)
            new_video.append(frame)

        return new_video

    def __create_color_jitter_transform(self, brightness, contrast, saturation, hue) -> List[Callable]:
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        return img_transforms


class Pad(VideoTransform):

    def __init__(self, std_height: int, std_width: int):
        self.std_height = std_height
        self.std_width = std_width

    def __call__(self, video: List[Image.Image]) -> List[Image.Image]:
        width, height = video[0].size

        width_pad = max(self.std_width - width, 0)
        if width_pad % 2 == 0:
            left_pad = right_pad = width_pad // 2
        else:
            left_pad = width_pad // 2 + 1
            right_pad = width_pad // 2

        height_pad = max(self.std_height - height, 0)
        if height_pad % 2 == 0:
            top_pad = bottom_pad = height_pad // 2
        else:
            top_pad = height_pad // 2 + 1
            bottom_pad = height_pad // 2

        pad = torchvision.transforms.Pad((left_pad, top_pad, right_pad, bottom_pad))

        return [pad(frame) for frame in video]
