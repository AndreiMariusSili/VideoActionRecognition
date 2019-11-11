from typing import List, Tuple, Union

import numpy as np
from PIL import Image

Size = Union[int, Tuple[int, int]]


def crop_video(video: List[Image.Image], min_h: int, min_w: int, h: int, w: int) -> List[Image.Image]:
    return [img.crop((min_w, min_h, min_w + w, min_h + h)) for img in video]


def resize_video(video: List[Image.Image], size: Size, interpolation='bilinear') -> List[Image.Image]:
    if isinstance(size, int):
        im_w, im_h = video[0].size
        if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
            return video
        new_h, new_w = get_resize_sizes(im_h, im_w, size)
        size = (new_w, new_h)
    else:
        size = size[1], size[0]

    if interpolation == 'bilinear':
        pil_inter = Image.BILINEAR
    else:
        pil_inter = Image.NEAREST
    return [img.resize(size, pil_inter) for img in video]


def get_resize_sizes(im_h: int, im_w: int, size: int):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)

    return oh, ow


def convert_img_to_channel_first(img):
    """Converts (H, W, C) numpy.ndarray to (C, W, H) format
    """
    if len(img.shape) == 3:
        img = img.transpose(2, 0, 1)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)
    return img
