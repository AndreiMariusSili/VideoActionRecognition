from PIL import Image
import numpy as np
import numbers
import cv2


def crop_clip(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
    return cropped


def resize_clip(clip, size, interpolation='bilinear'):
    if isinstance(clip[0], np.ndarray):
        scaled = _resize_clip_ndarray(clip, size, interpolation)
    elif isinstance(clip[0], Image.Image):
        scaled = _resize_clip_pil(clip, size, interpolation)
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image but got list of {0}'.format(type(clip[0])))
    return scaled


def _resize_clip_ndarray(clip, size, interpolation):
    if isinstance(size, numbers.Number):
        im_h, im_w, im_c = clip[0].shape
        # Min spatial dim already matches minimal size
        if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
            return clip
        new_h, new_w = get_resize_sizes(im_h, im_w, size)
        size = (new_w, new_h)
    else:
        size = size[1], size[0]
    if interpolation == 'bilinear':
        np_inter = cv2.INTER_LINEAR
    else:
        np_inter = cv2.INTER_NEAREST
    scaled = [cv2.resize(img, size, interpolation=np_inter) for img in clip]

    return scaled


def _resize_clip_pil(clip, size, interpolation):
    if isinstance(size, numbers.Number):
        im_w, im_h = clip[0].size
        # Min spatial dim already matches minimal size
        if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
            return clip
        new_h, new_w = get_resize_sizes(im_h, im_w, size)
        size = (new_w, new_h)
    else:
        size = size[1], size[0]
    if interpolation == 'bilinear':
        pil_inter = Image.NEAREST
    else:
        pil_inter = Image.BILINEAR
    return [img.resize(size, pil_inter) for img in clip]


def get_resize_sizes(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
    else:
        oh = size
        ow = int(size * im_w / im_h)
    return oh, ow
