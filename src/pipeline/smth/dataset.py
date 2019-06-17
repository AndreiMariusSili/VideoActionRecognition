import os
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch as th
from torch.utils import data as thd

import constants as ct
import helpers as hp
import options.pipe_options as pio
import pipeline.label as pil
import pipeline.transforms as pit
import pipeline.video as piv
import pipeline.video_meta as pim

LOADED_ITEM = Tuple[piv.Video, pil.Label]
COLLATED_ITEMS = Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor, piv.Video, pil.Label]]


class SmthDataset(thd.Dataset):
    presenting: bool
    evaluating: bool
    do: pio.DataOptions
    so: pio.SamplingOptions
    transform: pit.VideoCompose

    def __init__(self, data_opts: pio.DataOptions, sampling_opts: pio.SamplingOptions):
        """Initialize a smth-smth dataset from the DataFrame containing meta information."""
        assert 0.0 <= data_opts.cut <= 1.0, f'Cut should be between 0.0, and 1.0. Received: {data_opts.cut}.'
        assert data_opts.setting in ['train', 'valid'], f'Unknown setting: {data_opts.setting}.'

        self.presenting = False
        self.evaluating = False

        self.do = data_opts
        self.so = sampling_opts

        self.meta = hp.read_smth_meta(data_opts.meta_path)

        if data_opts.keep is not None:
            if 0 <= data_opts.keep < 1:
                self._stratified_sample_meta(data_opts.keep)
            else:
                self.meta = self.meta.iloc[0:data_opts.keep]
        self.transform = data_opts.transform

    def _stratified_sample_meta(self, keep: float):
        """Selects the first instances of a class up to a proportion keep."""
        samples = []
        for template_id in self.meta.template_id.unique():
            class_meta = self.meta[self.meta.template_id == template_id]
            cut_off = round(len(class_meta) * keep)
            class_meta_sample = class_meta.iloc[0:cut_off]
            samples.append(class_meta_sample)

        self.meta = pd.concat(samples, verify_integrity=True)

    def __getitem__(self, item: int) -> LOADED_ITEM:
        video_meta = pim.VideoMeta(**self.meta.iloc[item][pim.VideoMeta.fields].to_dict())

        if self.presenting:
            video = piv.Video(video_meta, self.do.cut, self.do.setting, None, None)
            label = pil.Label(video_meta)
            video.data = pit.CenterCrop(224)(video.data)
        else:
            video = piv.Video(video_meta, self.do.cut, self.do.setting, self.so.num_segments, self.so.segment_size)
            label = pil.Label(video_meta)
            video.data = self.transform(video.data)

        return video, label

    def get_batch(self, n: int) -> Tuple[List[piv.Video], List[pil.Label]]:
        self.presenting = True

        videos, labels, metas, items = [], [], [], []
        for i, row in self.meta.sample(n=n).iterrows():
            iloc = self.meta.index.get_loc(i)
            video, label = self[iloc]
            videos.append(video)
            labels.append(label)
        self.presenting = False

        return videos, labels

    def __len__(self):
        return len(self.meta)

    def __str__(self):
        self.presenting = True
        string = f"""Something-Something Dataset: {len(self)} x {self[0]}"""
        self.presenting = False

        return string

    def collate(self, batch: List[Tuple[np.ndarray, int]]) -> COLLATED_ITEMS:
        """Transform list of videos and labels to fixed size tensors. If in evaluation or presentation mode, will
        also return the video and label objects."""
        videos, labels = zip(*batch)

        videos_data = [video.data for video in videos]
        labels_data = [label.data for label in labels]

        ml, mh, mw = self._max_dimensions(videos_data)
        videos_data = self._pad_videos(videos_data, ml)

        videos_data = thd.dataloader.default_collate(videos_data)
        labels_data = thd.dataloader.default_collate(labels_data)

        if self.evaluating or self.presenting:
            batch = (videos_data, labels_data, videos, labels)
        else:
            batch = (videos_data, labels_data)

        return batch

    def _pad_videos(self, videos: List[np.ndarray], ml: int) -> Tuple[np.ndarray]:
        """Pad a tuple of videos to the same length."""
        return tuple(pit.VideoPad(ml)(video) for video in videos)

    def _max_dimensions(self, videos: List[np.ndarray]) -> Tuple[int, int, int]:
        """Get the maximum length, height, and width of a batch of videos."""
        ml = mh = mw = 0
        for video in videos:
            l, c, h, w = video.shape
            if l > ml:
                ml = l
            if h > mh:
                mh = h
            if w > mw:
                mw = w

        return ml, mh, mw


def setup():
    os.chdir('/Users/Play/Code/AI/master-thesis/src/')
    base_transform = pit.VideoCompose([
        pit.RandomCrop(224),
        pit.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        pit.ToVolumeArray(),
        pit.ArrayStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS),
    ])
    _data_opts = pio.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=0.5,
        setting='valid',
        transform=base_transform,
        keep=0.1
    )
    _sampling_opts = pio.SamplingOptions(
        num_segments=4,
        segment_size=1
    )
    dataset = SmthDataset(_data_opts, _sampling_opts)

    return dataset


if __name__ == '__main__':
    import timeit

    print(timeit.timeit("dataset[0]", setup="from __main__ import setup; dataset=setup()", number=10))

    b = setup().get_batch(10)
