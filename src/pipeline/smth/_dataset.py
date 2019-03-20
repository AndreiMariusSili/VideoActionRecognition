from torch.utils import data as thd
from typing import List, Tuple, Union
import numpy as np
import torch as th
import os

import pipeline as pipe
import constants as ct
import helpers as hp

LOADED_ITEM = Tuple[pipe.Video, pipe.Label]
COLLATED_ITEMS = Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor, th.Tensor, pipe.Video, pipe.Label]]


class SmthDataset(thd.Dataset):
    presenting: bool
    evaluating: bool
    do: pipe.DataOptions
    so: pipe.SamplingOptions
    transform: pipe.VideoCompose

    def __init__(self, data_opts: pipe.DataOptions, sampling_opts: pipe.SamplingOptions):
        """Initialize a smth-smth dataset from the DataFrame containing meta information."""
        assert 0.0 <= data_opts.cut <= 1.0, f'Cut should be between 0.0, and 1.0. Received: {data_opts.cut}.'
        assert data_opts.setting in ['train', 'valid'], f'Unknown setting: {data_opts.setting}.'

        self.presenting = False
        self.evaluating = False

        self.do = data_opts
        self.so = sampling_opts

        self.meta = hp.read_smth_meta(data_opts.meta_path)
        self.labels2id = hp.read_smth_labels2id(ct.SMTH_LABELS2ID)

        if data_opts.keep is not None:
            self.meta = self.meta.sample(n=data_opts.keep)
        self.transform = data_opts.transform

    def __getitem__(self, item: int) -> LOADED_ITEM:
        video_meta = pipe.VideoMeta(**self.meta.iloc[item][pipe.VideoMeta.fields].to_dict())

        if self.presenting:
            video = pipe.Video(video_meta, self.do.cut, self.do.setting, None, None)
            label = pipe.Label(video_meta)
            video.data = pipe.CenterCrop(224)(video.data)
        else:
            video = pipe.Video(video_meta, self.do.cut, self.do.setting, self.so.num_segments, self.so.segment_size)
            label = pipe.Label(video_meta)
            video.data = self.transform(video.data)

        return video, label

    def get_batch(self, n: int) -> Tuple[List[pipe.Video], List[pipe.Label]]:
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
        return tuple(pipe.VideoPad(ml)(video) for video in videos)

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


if __name__ == '__main__':
    os.chdir('/Users/Play/Code/AI/master-thesis/src/')
    base_transform = pipe.VideoCompose([
        pipe.RandomCrop(224),
        pipe.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        pipe.ToVolumeArray(),
        pipe.ArrayStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS),
    ])
    _data_opts = pipe.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=1.0,
        setting='train',
        transform=base_transform,
    )
    _sampling_opts = pipe.SamplingOptions(
        num_segments=4,
        segment_size=2
    )
    dataset = SmthDataset(_data_opts, _sampling_opts)

    x, y = dataset[0]
    b = dataset.get_batch(10)
