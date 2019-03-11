from torch.utils import data as thd
from typing import List, Tuple, Union
from PIL import Image
import pathlib as pl
import pandas as pd
import numpy as np
import os

import pipeline as pipe
import constants as ct
import helpers as hp

LOADED_ITEM = Union[Tuple[pipe.Video, pipe.Label], Tuple[List[Image.Image], int]]


class SmthDataset(thd.Dataset):
    presenting: bool
    cut: float
    sample_size: int
    keep: int
    meta: pd.DataFrame
    transform: pipe.VideoCompose

    def __init__(self, meta: pl.Path, cut: float, sample_size: int,
                 transform: pipe.VideoCompose = None, split: str = None, keep: int = None):
        """Initialize a smth-smth dataset from the DataFrame containing meta information."""
        assert 0.0 <= cut <= 1.0, f'Cut should be between 0.0, and 1.0. Received: {cut}.'
        assert split in ['train', 'valid', None], f'Split can be one of: train, valid. Given: {split}.'
        self.presenting = False
        self.cut = cut
        self.sample_size = sample_size
        self.keep = keep

        self.meta = hp.read_smth_meta(meta)
        self.labels2id = hp.read_smth_labels2id(ct.SMTH_LABELS2ID)

        if split is not None and 'split' in self.meta.columns:
            self.meta = self.meta[self.meta['split'] == split]
        if sample_size is not None:
            self.meta = self.meta[self.meta['length'] >= sample_size]
        if keep is not None:
            self.meta = self.meta.sample(n=keep)
        self.transform = transform

    def __getitem__(self, item: int) -> LOADED_ITEM:
        video_meta = pipe.VideoMeta(**self.meta.iloc[item][pipe.VideoMeta.fields].to_dict())
        if self.presenting:
            video = pipe.Video(video_meta, self.cut)
            label = pipe.Label(video_meta)
            video.data = pipe.CenterCrop(224)(video.data)  # Hacky magic number. Should change.

            return video, label
        else:
            video = pipe.Video(video_meta, self.cut, self.sample_size)
            label = pipe.Label(video_meta)
            if self.transform is not None:
                video.data = self.transform(video.data)

            return video.data, label.data

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

    def collate(self, batch: List[Tuple[np.ndarray, int]]):
        if self.presenting:
            raise AttributeError('In presentation mode. Do not use DataLoader.')

        videos, labels = zip(*batch)
        ml, mh, mw = self._max_dimensions(videos)
        videos = self._pad_videos(videos, ml)

        return thd.dataloader.default_collate(videos), thd.dataloader.default_collate(labels)

    def _pad_videos(self, videos: Tuple[np.ndarray], ml: int) -> Tuple[np.ndarray]:
        """Pad a tuple of videos to the same length."""
        return tuple(pipe.VideoPad(ml)(video) for video in videos)

    def _max_dimensions(self, videos: Tuple[np.ndarray]) -> Tuple[int, int, int]:
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
    dataset = SmthDataset(ct.SMTH_META_TRAIN, 1.0, 16, base_transform)
    x, y = dataset[0]
    b = dataset.get_batch(10)
