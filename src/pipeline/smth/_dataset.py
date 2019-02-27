from torch.utils import data as thd
from typing import List, Tuple
import pathlib as pl
import pandas as pd
import numpy as np
import os

import pipeline as pipe
import constants as ct
import helpers as hp


class SmthDataset(thd.Dataset):
    presenting: bool
    cut: float
    meta: pd.DataFrame
    transform: pipe.TransformComposition

    def __init__(self, meta: pl.Path, cut: float, transform: pipe.TransformComposition = None,
                 split: str = None, keep: int = None):
        """Initialize a smth-smth dataset from the DataFrame containing meta information."""
        assert 0.0 <= cut <= 1.0, f'Cut should be between 0.0, and 1.0. Received: {cut}.'
        assert split in ['train', 'valid', None], f'Split can be one of: train, valid. Given: {split}.'
        self.presenting = False
        self.cut = cut
        self.meta = hp.read_smth_meta(meta)
        self.labels2id = hp.read_smth_labels2id(ct.SMTH_LABELS2ID)
        if split is not None and 'split' in self.meta.columns:
            self.meta = self.meta[self.meta['split'] == split]
        self.transform = transform

        if keep is not None:
            self.meta = self.meta.sample(n=keep)
        transforms = map(lambda t: type(t).__name__, self.transform.transforms) if self.transform is not None else []
        transforms = list(transforms)
        if 'ToVolumeArray' in transforms or 'ToStackedArray' in transforms:
            self.channel_first = True
        else:
            self.channel_first = False

    def __getitem__(self, item: int):
        video_meta = pipe.VideoMeta(**self.meta.iloc[item][pipe.VideoMeta.fields].to_dict())

        video = pipe.Video(video_meta, self.cut)
        label = pipe.Label(video_meta)
        if self.transform is not None:
            video.data = self.transform(video.data)

        if self.presenting:
            self.presenting = False
            return video, label
        return video.data.astype(np.float32), label.data

    def __len__(self):
        return len(self.meta)

    def __str__(self):
        self.presenting = True
        return f""" Something-Something Dataset: {len(self)} x {self[0]}"""

    def collate(self, batch: List[Tuple[np.ndarray, pipe.Label]]):
        videos, labels = zip(*batch)
        ml, mh, mw = self._max_dimensions(videos)
        videos = self._pad_videos(videos, ml)

        return thd.dataloader.default_collate(videos), thd.dataloader.default_collate(labels)

    def _pad_videos(self, videos: Tuple[np.ndarray], ml) -> Tuple[np.ndarray]:
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
    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    base_transform = pipe.TransformComposition([
        pipe.VideoRandomCrop(224),
        pipe.VideoColorJitter(brightness=2, contrast=2, saturation=2, hue=0.5),
        pipe.VideoNormalize(255),
        pipe.VideoStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS),
        pipe.ToVolumeArray()
    ])
    dataset = SmthDataset(ct.SMTH_META_TRAIN, 0.3, base_transform)
    for i in range(10):
        x, y = dataset[i]
