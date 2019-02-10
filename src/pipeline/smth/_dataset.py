from torch.utils.data import dataloader
from typing import List, Tuple
import torch.utils.data
import pathlib as pl
import pandas as pd
import numpy as np

from pipeline._transforms import TransformComposition, VideoPad, FramePad
from pipeline._video_meta import VideoMeta
from pipeline._label import Label
from pipeline._video import Video
import constants as ct
import helpers as hp


class SmthDataset(torch.utils.data.dataset.Dataset):
    presenting: bool
    cut: float
    meta: pd.DataFrame
    transform: TransformComposition

    def __init__(self, meta: pl.Path, cut: float, transform: TransformComposition = None, split: str = None):
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
        transforms = map(lambda t: type(t).__name__, self.transform.transforms) if self.transform is not None else []
        transforms = list(transforms)
        if 'ToVolumeArray' in transforms or 'ToStackedArray' in transforms:
            self.channel_first = True
        else:
            self.channel_first = False

    def __getitem__(self, item: int):
        video_meta = VideoMeta(**self.meta.iloc[item][VideoMeta.fields].to_dict())

        video = Video(video_meta, self.cut)
        label = Label(video_meta)
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

    def collate(self, batch: List[Tuple[np.ndarray, Label]]):
        videos, labels = zip(*batch)
        ml, mh, mw = self._max_dimensions(videos)
        videos = self._pad_videos(videos, ml)

        return dataloader.default_collate(videos), dataloader.default_collate(labels)

    def _pad_frames(self, videos: Tuple[np.ndarray], mh: int, mw: int) -> Tuple[np.ndarray]:
        """Pad a tuple of videos to the same width and height."""
        return tuple(FramePad(mh, mw, self.channel_first)(video) for video in videos)

    def _pad_videos(self, videos: Tuple[np.ndarray], ml) -> Tuple[np.ndarray]:
        """Pad a tuple of videos to the same length."""
        return tuple(VideoPad(ml)(video) for video in videos)

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
