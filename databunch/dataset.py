import typing as tp

import numpy as np
import pandas as pd
import torch as th
from torch.utils import data as thd

import constants as ct
import databunch.label as pil
import databunch.transforms as dt
import databunch.video as piv
import databunch.video_meta as pim
import helpers as hp
import options.data_options as do


def _pad_videos(videos: tp.List[np.ndarray], ml: int) -> tp.Tuple[np.ndarray, ...]:
    """Pad a tuple of videos to the same length."""
    pad = dt.VideoPad(ml)
    return tuple(pad(video) for video in videos)


def _max_dimensions(videos: tp.List[np.ndarray]) -> tp.Tuple[int, int, int]:
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


def collate(batch: [(piv.Video, pil.Label)]) -> (th.Tensor, th.Tensor, piv.Video, pil.Label):
    """Transform list of videos and labels to fixed size tensors. Return original list and collated batch."""
    videos, labels = zip(*batch)

    videos_data = [video.data for video in videos]
    labels_data = [label.data for label in labels]

    ml, mh, mw = _max_dimensions(videos_data)
    videos_data = _pad_videos(videos_data, ml)

    videos_data = thd.dataloader.default_collate(videos_data)  # noqa
    labels_data = thd.dataloader.default_collate(labels_data)  # noqa

    return videos_data, labels_data, videos, labels


def init_worker(_id: int):
    np.random.seed(_id)


class VideoDataset(thd.Dataset):
    def __init__(self, cut: float, frame_size: int, data_opts: do.DataSetOptions, sampling_opts: do.SamplingOptions):
        """Initialize a video dataset from the DataFrame containing meta information."""
        assert 0.0 <= cut <= 1.0, f'Cut should be between 0.0, and 1.0. Received: {cut}.'
        assert data_opts.setting in ['train', 'eval'], f'Unknown setting: {data_opts.setting}.'

        self.cut = cut
        self.frame_size = frame_size
        self.do = data_opts
        self.so = sampling_opts

        self.meta = hp.read_meta(data_opts.meta_path)
        self.lids = self.meta['lid'].unique()
        self.lid2labels = self.meta.groupby('lid')['label'].head(1)
        self.labels2lid = self.lid2labels.reset_index().set_index('label')

        if data_opts.keep is not None:
            if 0 <= data_opts.keep < 1:
                self._stratified_sample_meta(data_opts.keep)
            else:
                self.meta = self.meta.iloc[0:data_opts.keep]
        self.transform = self._compose_transforms()

    def __getitem__(self, item: int) -> tp.Tuple[piv.Video, pil.Label]:
        video_meta = pim.VideoMeta(**self.meta.iloc[item][pim.VideoMeta.fields].to_dict())

        video = piv.Video(video_meta, ct.WORK_ROOT / self.do.root_path, self.do.read_jpeg,
                          self.cut, self.do.setting, self.so.num_segments, self.so.segment_size)
        label = pil.Label(video_meta)

        video.data = self.transform(video.data)

        return video, label

    def get_batch(self, n: int) -> tp.Tuple[tp.List[piv.Video], tp.List[pil.Label]]:
        videos, labels = [], []
        for i, row in self.meta.sample(n=n).iterrows():
            iloc = self.meta.index.get_loc(i)
            video, label = self[iloc]
            videos.append(video)
            labels.append(label)

        return videos, labels

    def __len__(self):
        return len(self.meta)

    def __str__(self):
        string = f"""Something-Something Dataset: {len(self)} x {self[0]}"""

        return string

    def _compose_transforms(self) -> dt.VideoCompose:
        """Create transformation pipeline depending on setting."""
        if self.do.setting == 'train':
            return dt.VideoCompose([
                dt.Pad(self.frame_size, self.frame_size),
                dt.RandomCrop(self.frame_size),
                dt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                dt.ToVolumeArray(3, True),
                dt.ArrayNormalize(255)
            ])
        else:
            return dt.VideoCompose([
                dt.Pad(self.frame_size, self.frame_size),
                dt.CenterCrop(self.frame_size),
                dt.ToVolumeArray(3, True),
                dt.ArrayNormalize(255)
            ])

    def _stratified_sample_meta(self, keep: float):
        """Selects the first instances of a class up to a proportion keep."""
        samples = []
        for lid in self.meta['lid'].unique():
            class_meta = self.meta[self.meta['lid'] == lid]
            cut_off = round(len(class_meta) * keep)
            class_meta_sample = class_meta.iloc[0:cut_off]
            samples.append(class_meta_sample)

        self.meta = pd.concat(samples, verify_integrity=True)
