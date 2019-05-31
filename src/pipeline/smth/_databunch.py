import os
from typing import List, Optional, Tuple

import pandas as pd
import torch as th
from dataclasses import asdict
from torch.utils import data as thd
from tqdm import tqdm

import constants as ct
import helpers as hp
import pipeline as pipe


class SmthDataBunch(object):
    db_opts: 'pipe.DataBunchOptions'
    train_ds_opts: 'pipe.DataSetOptions'
    valid_ds_opts: 'pipe.DataSetOptions'
    train_dl_opts: 'pipe.DataLoaderOptions'
    valid_dl_opts: 'pipe.DataLoaderOptions'
    stats: 'pd.DataFrame'

    train_set: 'pipe.SmthDataset'
    train_sampler: Optional['thd.DistributedSampler']
    train_loader: 'thd.DataLoader'
    valid_set: 'pipe.SmthDataset'
    valid_sampler: Optional['thd.DistributedSampler']
    valid_loader: 'thd.DataLoader'

    def __init__(self, db_opts: pipe.DataBunchOptions,
                 train_ds_opts: pipe.DataSetOptions, valid_ds_opts: pipe.DataSetOptions,
                 train_dl_opts: pipe.DataLoaderOptions, valid_dl_opts: pipe.DataLoaderOptions):
        assert db_opts.shape in ['stack', 'volume'], f'Unknown shape {db_opts.shape}.'
        self.db_opts = db_opts
        self.train_ds_opts = train_ds_opts
        self.train_dl_opts = train_dl_opts
        self.valid_ds_opts = valid_ds_opts
        self.valid_dl_opts = valid_dl_opts
        self.stats = hp.read_smth_stats()
        min_height = self.stats['min_height'].item()
        min_width = self.stats['min_width'].item()
        tfms = []
        if db_opts.frame_size > min_height or db_opts.frame_size > min_width:
            tfms.append(pipe.Pad(self.db_opts.frame_size, self.db_opts.frame_size))

        train_tfms = tfms.copy()
        train_tfms.extend([
            pipe.RandomCrop(db_opts.frame_size),
            pipe.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            pipe.ToVolumeArray(3, True) if self.db_opts.shape == 'volume' else pipe.ClipToStackedArray(3),
        ])
        train_ds_opts.do.transform = pipe.VideoCompose(train_tfms)

        valid_tfms = tfms.copy()
        valid_tfms.extend([
            pipe.CenterCrop(db_opts.frame_size),
            pipe.ToVolumeArray() if self.db_opts.shape == 'volume' else pipe.ClipToStackedArray(3),
        ])
        valid_ds_opts.do.transform = pipe.VideoCompose(valid_tfms)

        self.train_set = pipe.SmthDataset(self.train_ds_opts.do, self.train_ds_opts.so)
        self.valid_set = pipe.SmthDataset(self.valid_ds_opts.do, self.valid_ds_opts.so)

        self.train_sampler = None
        self.valid_sampler = None
        if self.db_opts.distributed:
            self.train_sampler = thd.distributed.DistributedSampler(self.train_set)
            self.train_dl_opts.shuffle = False
            self.valid_sampler = thd.distributed.DistributedSampler(self.valid_set)
            self.valid_dl_opts.shuffle = False

        self.train_loader = thd.DataLoader(self.train_set,
                                           collate_fn=self.train_set.collate,
                                           sampler=self.train_sampler,
                                           **asdict(self.train_dl_opts))
        self.valid_loader = thd.DataLoader(self.valid_set,
                                           collate_fn=self.valid_set.collate,
                                           sampler=self.valid_sampler,
                                           **asdict(self.valid_dl_opts))

    def get_batch(self, n: int, spl: str) -> Tuple[List[pipe.Video], List[pipe.Label]]:
        """Retrieve a random batch from one of the datasets."""
        assert spl in ['train', 'valid'], f'Unknown split: {spl}.'

        if spl == 'train':
            batch = self.train_set.get_batch(n)
        else:
            batch = self.valid_set.get_batch(n)

        return batch

    def __str__(self):
        return (f"""Something-Something-v2 DataBunch.
            [DataBunch config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.db_opts).items())}] 
            [Train Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.train_ds_opts).items())}]
            [Valid Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.valid_ds_opts).items())}]
            [Train Loader Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.train_dl_opts).items())}]
            [Valid Loader Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.valid_dl_opts).items())}]
            [Train Set: {self.train_set}]
            [Valid Set: {self.valid_set}]""")


if __name__ == '__main__':
    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    _db_opts = pipe.DataBunchOptions('volume', 224)

    _train_do = pipe.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=1.0,
        setting='valid',
        transform=None,
        keep=2
    )
    _train_so = pipe.SamplingOptions(
        num_segments=4,
        segment_size=1
    )
    _train_dl_opts = pipe.DataLoaderOptions(
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    _valid_do = pipe.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=1.0,
        setting='valid',
        transform=None,
        keep=2
    )
    _valid_so = pipe.SamplingOptions(
        num_segments=4,
        segment_size=1
    )
    _train_ds_opts = pipe.DataSetOptions(
        do=_train_do,
        so=_train_so
    )
    _valid_ds_opts = pipe.DataSetOptions(
        do=_valid_do,
        so=_valid_so
    )
    _valid_dl_opts = pipe.DataLoaderOptions(
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    bunch = SmthDataBunch(_db_opts, _train_ds_opts, _valid_ds_opts, _train_dl_opts, _valid_dl_opts)
    print(bunch)
    for i, (train_x, train_y) in tqdm(enumerate(bunch.train_loader), total=len(bunch.train_loader)):
        continue
    for i, (valid_x, valid_y) in tqdm(enumerate(bunch.valid_loader), total=len(bunch.valid_loader)):
        continue
    assert th.all(th.eq(train_x, valid_x))
    assert th.all(th.eq(train_y, valid_y))
    print('Everything is fine.')
