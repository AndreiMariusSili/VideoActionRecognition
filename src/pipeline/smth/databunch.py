import os
from typing import List, Optional, Tuple

import pandas as pd
from dataclasses import asdict
from torch.utils import data as thd
from tqdm import tqdm

import constants as ct
import helpers as hp
import options.pipe_options as pio
import pipeline.label as pil
import pipeline.smth.dataset as dataset
import pipeline.transforms as pit
import pipeline.video as piv


class SmthDataBunch(object):
    db_opts: 'pio.DataBunchOptions'
    train_ds_opts: 'pio.DataSetOptions'
    valid_ds_opts: 'pio.DataSetOptions'
    train_dl_opts: 'pio.DataLoaderOptions'
    valid_dl_opts: 'pio.DataLoaderOptions'
    stats: 'pd.DataFrame'

    train_set: 'dataset.SmthDataset'
    train_sampler: Optional['thd.DistributedSampler']
    train_loader: 'thd.DataLoader'
    valid_set: 'dataset.SmthDataset'
    valid_sampler: Optional['thd.DistributedSampler']
    valid_loader: 'thd.DataLoader'

    def __init__(self, db_opts: pio.DataBunchOptions,
                 train_ds_opts: pio.DataSetOptions, dev_ds_opts: pio.DataSetOptions,
                 valid_ds_opts: pio.DataSetOptions,
                 train_dl_opts: pio.DataLoaderOptions, dev_dl_opts: pio.DataLoaderOptions,
                 valid_dl_opts: pio.DataLoaderOptions):
        assert db_opts.shape in ['stack', 'volume'], f'Unknown shape {db_opts.shape}.'
        self.db_opts = db_opts
        self.train_ds_opts = train_ds_opts
        self.dev_ds_opts = dev_ds_opts
        self.valid_ds_opts = valid_ds_opts
        self.train_dl_opts = train_dl_opts
        self.dev_dl_opts = dev_dl_opts
        self.valid_dl_opts = valid_dl_opts
        self.stats = hp.read_smth_stats()

        min_height = self.stats['min_height'].item()
        min_width = self.stats['min_width'].item()
        tfms = []
        if db_opts.frame_size > min_height or db_opts.frame_size > min_width:
            tfms.append(pit.Pad(self.db_opts.frame_size, self.db_opts.frame_size))

        train_tfms = tfms.copy()
        train_tfms.extend([
            pit.RandomCrop(db_opts.frame_size),
            pit.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            pit.ToVolumeArray(3, True) if self.db_opts.shape == 'volume' else pit.ClipToStackedArray(3),
        ])
        train_ds_opts.do.transform = pit.VideoCompose(train_tfms)

        dev_tfms = tfms.copy()
        dev_tfms.extend([
            pit.CenterCrop(db_opts.frame_size),
            pit.ToVolumeArray(3, True) if self.db_opts.shape == 'volume' else pit.ClipToStackedArray(3),
        ])
        dev_ds_opts.do.transform = pit.VideoCompose(dev_tfms)

        valid_tfms = tfms.copy()
        valid_tfms.extend([
            pit.CenterCrop(db_opts.frame_size),
            pit.ToVolumeArray() if self.db_opts.shape == 'volume' else pit.ClipToStackedArray(3),
        ])
        valid_ds_opts.do.transform = pit.VideoCompose(valid_tfms)

        self.train_set = dataset.SmthDataset(self.train_ds_opts.do, self.train_ds_opts.so)
        self.dev_set = dataset.SmthDataset(self.dev_ds_opts.do, self.dev_ds_opts.so)
        self.valid_set = dataset.SmthDataset(self.valid_ds_opts.do, self.valid_ds_opts.so)

        self.train_sampler = None
        self.dev_sampler = None
        self.valid_sampler = None
        if self.db_opts.distributed:
            self.train_sampler = thd.distributed.DistributedSampler(self.train_set)
            self.train_dl_opts.shuffle = False
            self.dev_sampler = thd.distributed.DistributedSampler(self.dev_set)
            self.dev_dl_opts.shuffle = False
            self.valid_sampler = thd.distributed.DistributedSampler(self.valid_set)
            self.valid_dl_opts.shuffle = False

        self.train_loader = thd.DataLoader(self.train_set,
                                           collate_fn=self.train_set.collate,
                                           sampler=self.train_sampler,
                                           **asdict(self.train_dl_opts))
        self.dev_loader = thd.DataLoader(self.dev_set,
                                         collate_fn=self.dev_set.collate,
                                         sampler=self.dev_sampler,
                                         **asdict(self.dev_dl_opts))
        self.valid_loader = thd.DataLoader(self.valid_set,
                                           collate_fn=self.valid_set.collate,
                                           sampler=self.valid_sampler,
                                           **asdict(self.valid_dl_opts))

    def get_batch(self, n: int, spl: str) -> Tuple[List[piv.Video], List[pil.Label]]:
        """Retrieve a random batch from one of the datasets."""
        assert spl in ['train', 'valid'], f'Unknown split: {spl}.'

        if spl == 'train':
            batch = self.train_set.get_batch(n)
        elif spl == 'dev':
            batch = self.dev_set.get_batch(n)
        else:
            batch = self.valid_set.get_batch(n)

        return batch

    def __str__(self):
        return (f"""Something-Something-v2 DataBunch.
            [DataBunch config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.db_opts).items())}] 
            [Train Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.train_ds_opts).items())}]
            [Dev Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dev_ds_opts).items())}]
            [Valid Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.valid_ds_opts).items())}]
            [Train Loader Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.train_dl_opts).items())}]
            [Dev Loader Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dev_dl_opts).items())}]
            [Valid Loader Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.valid_dl_opts).items())}]
            [Train Set: {self.train_set}]
            [Dev Set: {self.dev_set}]
            [Valid Set: {self.valid_set}]""")


if __name__ == '__main__':
    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    _db_opts = pio.DataBunchOptions('volume', 224)
    # TRAIN
    _train_do = pio.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=1.0,
        setting='valid',
        transform=None,
        keep=0.01
    )
    _train_so = pio.SamplingOptions(
        num_segments=4,
        segment_size=1
    )
    _train_ds_opts = pio.DataSetOptions(
        do=_train_do,
        so=_train_so
    )
    _train_dl_opts = pio.DataLoaderOptions(
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # DEV
    _dev_do = pio.DataOptions(
        meta_path=ct.SMTH_META_DEV,
        cut=1.0,
        setting='valid',
        transform=None,
        keep=0.01
    )
    _dev_so = pio.SamplingOptions(
        num_segments=4,
        segment_size=1
    )
    _dev_ds_opts = pio.DataSetOptions(
        do=_dev_do,
        so=_dev_so
    )
    _dev_dl_opts = pio.DataLoaderOptions(
        batch_size=2,
        shuffle=False,
        num_workers=0
    )

    # VALID
    _valid_do = pio.DataOptions(
        meta_path=ct.SMTH_META_VALID,
        cut=1.0,
        setting='valid',
        transform=None,
        keep=0.01
    )
    _valid_so = pio.SamplingOptions(
        num_segments=4,
        segment_size=1
    )
    _valid_ds_opts = pio.DataSetOptions(
        do=_valid_do,
        so=_valid_so
    )
    _valid_dl_opts = pio.DataLoaderOptions(
        batch_size=2,
        shuffle=False,
        num_workers=0
    )
    bunch = SmthDataBunch(_db_opts, _train_ds_opts, _dev_ds_opts, _valid_ds_opts,
                          _train_dl_opts, _dev_dl_opts, _valid_dl_opts)
    print(bunch)

    tqdm().clear()
    for i, (train_x, train_y) in tqdm(enumerate(bunch.train_loader), total=len(bunch.train_loader)):
        continue
    for i, (valid_x, valid_y) in tqdm(enumerate(bunch.valid_loader), total=len(bunch.valid_loader)):
        continue
    for i, (train_x, train_y) in tqdm(enumerate(bunch.dev_loader), total=len(bunch.dev_loader)):
        continue
    tqdm().clear()
    print('Everything is fine.')
