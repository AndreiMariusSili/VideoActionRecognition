from typing import Optional, Tuple, List
from torch.utils import data as thd
from dataclasses import asdict
from tqdm import tqdm
import pandas as pd
import os

import pipeline as pipe
import constants as ct
import helpers as hp


class SmthDataBunch(object):
    ds_opts: 'pipe.DataSetOptions'
    dl_opts: 'pipe.DataLoaderOptions'
    stats: 'pd.DataFrame'

    train_set: 'pipe.SmthDataset'
    train_loader: 'thd.DataLoader'
    valid_set: 'pipe.SmthDataset'
    valid_loader: 'thd.DataLoader'
    test_set: Optional['pipe.SmthDataset']
    test_loader: Optional['thd.DataLoader']

    def __init__(self, db_opts: pipe.DataBunchOptions, ds_opts: pipe.DataSetOptions, dl_opts: pipe.DataLoaderOptions):
        assert db_opts.shape in ['stack', 'volume'], f'Unknown shape {db_opts.shape}.'

        self.db_opts = db_opts
        self.ds_opts = ds_opts
        self.dl_opts = dl_opts
        self.stats = hp.read_smth_stats()

        train_transform = pipe.VideoCompose([
            pipe.RandomCrop(db_opts.frame_size),
            pipe.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            pipe.ToVolumeArray(3, True) if self.db_opts.shape == 'volume' else pipe.ClipToStackedArray(3),
            pipe.ArrayNormalize(255),
            pipe.ArrayStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS),
        ])
        valid_transform = pipe.VideoCompose([
            pipe.CenterCrop(db_opts.frame_size),
            pipe.ToVolumeArray() if self.db_opts.shape == 'volume' else pipe.ClipToStackedArray(3),
            pipe.ArrayNormalize(255),
            pipe.ArrayStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS),
        ])

        self.train_set = pipe.SmthDataset(ct.SMTH_META_TRAIN, transform=train_transform, **asdict(self.ds_opts))
        self.valid_set = pipe.SmthDataset(ct.SMTH_META_VALID, transform=valid_transform, **asdict(self.ds_opts))
        self.train_loader = thd.DataLoader(self.train_set, collate_fn=self.train_set.collate, **asdict(self.dl_opts))
        self.valid_loader = thd.DataLoader(self.valid_set, collate_fn=self.valid_set.collate, **asdict(self.dl_opts))
        if self.db_opts.test:
            self.test_set = pipe.SmthDataset(ct.SMTH_META_TEST, transform=valid_transform, **asdict(self.ds_opts))
            self.test_loader = thd.DataLoader(self.test_set, collate_fn=self.test_set.collate, **asdict(self.dl_opts))
        else:
            self.test_set = None
            self.test_loader = None

    def get_batch(self, n: int, spl: str)-> Tuple[List[pipe.Video], List[pipe.Label]]:
        """Retrieve a random batch from one of the datasets."""
        assert spl in ['train', 'valid', 'test'], f'Unknown split: {spl}.'

        if spl == 'train':
            batch = self.train_set.get_batch(n)
        elif spl == 'valid':
            batch = self.valid_set.get_batch(n)
        else:
            batch = self.test_set.get_batch(n)

        return batch

    def __str__(self):
        return (f"""Something-Something-v2 DataBunch.
            [databunch config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.db_opts).items())}] 
            [dataset config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.ds_opts).items())}]
            [loader config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dl_opts).items())}]
            [Train Set: {self.train_set}]
            [Valid Set: {self.valid_set}]
            [Test Set: {self.test_set}]""")


if __name__ == '__main__':
    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    _db_opts = pipe.DataBunchOptions('volume', 224, False)
    _ds_opts = pipe.DataSetOptions(1.0, 16)
    _dl_opts = pipe.DataLoaderOptions(8, False, 0, False, False)
    bunch = SmthDataBunch(_db_opts, _ds_opts, _dl_opts)
    print(bunch)
    for i, _ in tqdm(enumerate(bunch.train_loader), total=len(bunch.train_loader)):
        if i > 10:
            break
