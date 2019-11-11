import os
from dataclasses import asdict
from typing import List, Tuple

from torch.utils import data as thd
from tqdm import tqdm

import databunch.dataset as dataset
import databunch.label as pil
import databunch.transforms as dt
import databunch.video as piv
import helpers as hp
import options.data_options as do


class VideoDataBunch(object):
    def __init__(self, db_opts: do.DataBunchOptions):
        assert db_opts.shape in ['stack', 'volume'], f'Unknown shape {db_opts.shape}.'
        self.dbo = db_opts
        self.train_dso = self.dbo.train_dso
        self.dev_dso = self.dbo.dev_dso
        self.test_dso = self.dbo.test_dso
        self.stats = hp.read_stats(self.dbo.stats_path)

        min_height = self.stats.loc[0, 'min_height']
        min_width = self.stats.loc[0, 'min_width']
        tfms = []
        if self.dbo.frame_size > min_height or self.dbo.frame_size > min_width:
            tfms.append(dt.Pad(self.dbo.frame_size, self.dbo.frame_size))

        train_tfms = tfms.copy()
        self._compose_transforms(train_tfms, self.train_dso.do.setting)
        self.train_dso.do.transform = dt.VideoCompose(train_tfms)
        dev_tfms = tfms.copy()
        self._compose_transforms(dev_tfms, self.dev_dso.do.setting)
        self.dev_dso.do.transform = dt.VideoCompose(dev_tfms)
        valid_tfms = tfms.copy()
        self._compose_transforms(valid_tfms, self.test_dso.do.setting)
        self.test_dso.do.transform = dt.VideoCompose(valid_tfms)

        self.train_set = dataset.VideoDataset(self.dbo.cut, self.train_dso.do, self.train_dso.so)
        self.dev_set = dataset.VideoDataset(self.dbo.cut, self.dev_dso.do, self.dev_dso.so)
        self.test_set = dataset.VideoDataset(self.dbo.cut, self.test_dso.do, self.test_dso.so)

        self.lids = self.train_set.lids

        self.train_sampler = None
        self.dev_sampler = None
        self.test_sampler = None
        if self.dbo.distributed:
            self.dbo.dlo.shuffle = False
            self.train_sampler = thd.distributed.DistributedSampler(self.train_set)
            self.dev_sampler = thd.distributed.DistributedSampler(self.dev_set)
            self.test_sampler = thd.distributed.DistributedSampler(self.test_set)

        self.train_loader = thd.DataLoader(self.train_set,
                                           collate_fn=self.train_set.collate,
                                           sampler=self.train_sampler,
                                           **asdict(self.dbo.dlo))
        self.dev_loader = thd.DataLoader(self.dev_set,
                                         collate_fn=self.dev_set.collate,
                                         sampler=self.dev_sampler,
                                         **asdict(self.dbo.dlo))
        self.test_loader = thd.DataLoader(self.test_set,
                                          collate_fn=self.test_set.collate,
                                          sampler=self.test_sampler,
                                          **asdict(self.dbo.dlo))

    def _compose_transforms(self, tfms: List[dt.VideoTransform], setting: str) -> None:
        """Create transformation pipeline depending on setting."""
        assert setting in ['train', 'eval'], f'Unknown setting: {setting}.'
        if setting == 'train':
            tfms.extend([
                dt.RandomCrop(self.dbo.frame_size),
                dt.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                dt.ToVolumeArray(3, True) if self.dbo.shape == 'volume' else dt.ClipToStackedArray(3),
                dt.ArrayNormalize(255)
            ])
        else:
            tfms.extend([
                dt.CenterCrop(self.dbo.frame_size),
                dt.ToVolumeArray(3, True) if self.dbo.shape == 'volume' else dt.ClipToStackedArray(3),
                dt.ArrayNormalize(255)
            ])

        return

    def get_batch(self, n: int, spl: str) -> Tuple[List[piv.Video], List[pil.Label]]:
        """Retrieve a random batch from one of the datasets."""
        assert spl in ['train', 'dev', 'valid'], f'Unknown split: {spl}.'

        if spl == 'train':
            batch = self.train_set.get_batch(n)
        elif spl == 'dev':
            batch = self.dev_set.get_batch(n)
        else:
            batch = self.test_set.get_batch(n)

        return batch

    def __str__(self):
        return (f"""Something-Something-v2 DataBunch.
            [DataBunch config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dbo).items())}] 
            [Train Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.train_dso).items())}]
            [Dev Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dev_dso).items())}]
            [Test Dataset Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.test_dso).items())}]
            [DataLoader Config: {" ".join("{}={}".format(k, v) for k, v in asdict(self.dbo.dlo).items())}]
            [Train Set: {self.train_set}]
            [Dev Set: {self.dev_set}]
            [Test Set: {self.test_set}]""")


if __name__ == '__main__':
    os.chdir('/Users/Play/Code/AI/master-thesis/src')

    import specs

    specs.datasets.common.dlo.batch_size = 4
    specs.datasets.hmdb1.dbo.cut = 1.0

    bunch = VideoDataBunch(specs.datasets.hmdb1.dbo)
    print(bunch)

    tqdm().clear()
    for i, (x, y, videos, labels) in tqdm(enumerate(bunch.train_loader), total=len(bunch.train_loader)):
        continue
    tqdm().clear()
    for i, (x, y, videos, labels) in tqdm(enumerate(bunch.test_loader), total=len(bunch.test_loader)):
        continue
    tqdm().clear()
    for i, (x, y, videos, labels) in tqdm(enumerate(bunch.dev_loader), total=len(bunch.dev_loader)):
        continue
    tqdm().clear()
