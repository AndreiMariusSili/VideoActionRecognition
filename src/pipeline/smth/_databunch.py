from typing import Optional, Dict, Any, Union, Tuple
from torch.utils import data as thd
from tqdm import tqdm
import pandas as pd
import os

import pipeline as pipe
import constants as ct
import helpers as hp


class SmthDataBunch(object):
    cut: float
    shape: str
    size: Union[int, Tuple[int, int]]
    dl_args: Dict[str, Any]
    test: bool
    stats: pd.DataFrame
    train_set: 'pipe.SmthDataset'
    train_loader: 'thd.DataLoader'
    valid_set: 'pipe.SmthDataset'
    valid_loader: 'thd.DataLoader'
    test_set: Optional['pipe.SmthDataset']
    test_loader: Optional['thd.DataLoader']

    def __init__(self, cut: float, shape: str, size: Union[int, Tuple[int, int]], test: bool, dl_args: Dict[str, Any]):
        assert shape in ['stack', 'volume'], f'Unknown shape {shape}. Possible shapes are "stack" and "volume".'

        self.cut = cut
        self.shape = shape
        self.size = size
        self.test = test
        self.dl_args = dl_args

        self.stats = hp.read_smth_stats()
        base_transform = pipe.TransformComposition([
            pipe.Resize(80, 'inter_area'),
            pipe.Normalize(255),
            pipe.Standardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS),
            pipe.FramePad(ct.IMAGE_NET_STD_HEIGHT, ct.IMAGE_NET_STD_WIDTH, False),
            pipe.ToVolumeArray() if self.shape == 'volume' else pipe.ToStackedArray()
        ])
        self.train_set = pipe.SmthDataset(ct.SMTH_META_TRAIN, self.cut, transform=base_transform)
        self.valid_set = pipe.SmthDataset(ct.SMTH_META_VALID, self.cut, transform=base_transform)
        self.train_loader = thd.DataLoader(self.train_set, collate_fn=self.train_set.collate, **self.dl_args)
        self.valid_loader = thd.DataLoader(self.valid_set, collate_fn=self.valid_set.collate, **self.dl_args)
        if self.test:
            self.test_set = pipe.SmthDataset(ct.SMTH_META_TEST, self.cut, transform=base_transform)
            self.test_loader = thd.DataLoader(self.test_set, collate_fn=self.test_set.collate, **self.dl_args)
        else:
            self.test_set = None
            self.test_loader = None

    def __str__(self):
        return (f"""Something-Something DataBunch. 
            [cut: {self.cut}]
            [shape: {self.shape}]
            [data loader: {" ".join("{}={}".format(k, v) for k, v in self.dl_args.items())}]
            [Train Set: {self.train_set}]
            [Valid Set: {self.valid_set}]
            [Test Set: {self.test_set}]""")


if __name__ == '__main__':
    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    bunch = SmthDataBunch(1.0, 'volume', 80, False, dict(batch_size=24, pin_memory=True, shuffle=True, num_workers=8))
    print(bunch)
    for i, _ in tqdm(enumerate(bunch.train_loader)):
        continue
