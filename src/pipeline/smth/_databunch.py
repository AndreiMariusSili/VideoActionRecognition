from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union, Tuple
from tqdm import tqdm
import pandas as pd
import os

from pipeline._transforms import TransformComposition, Normalize, Standardize, ToVolumeArray, ToStackedArray, Resize
from pipeline.smth._dataset import SmthDataset
import constants as ct
import helpers as hp


class SmthDataBunch(object):
    cut: float
    shape: str
    size: Union[int, Tuple[int, int]]
    dl_args: Dict[str, Any]
    test: bool
    stats: pd.DataFrame
    train_set: 'SmthDataset'
    train_loader: 'DataLoader'
    valid_set: 'SmthDataset'
    valid_loader: 'DataLoader'
    test_set: Optional['SmthDataset']
    test_loader: Optional['DataLoader']

    def __init__(self, cut: float, shape: str, size: Union[int, Tuple[int, int]], test: bool, dl_args: Dict[str, Any]):
        assert shape in ['stack', 'volume'], f'Unknown shape {shape}. Possible shapes are "stack" and "volume".'

        self.cut = cut
        self.shape = shape
        self.size = size
        self.test = test
        self.dl_args = dl_args

        self.stats = hp.read_smth_stats()
        base_transform = TransformComposition([
            Resize(80, 'inter_area'),
            Normalize(255),
            Standardize((self.stats['mean_r'], self.stats['mean_g'], self.stats['mean_b']),
                        (self.stats['std_r'], self.stats['std_g'], self.stats['std_b'])),
            ToVolumeArray() if self.shape == 'volume' else ToStackedArray()
        ])
        self.train_set = SmthDataset(ct.SMTH_META_TRAIN, self.cut, transform=base_transform)
        self.valid_set = SmthDataset(ct.SMTH_META_VALID, self.cut, transform=base_transform)
        self.train_loader = DataLoader(self.train_set, collate_fn=self.train_set.collate, **self.dl_args)
        self.valid_loader = DataLoader(self.valid_set, collate_fn=self.valid_set.collate, **self.dl_args)
        if self.test:
            self.test_set = SmthDataset(ct.SMTH_META_TEST, self.cut, transform=base_transform)
            self.test_loader = DataLoader(self.test_set, collate_fn=self.test_set.collate, **self.dl_args)
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
