import pathlib as pl
from typing import Tuple

import pandas as pd
import sklearn.model_selection as skm

import constants as ct
import env
import helpers as hp
import prepro.helpers as php


def _split_train_dev(meta_path: pl.Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a meta file into train and dev."""
    train = hp.read_meta(meta_path).sort_index()
    train, dev = skm.train_test_split(train, random_state=ct.RANDOM_STATE, stratify=train['lid'], test_size=ct.DEV_SIZE)

    return train.sort_index(), dev.sort_index()


def main(dataset: str, split: int):
    [train, dev, _, _] = php.get_meta_paths(dataset, split)
    env.LOGGER.info(f'Splitting {train} into train and dev...')
    meta_train, meta_dev = _split_train_dev(train)
    meta_train.to_json(ct.WORK_ROOT / train, orient='index')
    meta_dev.to_json(ct.WORK_ROOT / dev, orient='index')
    env.LOGGER.info('...Done')
