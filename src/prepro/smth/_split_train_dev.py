from typing import Tuple

import pandas as pd
import sklearn.model_selection as skms

import constants as ct
import helpers as hp
from env import logging


def _split_train_dev() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a meta file into train and dev."""
    old_train = hp.read_smth_meta(ct.SMTH_META_TRAIN).sort_index()
    train, dev = skms.train_test_split(old_train, random_state=ct.RANDOM_STATE,
                                       stratify=old_train.template_id, test_size=ct.DEV_SIZE)

    return train, dev


def main():
    logging.info(f'Splitting into train and dev for the {ct.SETTING} set...')

    meta_train, meta_dev = _split_train_dev()
    meta_train.to_json(ct.SMTH_META_TRAIN, orient='records')
    meta_dev.to_json(ct.SMTH_META_DEV, orient='records')

    logging.info('...Done')
