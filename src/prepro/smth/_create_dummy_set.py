import os
import random
import shutil
from typing import Tuple

import pandas as pd

import constants as ct
from env import logging
import helpers as hp


def create_dummy_labels(full_labels: pd.DataFrame, dummy_labels2id: pd.DataFrame,
                        dummy_split: str, split_name: str) -> pd.DataFrame:
    """Create the dummy labels DataFrame by matching with the dummy labels2id templates and sampling from ."""
    mask = full_labels['template'].isin(dummy_labels2id['template'])
    dummy_labels = full_labels[mask].reset_index(drop=True)

    samples = []
    for template in dummy_labels2id['template']:
        if split_name == 'train':
            n = random.randint(*ct.SMTH_TRAIN_DUMMY_SAMPLE)
        elif split_name == 'valid':
            n = random.randint(*ct.SMTH_VALID_DUMMY_SAMPLE)
        else:
            raise ValueError(f'Unknown split: {split_name}')

        sample = dummy_labels[dummy_labels['template'] == template].sample(n=n, random_state=1).index
        samples.extend(sample)

    dummy_labels = dummy_labels.loc[samples]
    dummy_labels.to_json(dummy_split, orient='records')

    return dummy_labels


def create_dummy_inputs(dummy_labels: pd.DataFrame) -> Tuple[str, str, int]:
    """Copy .webm files with an id in the dummy labels DataFrame to the dummy folder."""
    os.makedirs(ct.SMTH_VIDEO_DIR, exist_ok=True)

    smth_full_raw_data = hp.change_setting(ct.SMTH_VIDEO_DIR, ct.SETTING, 'full')
    for _id in dummy_labels['id']:
        shutil.copy(f'{os.path.join(smth_full_raw_data, _id)}.webm', f'{os.path.join(ct.SMTH_VIDEO_DIR, _id)}.webm')

    return smth_full_raw_data, ct.SMTH_VIDEO_DIR, len(dummy_labels)


def main():
    """Create dummy sets for the something-something dataset based on hand-picked labels."""
    logging.info('Creating something-something dummy dataset...')
    # load dummy labels
    dummy_labels2id = hp.read_smth_labels2id(ct.SMTH_LABELS2ID)
    splits = [
        (ct.SMTH_META_TRAIN, hp.change_setting(ct.SMTH_META_TRAIN, ct.SETTING, 'full'), 'train'),
        (ct.SMTH_META_VALID, hp.change_setting(ct.SMTH_META_VALID, ct.SETTING, 'full'), 'valid'),
    ]
    logging.info('Loaded dummy labels2id DataFrame.')

    # for each split
    for dummy_split, full_split, split_name in splits:
        # load split
        full_labels = hp.read_smth_meta(full_split)
        logging.info(f'Loaded full labels DataFrame from {full_split}.')
        # create dummy label files
        dummy_labels = create_dummy_labels(full_labels, dummy_labels2id, dummy_split, split_name)
        logging.info(f'Created dummy labels DataFrame with {len(dummy_labels)} entries.')
        # copy matching inputs to dummy location
        _from, _to, total = create_dummy_inputs(dummy_labels)
        logging.info(f'Copied {total} inputs from {_from} to {_to}.')
    logging.info('...Done')
