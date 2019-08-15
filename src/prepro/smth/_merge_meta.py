import constants as ct
import helpers as hp
from env import logging


def merge_meta() -> None:
    """Merges train and dev DataFrames into single DataFrame and stores as json."""
    train = hp.read_smth_meta(ct.SMTH_META_TRAIN)
    dev = hp.read_smth_meta(ct.SMTH_META_DEV)

    train['split'] = 'train'
    dev['split'] = 'dev'

    merged = train.append(dev, ignore_index=True, verify_integrity=True)
    logging.info(f'Merged: {len(train)} + {len(dev)} = {len(merged)}.')
    merged.to_json(ct.SMTH_META_MERGED, orient='records')

    assert len(merged) == len(train) + len(dev)


def main():
    logging.info('Merging train and dev DataFrames into one...')
    merge_meta()
    logging.info('...Done.')
