import constants as ct
import helpers as hp
from env import logging


def merge_meta() -> None:
    """Merges train and valid DataFrames into single DataFrame and stores as json."""
    train = hp.read_smth_meta(ct.SMTH_META_TRAIN)
    valid = hp.read_smth_meta(ct.SMTH_META_VALID)

    train['split'] = 'train'
    valid['split'] = 'valid'

    merged = train.append(valid, ignore_index=True, verify_integrity=True)
    logging.info(f'Merged: {len(train)} + {len(valid)} = {len(merged)}.')
    merged.to_json(ct.SMTH_META_MERGED, orient='records')

    assert len(merged) == len(train) + len(valid)


def main():
    logging.info('Merging train and valid DataFrames into one...')
    merge_meta()
    logging.info('...Done.')
