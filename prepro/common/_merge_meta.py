import pathlib as pl

import constants as ct
import env
import helpers as ghp
import prepro.helpers as php


def merge_meta(train_path: pl.Path, dev_path: pl.Path, merged_path: pl.Path) -> None:
    """Merges train and dev DataFrames into single DataFrame and stores as json."""
    train_meta = ghp.read_meta(train_path)
    dev_meta = ghp.read_meta(dev_path)

    train_meta['split'] = 'train'
    dev_meta['split'] = 'dev'

    merged = train_meta.append(dev_meta, verify_integrity=True)
    assert len(merged) == len(train_meta) + len(dev_meta)
    env.LOGGER.info(f'Merged: {len(train_meta)} + {len(dev_meta)} = {len(merged)}.')

    merged.to_json(ct.WORK_ROOT / merged_path, orient='index')


def main(dataset: str, split: int):
    env.LOGGER.info(f'Merging {dataset} train and dev DataFrames into one...')
    [train, dev, merged, _] = php.get_meta_paths(dataset, split)
    merge_meta(train, dev, merged)
    env.LOGGER.info('...Done.')
