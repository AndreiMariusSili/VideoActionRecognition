import pathlib as pl
from typing import Dict, List

import pandas as pd

import env
import helpers as ghp
import prepro.helpers as php


def _store_stats(stats: Dict[str, List[float]], stats_path: pl.Path) -> None:
    """Store the dimension statistics in the merged stats file."""
    pd.DataFrame.from_dict(data=stats, orient='columns').to_json(stats_path, orient='index')


def main(dataset: str, split: int):
    """Find maximum width and maximum height and store in a DataFrame."""
    _, _, merged_meta_path, _ = php.get_meta_paths(dataset, split)
    stats_path = php.get_stats_path(dataset, split)

    env.LOGGER.info(f'Gathering size statistics for the {dataset} dataset...')
    meta = ghp.read_meta(merged_meta_path)
    stats = {
        'min_length': [meta['length'].min()],
        'mean_length': [meta['length'].mean()],
        'max_length': [meta['length'].max()],
        'min_height': [meta['height'].min()],
        'mean_height': [meta['height'].mean()],
        'max_height': [meta['height'].max()],
        'min_width': [meta['width'].min()],
        'mean_width': [meta['width'].mean()],
        'max_width': [meta['width'].max()],
    }
    _store_stats(stats, stats_path)
    env.LOGGER.info('...Done.')
