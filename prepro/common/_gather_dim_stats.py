import math
import pathlib as pl
from typing import Dict, List, Tuple

import pandas as pd
import skvideo.io

import constants as ct
import helpers as ghp
import prepro.helpers as php
from env import logging


def _gather_size_stats(batch: Tuple[int, List[pl.Path]]) -> ghp.parallel.Result:
    """Gather length, height, width, channel dimensions for a batch."""
    no, batch = batch

    stats = []
    for filename in batch:
        video = skvideo.io.vread(filename.as_posix())
        stat = {
            'length': video.shape[0],
            'height': video.shape[1],
            'width': video.shape[2]
        }
        stats.append(stat)

    return ghp.parallel.Result(len(batch), stats)


def compute_stats(stats: int, obs: int, min_stat: int, mean_stat: float, max_stat: int) -> Tuple[int, float, int]:
    if stats > max_stat:
        max_stat = stats
    if stats < min_stat:
        min_stat = stats
    mean_stat = mean_stat + 1 / obs * (stats - mean_stat)

    return min_stat, mean_stat, max_stat


def _store_stats(stats: Dict[str, Tuple[int, float, int]], stats_path: pl.Path) -> None:
    """Store the dimension statistics in the merged stats file."""
    keys = ('length', 'height', 'width')
    data = []
    for key in keys:
        data.extend(stats[key])

    columns = [
        'min_length', 'mean_length', 'max_length',
        'min_height', 'mean_height', 'max_height',
        'min_width', 'mean_width', 'max_width',
    ]
    pd.DataFrame(data=[data], columns=columns).to_json(stats_path, orient='index')


def main(dataset: str, split: int):
    """Find maximum width and maximum height and store in a DataFrame."""
    stats_path = php.get_stats_path(dataset, split)
    video_paths = php.get_video_paths(dataset)

    logging.info(f'Gathering size statistics for the {ct.SETTING} {dataset} set...')
    global_stats = {
        'length': (math.inf, 0, -math.inf),
        'height': (math.inf, 0, -math.inf),
        'width': (math.inf, 0, -math.inf),
    }

    for obs, stats in enumerate(ghp.parallel.execute(_gather_size_stats, video_paths, 1)):
        obs += 1
        for stat, value in stats.items():
            global_stats[stat] = compute_stats(value, obs, *global_stats[stat])
    _store_stats(global_stats, stats_path)
    logging.info('...Done.')
