from typing import List, Tuple, Dict
import pandas as pd
import skvideo.io
import math

from env import logging
import constants as ct
import helpers as hp


def _gather_size_stats(batch: Tuple[int, List[str]]) -> hp.parallel.Result:
    """Gather length, height, width, channel dimensions for a batch."""
    no, batch = batch

    stats = []
    for filename in batch:
        video = skvideo.io.vread(filename)
        stat = {
            'length': video.shape[0],
            'height': video.shape[1],
            'width': video.shape[2]
        }
        stats.append(stat)

    return hp.parallel.Result(len(batch), stats)


def compute_stats(stats: int, obs: int, min_stat: int, mean_stat: float, max_stat: int) -> Tuple[int, float, int]:
    if stats > max_stat:
        max_stat = stats
    if stats < min_stat:
        min_stat = stats
    mean_stat = mean_stat + 1 / obs * (stats - mean_stat)

    return min_stat, mean_stat, max_stat


def _store_stats(stats: Dict[str, Tuple[int, float, int]]) -> None:
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
    (pd.DataFrame(data=[data], columns=columns)
     .to_json(ct.SMTH_STATS_MERGED, orient='records'))


def main():
    """Find maximum width and maximum height and store in a DataFrame."""
    logging.info(f'Gathering size statistics for the {ct.SETTING} set...')
    global_stats = {
        'length': (math.inf, 0, -math.inf),
        'height': (math.inf, 0, -math.inf),
        'width': (math.inf, 0, -math.inf),
    }
    videos = hp.get_smth_videos()
    for obs, stats in enumerate(hp.parallel.execute(_gather_size_stats, videos, 1)):
        obs += 1
        for stat, value in stats.items():
            global_stats[stat] = compute_stats(value, obs, *global_stats[stat])
    _store_stats(global_stats)
    logging.info('...Done.')
