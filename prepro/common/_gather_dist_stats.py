import pathlib as pl
from typing import List, Tuple

import numpy as np
import skvideo.io

import constants as ct
import helpers as ghp
import prepro.helpers as php
from env import logging


def _calculate_stats(batch: Tuple[int, List[pl.Path]]) -> ghp.parallel.Result:
    """Compute mean, and variance over batch of videos and count number of observations."""
    no, batch = batch
    videos = []
    for filename in batch:
        video = skvideo.io.vread(filename.as_posix()) / 255  # bring to range 0.0-1.0
        length, width, height, channels = video.shape
        video = video.reshape(length * width * height, channels)
        videos.append(video)
    videos = np.vstack(videos)

    obs, channels = videos.shape
    mean = videos.mean(axis=0)
    var = videos.var(axis=0)

    return ghp.parallel.Result(len(batch), [(mean, var, obs)])


def store_stats(mean: np.ndarray, var: np.ndarray, std: np.ndarray, stats_path: pl.Path) -> None:
    """Store statistics in the merged stats DataFrame."""
    stats = ghp.read_stats(stats_path)

    stats['mean_r'] = mean[0]
    stats['mean_g'] = mean[1]
    stats['mean_b'] = mean[2]

    stats['var_r'] = var[0]
    stats['var_g'] = var[1]
    stats['var_b'] = var[2]

    stats['std_r'] = std[0]
    stats['std_g'] = std[1]
    stats['std_b'] = std[2]

    stats.to_json(stats_path, orient='index')

    return


def main(dataset: str, split: int):
    """Iteratively compute mean, var, std for entire dataset over channels in range 0-1."""
    stats_path = php.get_stats_path(dataset, split)
    video_paths = php.get_video_paths(dataset)

    logging.info(f'Gathering distribution statistics for the {ct.SETTING} set...')
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float)
    var = np.array([0.0, 0.0, 0.0], dtype=np.float)
    std = np.array([0.0, 0.0, 0.0], dtype=np.float)
    total_obs = 0.0

    for batch_stats in ghp.parallel.execute(_calculate_stats, video_paths, 1):
        cur_mean, cur_var, obs = batch_stats
        mean = total_obs / (total_obs + obs) * mean + obs / (obs + total_obs) * cur_mean
        var = (total_obs / (total_obs + obs) * var + obs / (total_obs + obs) * cur_var +
               total_obs * obs / (total_obs + obs) ** 2 * (mean - cur_mean) ** 2)
        std = np.sqrt(var)
        total_obs += obs
    store_stats(mean, var, std, stats_path)
    logging.info('...Done.')
