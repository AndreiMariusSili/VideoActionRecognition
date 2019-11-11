from typing import Any, List, Tuple

import pandas as pd
import skvideo.io

import helpers as ghp
import prepro.helpers as php
from env import logging

DATA_ROOT_DIR = None


def add_columns(meta: pd.DataFrame) -> None:
    """Add columns of interest to the DataFrame."""
    meta['length'] = None
    meta['height'] = None
    meta['width'] = None
    meta['framerate'] = None


def _augment_row(row: pd.Series) -> pd.Series:
    """Add video and label information to the row."""
    video_path = DATA_ROOT_DIR / row['video_path']

    video = skvideo.io.vread(video_path.as_posix())
    video_meta = skvideo.io.ffprobe(video_path)['video']

    row['height'], row['width'], row['length'], _ = video.shape
    row['framerate'] = int(video_meta['@avg_frame_rate'].split('/')[0])

    return row


def _augment_meta(batch: Tuple[int, List[Any]]) -> ghp.parallel.Result:
    """Create a batch of augmented rows."""
    no, batch = batch

    rows = []
    for index, row in batch:
        rows.append((index, _augment_row(row)))

    return ghp.parallel.Result(len(batch), rows)


def main(dataset: str, split: int):
    global DATA_ROOT_DIR
    DATA_ROOT_DIR = php.get_data_root_path(dataset)
    [train, dev, _, test] = php.get_meta_paths(dataset, split)

    for path in [train, dev, test]:
        logging.info(f'Augmenting metadata at {path.as_posix()}...')
        meta = ghp.read_meta(path)
        add_columns(meta)
        for index, row in ghp.parallel.execute(_augment_meta, list(meta.iterrows()), 1):
            meta.loc[index] = row
        meta.to_json(path, orient='index')
        logging.info('...Done')
