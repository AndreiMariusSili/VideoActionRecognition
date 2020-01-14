import os
import pathlib as pl
from typing import List, Tuple

import pandas as pd
import skvideo.io

import env
import helpers as ghp
import prepro.helpers as php

DATA_ROOT_DIR = None


def _extract_jpeg(batch: Tuple[int, List[pd.Series]]) -> ghp.parallel.Result:
    """Create a batch of augmented rows."""
    no, batch = batch

    for index, row in batch:
        video_path: pl.Path = DATA_ROOT_DIR / row['video_path']
        image_path: pl.Path = DATA_ROOT_DIR / row['image_path']
        video = skvideo.io.vread(video_path.as_posix())
        os.makedirs(image_path.as_posix(), exist_ok=True)
        skvideo.io.vwrite((image_path / '%4d.jpeg').as_posix(), video)

    return ghp.parallel.Result(len(batch), batch)


def main(dataset: str, split: int):
    global DATA_ROOT_DIR
    DATA_ROOT_DIR = php.get_data_root_path(dataset)
    [train, dev, _, test] = php.get_meta_paths(dataset, split)

    for path in [train, dev, test]:
        env.LOGGER.info(f'Extracting jpeg images from {path.as_posix()}...')
        meta = ghp.read_meta(path)
        for _ in ghp.parallel.execute(_extract_jpeg, list(meta.iterrows()), 1):
            continue
        env.LOGGER.info('...Done')
