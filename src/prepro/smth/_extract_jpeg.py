import os
from typing import Tuple, List
from glob import glob
import skvideo.io

from env import logging
import constants as ct
import helpers as hp


def _extract_jpeg(batch: Tuple[int, List[str]]) -> hp.parallel.Result:
    """Create a batch of augmented rows."""
    no, batch = batch

    for webm_path in batch:
        video = skvideo.io.vread(webm_path)

        jpeg_path = ct.SMTH_JPEG_DIR / webm_path.replace('.webm', '').split('/').pop()
        os.makedirs(jpeg_path)
        skvideo.io.vwrite(f'{jpeg_path}/%03d.jpeg', video, outputdict={
            '-q:v': '5'
        })

    return hp.parallel.Result(len(batch), batch)


def main():
    logging.info(f'Extracting jpeg images for {ct.SETTING} set...')
    webm_paths = sorted(glob((ct.SMTH_WEBM_DIR / '*.webm').as_posix()))
    for _ in hp.parallel.execute(_extract_jpeg, webm_paths, 1):
        continue
    logging.info('...Done')
