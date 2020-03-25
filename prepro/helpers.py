import pathlib as pl
from typing import List

import constants as ct


def get_data_root_path(dataset: str) -> pl.Path:
    """Get the path to the root data directory on the dataset."""
    if dataset == 'smth':
        root_path = getattr(ct, f'SMTH_ROOT_DIR')
    elif dataset == 'hmdb':
        root_path = getattr(ct, f'HMDB_ROOT_DIR')
    else:
        raise ValueError(f'Unknown dataset {dataset}.')

    return ct.WORK_ROOT / root_path


def get_stats_path(dataset: str, split: int) -> pl.Path:
    """Get the path to the stats file based on the dataset."""
    if dataset == 'smth':
        stats_path = getattr(ct, f'SMTH_STATS_MERGED_{split}')
    elif dataset == 'hmdb':
        stats_path = getattr(ct, f'HMDB_STATS_MERGED_{split}')
    else:
        raise ValueError(f'Unknown dataset {dataset}.')

    return ct.WORK_ROOT / stats_path


def get_video_path(dataset: str):
    """Get the path to the video folder based on the dataset."""
    if dataset == 'smth':
        video_path = ct.SMTH_WEBM_DIR
    elif dataset == 'hmdb':
        video_path = ct.HMDB_AVI_DIR
    else:
        raise ValueError(f'Unknown dataset {dataset}.')

    return video_path


def get_image_path(dataset: str):
    """Get the path to the image folder based on the dataset."""
    if dataset == 'smth':
        image_path = ct.SMTH_JPEG_DIR
    elif dataset == 'hmdb':
        image_path = ct.HMDB_JPEG_DIR
    else:
        raise ValueError(f'Unknown dataset {dataset}.')

    return image_path


def get_video_paths(dataset: str) -> List[pl.Path]:
    """Get a list of the paths to all raw videos based on the dataset."""
    if dataset == 'smth':
        paths = list(ct.SMTH_WEBM_DIR.glob('*/*.webm'))
    elif dataset == 'hmdb':
        paths = list(ct.HMDB_AVI_DIR.glob('*/*.avi'))
    else:
        raise ValueError(f'Unknown dataset {dataset}.')

    return paths


def get_meta_paths(dataset: str, split: int) -> List[List[pl.Path]]:
    """Get a list of the paths to meta files based on the dataset."""
    if dataset == 'smth':
        paths = [
            [ct.SMTH_META_TRAIN_1, ct.SMTH_META_DEV_1, ct.SMTH_META_MERGED_1, ct.SMTH_META_VALID_1],
        ]
    elif dataset == 'hmdb':
        paths = [
            [ct.HMDB_META_TRAIN_1, ct.HMDB_META_DEV_1, ct.HMDB_META_MERGED_1, ct.HMDB_META_TEST_1],
            [ct.HMDB_META_TRAIN_2, ct.HMDB_META_DEV_2, ct.HMDB_META_MERGED_2, ct.HMDB_META_TEST_2],
            [ct.HMDB_META_TRAIN_3, ct.HMDB_META_DEV_3, ct.HMDB_META_MERGED_3, ct.HMDB_META_TEST_3],
        ]
    else:
        raise ValueError(f'Unknown dataset {dataset}.')

    return paths.pop(split - 1)
