from typing import Dict, List, Union, Tuple
import pathlib as pl
import pandas as pd
import numpy as np
import glob
import json

import constants as ct


def pad(video: np.ndarray, max_width, max_height) -> np.ndarray:
    length, height, width, channels = video.shape
    left, top, right, bottom = padding_values(width, height, max_width, max_height)
    return np.pad(video, ((0, 0), (top, bottom), (left, right), (0, 0)), mode='constant')


def padding_values(width: int, height: int, max_width: int, max_height: int) -> Tuple[int, int, int, int]:
    left_pad = (max_width - width) // 2
    right_pad = (max_width - width) // 2 + (max_width - width) % 2

    top_pad = (max_height - height) // 2
    bottom_pad = (max_height - height) // 2 + (max_height - height) % 2

    return left_pad, top_pad, right_pad, bottom_pad


def change_setting(path: pl.Path, _from: str, _to: str) -> str:
    """Switch between setting pointers for a path."""
    settings = ['dummy', 'full']
    assert _from in settings, f'Unknown _from: {_from}. Possible values {settings}.'
    assert _to in settings, f'Unknown _to: {_to}. Possible values {settings}.'
    assert _from != _to, f'Identical _from and _to.'

    return path.as_posix().replace(f'/{_from}/', f'/{_to}/')


def get_smth_videos() -> List[str]:
    """Get a list of the paths to all raw videos."""
    return glob.glob((ct.SMTH_VIDEO_DIR / '*.webm').as_posix())


def read_smth_meta(path: Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the split labels json as a DataFrame"""
    path = pl.Path(path)
    with path.open('r') as file:
        df = pd.read_json(file, orient='record', typ='frame', dtype=False)
    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(str)
    df['template'] = df['template'].str.replace(r'\[something\]', r'something')

    return df


def read_smth_labels2id(path: str) -> pd.DataFrame:
    """Read in the label2id json as a DataFrame"""
    with open(path) as file:
        templates2id: Dict[str, str] = json.load(file)
    templates2id: List[Dict[str, str]] = list(map(_create_template2id_record, templates2id.items()))

    df = pd.read_json(json.dumps(templates2id), orient='record', typ='frame', dtype=True)
    df = df.set_index('template', drop=False, verify_integrity=True)
    df.index = df.index.map(str)

    return df


def read_smth_stats() -> pd.DataFrame:
    """Read in the stats json as a DataFrame"""
    with ct.SMTH_STATS_MERGED.open('r') as file:
        df = pd.read_json(file, orient='record', typ='frame', dtype=False)

    return df


def _create_template2id_record(_tuple: (str, str)):
    return {'id': _tuple[1], 'template': _tuple[0]}
