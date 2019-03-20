from typing import Dict, List, Union
import pathlib as pl
import pandas as pd
import glob
import json

import constants as ct


def change_setting(path: pl.Path, _from: str, _to: str) -> str:
    """Switch between setting pointers for a path."""
    settings = ['dummy', 'full']
    assert _from in settings, f'Unknown _from: {_from}. Possible values {settings}.'
    assert _to in settings, f'Unknown _to: {_to}. Possible values {settings}.'
    assert _from != _to, f'Identical _from and _to.'

    return path.as_posix().replace(f'/{_from}/', f'/{_to}/')


def get_smth_videos() -> List[str]:
    """Get a list of the paths to all raw videos."""
    return glob.glob((ct.SMTH_WEBM_DIR / '*.webm').as_posix())


def read_smth_meta(path: Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the split labels json as a DataFrame."""
    path = pl.Path(path)
    with path.open('r', encoding='utf-8') as file:
        df = pd.read_json(file, orient='record', typ='frame', dtype=False, encoding='utf-8')
    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(str)
    df['template'] = df['template'].str.replace(r'\[something\]', r'something')

    return df


def read_smth_results(path: Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the results json as a DataFrame."""
    path = pl.Path(path)
    with path.open('r', encoding='utf-8') as file:
        df = pd.read_json(file, orient='record', typ='frame', dtype=False, encoding='utf-8')
    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(str)

    return df


def read_smth_labels2id(path: str) -> pd.DataFrame:
    """Read in the label2id json as a DataFrame."""
    with open(path) as file:
        templates2id: Dict[str, str] = json.load(file)
    templates2id: List[Dict[str, str]] = list(map(_create_template2id_record, templates2id.items()))

    df = pd.read_json(json.dumps(templates2id), orient='record', typ='frame', dtype=True, encoding='utf-8')
    df = df.set_index('template', drop=False, verify_integrity=True)
    df.index = df.index.map(str)

    return df


def read_smth_id2labels(path: str) -> pd.DataFrame:
    """Read in the label2id json as a DataFrame and convert to id2labels."""
    with open(path, 'r', encoding='utf-8') as file:
        templates2id: Dict[str, str] = json.load(file)
    templates2id: List[Dict[str, str]] = list(map(_create_template2id_record, templates2id.items()))

    df = pd.read_json(json.dumps(templates2id), orient='record', typ='frame', dtype=True, encoding='utf-8')
    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(int)

    return df


def read_smth_label2group(path: str) -> pd.DataFrame:
    """Read in the label2group json as a DataFrame."""
    with open(path, 'r', encoding='utf-8') as file:
        template2group: Dict[str, str] = json.load(file)
    template2group: List[Dict[str, str]] = list(map(_create_template2group_record, template2group.items()))

    df = pd.read_json(json.dumps(template2group), orient='record', typ='frame', dtype=True, encoding='utf-8')
    df = df.set_index('template', drop=False, verify_integrity=True)
    df.index = df.index.map(str)
    df['group'] = df['group'].map(int)

    return df


def read_smth_group2group_label(path: str) -> pd.DataFrame:
    """Read in the group2id json as a DataFrame."""
    with open(path, 'r', encoding='utf-8') as file:
        group_id2group_label: Dict[str, str] = json.load(file)
    group_id2group_label: List[Dict[str, str]] = list(
        map(_create_group_id2group_label_record, group_id2group_label.items()))

    df = pd.read_json(json.dumps(group_id2group_label), orient='record', typ='frame', dtype=True, encoding='utf-8')
    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(int)

    return df


def read_smth_stats() -> pd.DataFrame:
    """Read in the stats json as a DataFrame."""
    with ct.SMTH_STATS_MERGED.open('r', encoding='utf-8') as file:
        df = pd.read_json(file, orient='record', typ='frame', dtype=False)

    return df


def _create_template2id_record(_tuple: (str, str)):
    return {'id': _tuple[1], 'template': _tuple[0]}


def _create_template2group_record(_tuple: (str, str)):
    return {'group': _tuple[1], 'template': _tuple[0]}


def _create_group_id2group_label_record(_tuple: (str, str)):
    return {'id': _tuple[0], 'label': _tuple[1]}
