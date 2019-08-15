import glob
import json
import pathlib as pl
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import requests
from torch import nn

import constants as ct


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def change_setting(path: pl.Path, _from: str, _to: str) -> pl.Path:
    """Switch between setting pointers for a path."""
    settings = ['dummy', 'full']
    assert _from in settings, f'Unknown _from: {_from}. Possible values {settings}.'
    assert _to in settings, f'Unknown _to: {_to}. Possible values {settings}.'
    assert _from != _to, f'Identical _from and _to.'

    return pl.Path(path.as_posix().replace(f'/{_from}/', f'/{_to}/'))


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
    df['template'] = df['template'].str.replace(r'\[Something\]', r'Something')

    return df


def read_smth_results(path: Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the results json as a DataFrame."""
    path = pl.Path(path)
    # with path.open('r', encoding='utf-8') as file:
    df = pd.read_pickle(path.as_posix(), compression=None)
    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(str)

    return df


def read_smth_embeddings(path: Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the results json as a DataFrame."""
    path = pl.Path(path)
    with path.open('r', encoding='utf-8') as file:
        df = pd.read_json(file, orient='records', typ='frame', dtype=False, encoding='utf-8')
    df.y = df.y.map(str)
    return df


def read_smth_stats(path: pl.Path = ct.SMTH_STATS_MERGED) -> pd.DataFrame:
    """Read in the stats json as a DataFrame."""
    with path.open('r', encoding='utf-8') as file:
        df = pd.read_json(file, orient='record', typ='frame', dtype=False)

    return df


def read_smth_label2lid(path: str = ct.SMTH_LABEL2LID) -> pd.DataFrame:
    """Read in the label2id json as a DataFrame. If json is not in record format, will convert it first."""
    try:
        df = pd.read_json(path, orient='index', typ='frame', dtype=True, encoding='utf-8')
    except ValueError:
        with open(path) as file:
            label2lid: Dict[str, str] = json.load(file)
        # noinspection PyTypeChecker
        label2lid: List[Dict[str, str]] = list(map(_create_label2lid_record, label2lid.items()))
        df = pd.read_json(json.dumps(label2lid), orient='records', typ='frame', dtype=True, encoding='utf-8')
        df = df.set_index('template', drop=False, verify_integrity=True)
        df.index = df.index.map(str)

    return df


def read_smth_lid2label(path: str = ct.SMTH_LABEL2LID) -> pd.DataFrame:
    """Read in the label2lid json as a DataFrame and convert to lid2labels."""
    try:
        df = pd.read_json(path, orient='index', typ='frame', dtype=True, encoding='utf-8')
    except ValueError:
        with open(path) as file:
            label2lid: Dict[str, str] = json.load(file)
        # noinspection PyTypeChecker
        label2lid: List[Dict[str, str]] = list(map(_create_label2lid_record, label2lid.items()))
        df = pd.read_json(json.dumps(label2lid), orient='records', typ='frame', dtype=True, encoding='utf-8')

    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(int)

    return df


def read_smth_gid2labels(path: str = ct.SMTH_GID2LABELS) -> pd.DataFrame:
    """Read in the gid2labels json as a DataFrame."""
    try:
        df = pd.read_json(path, orient='index', typ='frame', dtype=True, encoding='utf-8')
    except ValueError:
        with open(path) as file:
            gid2labels: Dict[str, str] = json.load(file)
        # noinspection PyTypeChecker
        gid2labels: List[List[Dict[str, str]]] = list(map(_create_gid2labels_record, gid2labels.items()))
        flat_gid2labels = [item for sublist in gid2labels for item in sublist]
        df = pd.read_json(json.dumps(flat_gid2labels), orient='records', typ='frame', dtype=True, encoding='utf-8')
    _range = pd.RangeIndex(0, len(df))
    df = df.set_index(['id', _range], drop=False, verify_integrity=True)
    df.index = df.index.map(lambda _tuple: (int(_tuple[0]), int(_tuple[1])))
    df = df.rename_axis(index=['group_id', 'range'])

    return df


def read_smth_label2gid(path: str = ct.SMTH_LABEL2GID) -> pd.DataFrame:
    """Read in the gid2labels json as a DataFrame."""
    try:
        df = pd.read_json(path, orient='index', typ='frame', dtype=True, encoding='utf-8')
    except ValueError:
        with open(path) as file:
            gid2labels: Dict[str, str] = json.load(file)
        # noinspection PyTypeChecker
        gid2labels: List[List[Dict[str, str]]] = list(map(_create_gid2labels_record, gid2labels.items()))
        flat_gid2labels = [item for sublist in gid2labels for item in sublist]
        df = pd.read_json(json.dumps(flat_gid2labels), orient='records', typ='frame', dtype=True, encoding='utf-8')
    df = df.set_index('template', drop=False, verify_integrity=True)
    df.index = df.index.map(str)

    return df


def read_smth_lid2gid() -> pd.DataFrame:
    label2gid = read_smth_label2gid(ct.SMTH_LABEL2GID).drop('template', axis=1)
    label2gid.columns = ['gid']

    label2lid = read_smth_label2lid(ct.SMTH_LABEL2LID).drop('template', axis=1)
    label2lid.columns = ['lid']

    lid2gid = label2gid.join(label2lid).set_index(['lid', 'gid'], drop=False, verify_integrity=True)

    return lid2gid


# TODO: Is this still needed?
def read_smth_gid2group(path: str) -> pd.DataFrame:
    """Read in the gid2group json as a DataFrame."""
    with open(path, 'r', encoding='utf-8') as file:
        group_id2group_label: Dict[str, str] = json.load(file)
    # noinspection PyTypeChecker
    group_id2group_label: List[Dict[str, str]] = list(map(_create_gid2group_record, group_id2group_label.items()))

    df = pd.read_json(json.dumps(group_id2group_label), orient='record', typ='frame', dtype=True, encoding='utf-8')
    df = df.set_index('id', drop=False, verify_integrity=True)
    df.index = df.index.map(int)

    return df


def notify(_type: str, title: str, text: str, fields: List[Dict[str, Any]] = None) -> int:
    """Send notification to slack channel."""
    if _type == 'good':
        colour = "#2E7D32"
    elif _type == 'bad':
        colour = "#C62828"
    else:
        raise ValueError(f'Unknown type: ${_type}.')

    payload = {
        'attachments': [
            {
                'fallback': f'{title}: {text}',
                'color': colour,
                'title': title,
                'text': text,
                'footer': 'Beasty',
                'ts': round(datetime.now().timestamp())
            }
        ]
    }
    if fields is not None:
        # noinspection PyTypeChecker
        payload['attachments'][0]['fields'] = fields

    response = requests.post(
        url=ct.SLACK_NOTIFICATION_URL,
        json=payload,
        headers={
            'Content-Type': 'application/json'
        }
    )

    return response.status_code


def _create_gid2labels_record(_tuple: Tuple[str, List[str]]):
    return [{'id': _tuple[0], 'template': template} for template in _tuple[1]]


def _create_label2lid_record(_tuple: Tuple[str, str]):
    return {'id': _tuple[1], 'template': _tuple[0]}


def _create_label2gid_record(_tuple: Tuple[str, str]):
    return {'group': _tuple[1], 'template': _tuple[0]}


def _create_gid2group_record(_tuple: Tuple[str, str]):
    return {'id': _tuple[0], 'label': _tuple[1]}
