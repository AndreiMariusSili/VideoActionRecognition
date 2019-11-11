import pathlib as pl
import typing as tp
from datetime import datetime

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


def read_meta(path: tp.Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the meta json as a DataFrame."""
    path = pl.Path(path)

    return pd.read_json(path, orient='index').sort_index()


def read_lid2gid(path: pl.Path) -> pd.DataFrame:
    """Read the lid2gid json as a DataFrame."""
    return pd.read_json(path, orient='index').astype({'lid': int, 'gid': int}).sort_index()


def read_stats(path: pl.Path) -> pd.DataFrame:
    """Read in the stats json as a DataFrame."""
    df = pd.read_json(path, orient='index', typ='frame')

    return df


def read_results(path: tp.Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the results json as a DataFrame."""
    path = pl.Path(path)
    df = pd.read_pickle(path, compression=None)

    return df


def read_label2lid(path: pl.Path) -> pd.DataFrame:
    """Read in the label2id json as a DataFrame. JSON expected to be in index format."""
    df = pd.read_json(path, orient='index', typ='frame', dtype=False)

    return df


def read_smth_lid2label(path: pl.Path) -> pd.DataFrame:
    """Read in the label2lid json as a DataFrame and convert to lid2labels."""
    df = pd.read_json(path, orient='index', typ='frame', dtype=True, encoding='utf-8')
    df = df.set_index('id', drop=False, verify_integrity=True)

    return df


def read_gid2labels(path: pl.Path = ct.SMTH_GID2LABELS) -> pd.DataFrame:
    """Read in the gid2labels json as a DataFrame."""
    df = pd.read_json(path, orient='index', typ='frame', dtype=True, encoding='utf-8')
    df.index = df.index.map(lambda x: (int(x.replace('[', '').replace(']', '').split(',')[0]),
                                       int(x.replace('[', '').replace(']', '').split(',')[1])))

    df = df.rename_axis(index=['group_id', 'range'])

    return df


def read_smth_label2gid(path: pl.Path = ct.SMTH_LABEL2GID) -> pd.DataFrame:
    """Read in the gid2labels json as a DataFrame."""
    df = pd.read_json(path, orient='index', typ='frame', dtype=True, encoding='utf-8')

    return df


def read_smth_lid2gid() -> pd.DataFrame:
    """Create a lid2gid DataFrame from 2 jsons."""
    label2gid = read_smth_label2gid(ct.SMTH_LABEL2GID)
    label2gid.columns = ['gid']

    label2lid = read_label2lid(ct.SMTH_LABEL2LID)
    label2lid.columns = ['lid']

    lid2gid = label2gid.join(label2lid).set_index(['lid', 'gid'], drop=False, verify_integrity=True)

    return lid2gid


def flatten_dict(nested_dict: tp.Dict[tp.Any, tp.Any], parent_key='', sep='@'):
    """Flatten a nexted dictionary joining keys with `sep`.

    :param nested_dict: The nested dict.
    :param sep: Key separator.
    :param parent_key: Key to prepend to each dict key.
    :return: the flattened dict.
    """
    items = []
    for k, v in nested_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


def notify(_type: str, title: str, text: str, fields: tp.List[tp.Dict[str, tp.Any]] = None) -> int:
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
