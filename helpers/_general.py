import json
import pathlib as pl
import typing as tp

import dacite as da
import pandas as pd
from torch import nn

import constants as ct
import specs
from options import experiment_options as eo, model_options as mo, data_options as do, job_options as jo


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def read_meta(path: tp.Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the meta json as a DataFrame."""
    path = ct.WORK_ROOT / pl.Path(path)

    return pd.read_json(path, orient='index').sort_index()


def read_stats(path: pl.Path) -> pd.DataFrame:
    """Read in the stats json as a DataFrame."""
    path = ct.WORK_ROOT / pl.Path(path)
    df = pd.read_json(path, orient='index', typ='frame')

    return df


def read_results(path: tp.Union[pl.Path, str]) -> pd.DataFrame:
    """Read in the results json as a DataFrame."""
    path = ct.WORK_ROOT / pl.Path(path)
    df = pd.read_pickle(path, compression=None)

    return df


def read_lid2gid(path: pl.Path) -> pd.DataFrame:
    """Read the lid2gid json as a DataFrame."""
    return pd.read_json(path, orient='index').astype({'lid': int, 'gid': int}).sort_index()


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


def flatten_dict(nested_dict: tp.Dict[tp.Any, tp.Any], parent_key='', sep='@') -> tp.Dict[tp.Any, str]:
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


def path_to_string(nested_dict: tp.Dict[tp.Any, tp.Any]):
    """Convert all paths into string in a dictionary.

    :param nested_dict: A possibly nested dictionary/
    :return:
    """
    for k, v in nested_dict.items():
        if isinstance(v, dict):
            path_to_string(nested_dict[k])
        elif isinstance(v, pl.Path):
            nested_dict[k] = str(v)


def build_spec(opts: jo.EXPERIMENT_JOB_OPTIONS) -> eo.ExperimentOptions:
    """Build the spec from the spec name provided."""

    name = _build_name(opts)
    model: mo.MODELS = getattr(specs.models, f'{opts.model}_{opts.frames}')
    dbo_name = f'dbo_{opts.frames}_flow' if '_flow' in opts.model else f'dbo_{opts.frames}'
    databunch: do.DataBunch = getattr(getattr(specs.datasets, opts.dataset), dbo_name)

    if opts.cut == '4q':
        databunch.cut = 1.00
    elif opts.cut == '3q':
        databunch.cut = 0.75
    elif opts.cut == '2q':
        databunch.cut = 0.50
    else:
        raise ValueError(f'Unknown cut: {opts.cut}.')

    if model.type == 'class':
        trainer = specs.trainers.class_trainer
        evaluator = specs.evaluators.class_evaluator
    elif model.type == 'ae':
        trainer = specs.trainers.class_ae_trainer
        evaluator = specs.evaluators.class_ae_evaluator
    elif model.type == 'gsnn':
        trainer = specs.trainers.class_gsnn_trainer
        evaluator = specs.evaluators.class_gsnn_evaluator
    elif model.type == 'vae':
        trainer = specs.trainers.class_vae_trainer
        evaluator = specs.evaluators.class_vae_evaluator
    else:
        raise ValueError(f'Unknown model type: {model.type}.')

    return eo.ExperimentOptions(
        name=name,
        resume=opts.resume,
        overfit=opts.overfit,
        dev=opts.dev,
        model=model,
        databunch=databunch,
        trainer=trainer,
        evaluator=evaluator
    )


def load_spec(opts: jo.EXPERIMENT_JOB_OPTIONS) -> eo.ExperimentOptions:
    name = _build_name(opts)
    with open(str(ct.WORK_ROOT / ct.RUNS_ROOT / name / 'run.json'), 'r') as file:
        spec = json.load(file)

    return da.from_dict(data_class=eo.ExperimentOptions, data=spec, config=da.Config(cast=[pl.Path], strict=True))


def _build_name(opts: jo.EXPERIMENT_JOB_OPTIONS) -> str:
    return f'{opts.dataset}/{opts.cut}/{opts.frames}/{opts.model}'
