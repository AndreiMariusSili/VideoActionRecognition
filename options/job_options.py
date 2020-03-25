import argparse as ap
import dataclasses as dc
import typing as t


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', ""):
        return False
    else:
        raise ap.ArgumentTypeError('Boolean value expected.')


@dc.dataclass
class SetupOptions:
    set: str
    resume: str2bool = False


@dc.dataclass
class SelectSubsetOptions:
    set: str
    num_classes: int


@dc.dataclass
class PreproSetOptions:
    set: str
    split: str
    jpeg: str2bool = False


@dc.dataclass
class RunExperimentOptions:
    dataset: str
    cut: str
    frames: str
    model: str
    resume: str2bool = False
    overfit: str2bool = False
    dev: str2bool = False


@dc.dataclass
class EvaluateExperimentOptions:
    dataset: str
    cut: str
    frames: str
    model: str
    resume: str2bool = False
    overfit: str2bool = False
    dev: str2bool = False


@dc.dataclass
class VisualiseExperimentOptions:
    dataset: str
    cut: str
    frames: str
    model: str


EXPERIMENT_JOB_OPTIONS = t.Union[RunExperimentOptions, EvaluateExperimentOptions, VisualiseExperimentOptions]
