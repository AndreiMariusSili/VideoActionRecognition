import dataclasses as dc


@dc.dataclass
class SetupOptions:
    set: str
    local_rank: int
    resume: bool = False


@dc.dataclass
class CreateDummySetOptions:
    set: str
    split: str
    local_rank: int
    resume: bool = False


@dc.dataclass
class PreproSetOptions:
    set: str
    split: str
    local_rank: int
    jpeg: bool = False
    resume: bool = False


@dc.dataclass
class RunExperimentOptions:
    spec: str
    local_rank: int
    resume: bool = False
    overfit: bool = False
    dev: bool = False


@dc.dataclass
class EvaluateExperimentOptions:
    spec: str
    local_rank: int
    resume: bool = False
    overfit: bool = False
    dev: bool = False


@dc.dataclass
class VisualiseOptions:
    page: str
    spec: str
    local_rank: int
    resume: bool = False
    overfit: bool = False
    dev: bool = False
