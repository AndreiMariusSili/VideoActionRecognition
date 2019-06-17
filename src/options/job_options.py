import dataclasses as dc


@dc.dataclass
class CreateDummySetOptions:
    set: str
    local_rank: int
    resume: bool = False


@dc.dataclass
class PreproSetOptions:
    set: str
    local_rank: int
    resume: bool = False


@dc.dataclass()
class ModelPrepareOptions:
    model: str
    local_rank: int
    resume: bool = False


@dc.dataclass
class ModelRunOptions:
    spec: str
    local_rank: int = -1
    resume: bool = False


@dc.dataclass
class ModelVisualiseOptions:
    page: str
    spec: str
    local_rank: int
    resume: bool = False


@dc.dataclass
class ModelEvaluateOptions:
    spec: str
    local_rank: int
    resume: bool = False
