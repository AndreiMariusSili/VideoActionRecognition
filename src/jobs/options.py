import dataclasses as dc


@dc.dataclass
class CreateDummySetOptions:
    set: str
    local_rank: int
    sample: bool = False


@dc.dataclass
class PreproSetOptions:
    set: str
    local_rank: int


@dc.dataclass()
class ModelPrepareOptions:
    model: str
    local_rank: int


@dc.dataclass
class ModelRunOptions:
    spec: str
    local_rank: int


@dc.dataclass
class ModelVisualiseOptions:
    page: str
    spec: str
    local_rank: int


@dc.dataclass
class ModelEvaluateOptions:
    spec: str
    local_rank: int


@dc.dataclass
class I3DPrepareOptions:
    rgb: bool
    flow: bool
    batch_size: int
    local_rank: int
