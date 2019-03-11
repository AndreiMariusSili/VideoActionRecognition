import dataclasses as dc


@dc.dataclass
class CreateDummySetOptions:
    set: str


@dc.dataclass
class PreproSetOptions:
    set: str


@dc.dataclass()
class ModelPrepareOptions:
    model: str


@dc.dataclass
class ModelRunOptions:
    spec: str


@dc.dataclass
class ModelVisualiseOptions:
    page: str
    run_dir: str
    spec: str


@dc.dataclass
class ModelEvaluateOptions:
    run_dir: str
    spec: str


@dc.dataclass
class I3DPrepareOptions(ModelPrepareOptions):
    rgb: bool
    flow: bool
    batch_size: int = 2
