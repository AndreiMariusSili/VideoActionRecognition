import dataclasses as dc


@dc.dataclass
class CreateDummySetOptions:
    set: str


@dc.dataclass
class PreproSetOptions:
    set: str


@dc.dataclass
class ModelRunOptions:
    spec: str
