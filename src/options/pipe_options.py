import dataclasses as dc
import pathlib as pl
from typing import Optional, Tuple, Union

import pipeline.transforms as pit

__all__ = ['DataBunchOptions', 'DataSetOptions', 'DataOptions', 'SamplingOptions', 'DataLoaderOptions']


@dc.dataclass
class DataBunchOptions:
    shape: str
    frame_size: Union[int, Tuple[int, int]]
    distributed: bool = False


@dc.dataclass
class DataSetOptions:
    do: 'DataOptions'
    so: 'SamplingOptions'


@dc.dataclass
class DataOptions:
    meta_path: pl.Path
    cut: float
    setting: Optional[str] = None
    transform: Optional['pit.VideoCompose'] = None
    keep: Optional[float] = None


@dc.dataclass
class SamplingOptions:
    num_segments: int
    segment_size: int


@dc.dataclass
class DataLoaderOptions:
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False
    num_workers: Optional[int] = 0
    pin_memory: Optional[bool] = False
    drop_last: Optional[bool] = False
    timeout: Optional[int] = 0
