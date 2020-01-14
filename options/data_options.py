import dataclasses as dc
import pathlib as pl
from typing import Optional, Tuple, Union

import databunch.transforms as dt


@dc.dataclass
class SamplingOptions:
    num_segments: int
    segment_size: int


@dc.dataclass
class DataSetOptions:
    root_path: pl.Path
    meta_path: pl.Path
    read_jpeg: bool
    setting: str
    transform: Optional[dt.VideoCompose] = None
    keep: Union[float, int, None] = None


@dc.dataclass
class DataLoaderOptions:
    batch_size: Optional[int] = None
    shuffle: Optional[bool] = False
    num_workers: Optional[int] = 0
    pin_memory: Optional[bool] = False
    drop_last: Optional[bool] = False
    timeout: Optional[int] = 0


@dc.dataclass
class DataBunch:
    shape: str
    cut: Optional[float]
    frame_size: Union[int, Tuple[int, int]]
    stats_path: pl.Path
    distributed: bool = False
    dlo: Optional[DataLoaderOptions] = None
    so: Optional[SamplingOptions] = None
    train_dso: Optional[DataSetOptions] = None
    dev_dso: Optional[DataSetOptions] = None
    test_dso: Optional[DataSetOptions] = None
