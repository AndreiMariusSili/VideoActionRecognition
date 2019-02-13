from typing import Union, Tuple, Optional
import dataclasses as dc
__all__ = ['DataBunchOptions', 'DataSetOptions', 'DataLoaderOptions']


@dc.dataclass
class DataBunchOptions:
    shape: str
    size: Union[int, Tuple[int, int]]
    test: Optional[bool] = False


@dc.dataclass
class DataSetOptions:
    cut: Optional[float] = 1.0
    split: Optional[str] = None
    keep: Optional[int] = None


@dc.dataclass
class DataLoaderOptions:
    batch_size: Optional[int] = 1
    shuffle: Optional[bool] = False
    num_workers: Optional[int] = 0
    pin_memory: Optional[bool] = False
    drop_last: Optional[bool] = False
    timeout: Optional[int] = 0
