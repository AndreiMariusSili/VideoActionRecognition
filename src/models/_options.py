from collections import namedtuple
from typing import Optional, Dict, Any, Union, Type, Tuple
import dataclasses as dc
from torch import optim
from torch import nn

import pipeline as pipe

__all__ = [
    'TrainerOptions', 'EvaluatorOptions', 'VideoLSTMOptions',
    'OptimizerOptions', 'AdamOptimizerOptions',
    'RunOptions'
]


@dc.dataclass
class OptimizerOptions:
    lr: Optional[float] = 0.001


@dc.dataclass
class AdamOptimizerOptions(OptimizerOptions):
    lr: Optional[float] = 0.001
    betas: Optional[Tuple[float, float]] = (0.9, 0.999)
    eps: Optional[float] = 1e-08
    weight_decay: Optional[float] = 0
    amsgrad: Optional[bool] = False


@dc.dataclass
class CriterionOptions:
    pass


@dc.dataclass
class VideoLSTMOptions:
    num_classes: int
    freeze_conv: bool
    freeze_fc: bool


@dc.dataclass
class TrainerOptions:
    epochs: int
    optimizer: Union[Type[optim.SGD], Type[optim.Adam], Type[optim.RMSprop]]
    criterion: Type[nn.CrossEntropyLoss]
    optimizer_opts: Optional[OptimizerOptions] = OptimizerOptions()
    criterion_opts: Optional[CriterionOptions] = CriterionOptions()


@dc.dataclass
class EvaluatorOptions:
    metrics: Dict[str, Any]


@dc.dataclass
class RunOptions:
    name: str
    resume: bool
    resume_from: Optional[str]
    log_interval: int
    model: Type[nn.Module]
    model_opts: VideoLSTMOptions
    data_bunch: Type[pipe.SmthDataBunch]
    data_bunch_opts: pipe.DataBunchOptions
    data_set_opts: pipe.DataSetOptions
    data_loader_opts: pipe.DataLoaderOptions
    trainer_opts: TrainerOptions
    evaluator_opts: EvaluatorOptions
