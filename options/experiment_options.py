import dataclasses as dc
import pathlib as pl
from typing import Any, Callable, Dict, Optional, Tuple

from torch import optim as optim

from options import data_options as do, model_options as mo


@dc.dataclass
class CriterionOptions:
    pass


@dc.dataclass
class AdamOptimizerOptions:
    algorithm: Any = optim.Adam
    lr: Optional[float] = 0.001
    betas: Optional[Tuple[float, float]] = (0.9, 0.999)
    eps: Optional[float] = 1e-08
    weight_decay: Optional[float] = 0
    amsgrad: Optional[bool] = False


@dc.dataclass
class TrainerOptions:
    epochs: int
    optimizer: AdamOptimizerOptions
    criterion: Any
    criterion_opts: CriterionOptions = CriterionOptions()
    metrics: Optional[Dict[str, Any]] = None


@dc.dataclass
class EvaluatorOptions:
    metrics: Dict[str, Any]
    criterion: Optional[Callable] = None
    criterion_opts: CriterionOptions = CriterionOptions()


@dc.dataclass
class ExperimentOptions:
    name: Optional[str] = None
    resume: Optional[bool] = None
    debug: Optional[bool] = None
    run_dir: Optional[pl.Path] = None
    model_opts: mo.MODEL_OPTS = None
    databunch_opts: Optional[do.DataBunchOptions] = None
    trainer: Optional[TrainerOptions] = None
    evaluator: Optional[EvaluatorOptions] = None
