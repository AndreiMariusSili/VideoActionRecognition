import dataclasses as dc
import pathlib as pl
from typing import Optional, List, Union

from options import data_options as do, model_options as mo

Models = Union[
    mo.I3DModel, mo.TADNModel, mo.TARNModel,
    mo.AEI3DModel, mo.AETARNModel,
    mo.GSNNI3DModel, mo.GSNNTARNModel,
    mo.VAEI3DModel, mo.VAETARNModel
]


@dc.dataclass
class AdamOptimizerOptions:
    lr: Optional[float] = 0.001


@dc.dataclass
class TrainerOptions:
    epochs: int
    lr_milestones: List[int]
    lr_gamma: float
    kld_warmup_epochs: Optional[int]
    optim_opts: AdamOptimizerOptions
    criterion: str
    metrics: str


@dc.dataclass
class EvaluatorOptions:
    metrics: str


@dc.dataclass
class ExperimentOptions:
    name: Optional[str] = None
    resume: Optional[bool] = None
    overfit: Optional[bool] = None
    dev: Optional[bool] = None
    debug: Optional[bool] = None
    run_dir: Optional[pl.Path] = None
    model: Optional[Models] = None
    databunch: Optional[do.DataBunch] = None
    trainer: Optional[TrainerOptions] = None
    evaluator: Optional[EvaluatorOptions] = None
    world_size: Optional[int] = None
    distributed: Optional[bool] = None
