from typing import Any, Callable, Dict, Optional, Tuple, Union

import dataclasses as dc

import options.pipe_options as pio

__all__ = [
    'TrainerOptions', 'EvaluatorOptions', 'LRCNOptions',
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
class VAECriterionOptions(CriterionOptions):
    mse_factor: float
    ce_factor: float
    kld_factor: float


@dc.dataclass
class AECriterionOptions(CriterionOptions):
    mse_factor: float
    ce_factor: float


@dc.dataclass
class LRCNOptions:
    num_classes: int
    freeze_features: bool
    freeze_fusion: bool


@dc.dataclass
class Unit3DOptions:
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int, int] = (1, 1, 1)
    stride: Tuple[int, int, int] = (1, 1, 1)
    activation: str = 'relu'
    padding: str = 'SAME'
    use_bias: bool = False
    use_bn: bool = True


@dc.dataclass
class VAEI3DOptions:
    latent_size: int
    dropout_prob: float
    num_classes: int
    vote_type: str


@dc.dataclass
class AEI3DOptions:
    embed_size: int
    dropout_prob: float
    num_classes: int


@dc.dataclass
class I3DOptions:
    num_classes: int
    dropout_prob: float = 0.0
    name: str = 'inception'


@dc.dataclass
class TADNOptions:
    num_classes: int
    time_steps: int
    growth_rate: int
    drop_rate: float


@dc.dataclass
class TARNOptions:
    num_classes: int
    time_steps: int
    drop_rate: float


@dc.dataclass
class TrainerOptions:
    epochs: int
    optimizer: Any
    criterion: Any
    optimizer_opts: Optional[OptimizerOptions] = OptimizerOptions()
    criterion_opts: Optional[CriterionOptions] = CriterionOptions()
    metrics: Optional[Dict[str, Any]] = None


@dc.dataclass
class EvaluatorOptions:
    metrics: Dict[str, Any]
    criterion: Optional[Callable] = None
    criterion_opts: Optional[CriterionOptions] = CriterionOptions()


@dc.dataclass
class RunOptions:
    name: str
    mode: str
    resume: bool
    log_interval: int
    patience: int
    model: Any
    model_opts: Union[LRCNOptions, I3DOptions, TADNOptions, TARNOptions, AEI3DOptions, VAEI3DOptions]
    data_bunch: Any
    db_opts: pio.DataBunchOptions
    train_ds_opts: pio.DataSetOptions
    dev_ds_opts: pio.DataSetOptions
    valid_ds_opts: pio.DataSetOptions
    train_dl_opts: pio.DataLoaderOptions
    dev_dl_opts: pio.DataLoaderOptions
    valid_dl_opts: pio.DataLoaderOptions
    trainer_opts: TrainerOptions
    evaluator_opts: EvaluatorOptions
