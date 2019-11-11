import dataclasses as dc
from typing import Any, Optional, Tuple, Union

MODEL_OPTS = Optional[Union[
    'I3DOptions',
    'TADNOptions',
    'TARNOptions',
    'AEI3DOptions',
    'AETARNOptions',
    'VAEI3DOptions',
    'VAETARNOptions'
]]


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
class I3DOptions:
    type = 'class'
    arch: Any
    dropout_prob: float = 0.0
    num_classes: Optional[int] = None


@dc.dataclass
class AEI3DOptions:
    type = 'ae'
    arch: Any
    embed_planes: int
    dropout_prob: float
    num_classes: Optional[int] = None


@dc.dataclass
class VAEI3DOptions:
    type = 'vae'
    arch: Any
    latent_planes: int
    dropout_prob: float
    vote_type: str
    num_classes: Optional[int] = None


@dc.dataclass
class TARNOptions:
    type = 'class'
    arch: Any
    time_steps: int
    classifier_drop_rate: float
    temporal_out_planes: int
    class_embed_planes: int
    encoder_planes: Tuple[int, ...]
    num_classes: Optional[int] = None


@dc.dataclass
class TADNOptions:
    type = 'class'
    arch: Any
    time_steps: int
    temporal_in_planes: int
    growth_rate: int
    temporal_drop_rate: float
    classifier_drop_rate: float
    class_embed_planes: int
    num_classes: Optional[int] = None


@dc.dataclass
class AETARNOptions:
    type = 'ae'
    arch: Any
    time_steps: int
    classifier_drop_rate: float
    temporal_out_planes: int
    class_embed_planes: int
    encoder_planes: Tuple[int, ...]
    decoder_planes: Tuple[int, ...]
    num_classes: Optional[int] = None


@dc.dataclass
class VAETARNOptions:
    type = 'vae'
    arch: Any
    time_steps: int
    classifier_drop_rate: float
    temporal_out_planes: int
    class_embed_planes: int
    encoder_planes: Tuple[int, ...]
    decoder_planes: Tuple[int, ...]
    vote_type: str
    num_classes: Optional[int] = None
