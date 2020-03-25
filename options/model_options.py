import dataclasses as dc
from typing import Optional, Union, List

MODELS = Union[
    'I3DModel', 'TADNModel', 'TARNModel',
    'AEI3DModel', 'AETARNModel',
    'GSNNI3DModel', 'GSNNTARNModel',
    'VAEI3DModel', 'VAETARNModel'
]


@dc.dataclass
class TADNModel:
    opts: 'TADNOptions'
    arch: str = 'tadn'
    type: str = 'class'
    size: Optional[str] = None


@dc.dataclass
class TARNModel:
    opts: 'TARNOptions'
    arch: str = 'tarn'
    type: str = 'class'
    size: Optional[str] = None


@dc.dataclass
class AETARNModel:
    opts: 'AETARNOptions'
    arch: str = 'tarn_ae'
    type: str = 'ae'
    size: Optional[str] = None


@dc.dataclass
class VAETARNModel:
    opts: 'VAETARNOptions'
    arch: str = 'tarn_vae'
    type: str = 'vae'
    size: Optional[str] = None


@dc.dataclass
class GSNNTARNModel:
    opts: 'GSNNTARNOptions'
    arch: str = 'tarn_gsnn'
    type: str = 'gsnn'
    size: Optional[str] = None


@dc.dataclass
class I3DModel:
    opts: 'I3DOptions'
    arch: str = 'i3d'
    type: str = 'class'
    size: Optional[str] = None


@dc.dataclass
class AEI3DModel:
    opts: 'AEI3DOptions'
    arch: str = 'i3d_ae'
    type: str = 'ae'
    size: Optional[str] = None


@dc.dataclass
class VAEI3DModel:
    opts: 'VAEI3DOptions'
    arch: str = 'i3d_vae'
    type: str = 'vae'
    size: Optional[str] = None


@dc.dataclass
class GSNNI3DModel:
    opts: 'GSNNI3DOptions'
    arch: str = 'i3d_gsnn'
    type: str = 'gsnn'
    size: Optional[str] = None


@dc.dataclass
class TADNOptions:
    batch_size: int
    time_steps: int
    temporal_in_planes: int
    growth_rate: int
    temporal_drop_rate: float
    classifier_drop_rate: float
    class_embed_planes: Optional[int] = None
    num_classes: Optional[int] = None


@dc.dataclass
class TARNOptions:
    batch_size: int
    time_steps: int
    spatial_encoder_planes: List[int]
    bottleneck_planes: int
    classifier_drop_rate: float
    class_embed_planes: Optional[int] = None
    num_classes: Optional[int] = None


@dc.dataclass
class AETARNOptions:
    batch_size: int
    time_steps: int
    spatial_encoder_planes: List[int]
    bottleneck_planes: int
    spatial_decoder_planes: List[int]
    classifier_drop_rate: float
    flow: bool
    class_embed_planes: Optional[int] = None
    num_classes: Optional[int] = None


@dc.dataclass
class GSNNTARNOptions:
    batch_size: int
    time_steps: int
    spatial_encoder_planes: List[int]
    bottleneck_planes: int
    classifier_drop_rate: float
    vote_type: str
    class_embed_planes: Optional[int] = None
    num_classes: Optional[int] = None


@dc.dataclass
class VAETARNOptions:
    batch_size: int
    time_steps: int
    spatial_encoder_planes: List[int]
    bottleneck_planes: int
    spatial_decoder_planes: List[int]
    classifier_drop_rate: float
    vote_type: str
    class_embed_planes: Optional[int] = None
    num_classes: Optional[int] = None


@dc.dataclass
class I3DOptions:
    batch_size: int
    time_steps: int
    dropout_prob: float = 0.0
    num_classes: Optional[int] = None


@dc.dataclass
class AEI3DOptions:
    batch_size: int
    time_steps: int
    embed_planes: int
    dropout_prob: float
    flow: bool
    num_classes: Optional[int] = None


@dc.dataclass
class GSNNI3DOptions:
    batch_size: int
    time_steps: int
    latent_planes: int
    dropout_prob: float
    vote_type: str
    num_classes: Optional[int] = None


@dc.dataclass
class VAEI3DOptions:
    batch_size: int
    time_steps: int
    latent_planes: int
    dropout_prob: float
    vote_type: str
    num_classes: Optional[int] = None


@dc.dataclass
class Unit3DOptions:
    in_channels: int
    out_channels: int
    kernel_size: List[int] = (1, 1, 1)
    stride: List[int] = (1, 1, 1)
    activation: str = 'relu'
    padding: str = 'SAME'
    use_bias: bool = False
    use_bn: bool = True
