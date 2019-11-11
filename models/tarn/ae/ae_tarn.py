from typing import Tuple

import torch as th
from torch import nn

from models.tarn.common import classifier as cls, spatial_decoder as sd, spatial_encoder as se, temporal_encoder as te


class AETimeAlignedResNet(nn.Module):
    spatial: nn.Module
    temporal: nn.Module
    aggregation: nn.Module
    classifier: nn.Module

    temporal_in_planes: int
    temporal_out_planes: int
    time_steps: int
    drop_rate: float
    num_classes: int

    def __init__(self, time_steps: int, classifier_drop_rate: float, num_classes: int,
                 encoder_planes: Tuple[int, ...],
                 temporal_out_planes: int, class_embed_planes: int, decoder_planes: Tuple[int, ...]):
        super(AETimeAlignedResNet, self).__init__()

        self.spatial_encoder_planes = encoder_planes
        self.temporal_in_planes = encoder_planes[-1]
        self.temporal_out_planes = temporal_out_planes
        self.spatial_decoder_planes = decoder_planes

        self.time_steps = time_steps
        self.drop_rate = classifier_drop_rate
        self.num_classes = num_classes
        self.class_embed_planes = class_embed_planes

        self.class_in_planes = self.temporal_out_planes * self.time_steps

        self.spatial_encoder = se.SpatialResNetEncoder(self.spatial_encoder_planes)
        self.temporal_encoder = te.TemporalResNetEncoder(self.time_steps,
                                                         self.temporal_in_planes,
                                                         self.temporal_out_planes)
        self.decoder = sd.SpatialResNetDecoder(self.temporal_out_planes, self.spatial_decoder_planes)
        self.classifier = cls.TimeAlignedResNetClassifier(self.class_in_planes,
                                                          self.class_embed_planes,
                                                          self.drop_rate,
                                                          self.num_classes)

        models.helpers.he_init(self)

    def forward(self, _in: th.Tensor, inference: bool) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if inference:
            return self._inference(_in)
        else:
            return self._forward(_in)

    def _inference(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        _spatial_out = self.spatial_encoder(_in)
        _temporal_out, _temporal_outs = self.temporal_encoder(_spatial_out)
        _pred, _embed = self.classifier(_temporal_outs)
        _recon = self.decoder(_temporal_outs)

        return _pred, _embed, _recon

    def _forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        _spatial_out = self.spatial_encoder(_in)
        _temporal_out, _temporal_outs = self.temporal_encoder(_spatial_out)
        _pred, _embed = self.classifier(_temporal_outs)
        _recon = self.decoder(_temporal_outs)

        return _pred, _embed, _recon


if __name__ == "__main__":
    import os
    import models.helpers

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    ae = AETimeAlignedResNet(4, 0.5, 30, (16, 32, 64, 128, 256), 128, 512, (256, 128, 64, 32, 16))
    print(ae)

    print("===INPUT===")
    _in = th.randn((2, 4, 3, 224, 224), dtype=th.float)
    print(_in.shape)

    print("===FORWARD===")
    y, z, x = ae(_in, False)
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"recon":20s}:\t{x.shape}')

    print("===INFERENCE===")
    y, z, _ = ae(_in, True)
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"recon":20s}:\t{_.shape}')

    print(f'{models.helpers.count_parameters(ae):,}')
    print(f'{models.helpers.count_parameters(ae.spatial_encoder):,}')
    print(f'{models.helpers.count_parameters(ae.temporal_encoder):,}')
    print(f'{models.helpers.count_parameters(ae.decoder):,}')
    print(f'{models.helpers.count_parameters(ae.classifier):,}')
