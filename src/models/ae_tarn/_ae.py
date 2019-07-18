from typing import Tuple

import torch as th
from torch import nn

from models.ae_tarn import _classifier as cls, _decoder as de, _encoder as en


class AETimeAlignedResNet(nn.Module):
    spatial: nn.Module
    temporal: nn.Module
    aggregation: nn.Module
    classifier: nn.Module

    in_planes: int
    out_planes: int
    time_steps: int
    drop_rate: float
    num_classes: int

    def __init__(self, time_steps: int, drop_rate: float, num_classes: int,
                 encoder_planes: Tuple[int, int, int, int, int], decoder_planes: Tuple[int, int, int, int, int]):
        super(AETimeAlignedResNet, self).__init__()

        self.spatial_encoder_planes = encoder_planes
        self.spatial_decoder_planes = decoder_planes

        self.temporal_in_planes = self.temporal_out_planes = encoder_planes[-1]

        self.time_steps = time_steps
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        self.spatial_encoder = en.SpatialResNetEncoder(self.spatial_encoder_planes)
        self.temporal_encoder = en.TemporalResNetEncoder(self.time_steps,
                                                         self.temporal_in_planes,
                                                         self.temporal_out_planes)
        self.decoder = de.SpatialResNetDecoder(self.temporal_out_planes, self.spatial_decoder_planes)
        self.classifier = cls.TimeAlignedResNetClassifier(self.temporal_out_planes * self.time_steps,
                                                          self.num_classes,
                                                          self.drop_rate)

        self._he_init()

    def _he_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                th.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, _in: th.Tensor, inference: bool) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if inference:
            return self._inference(_in)
        else:
            return self._forward(_in)

    def _inference(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        _spatial_out = self.spatial_encoder(_in)
        _temporal_out, _temporal_outs = self.temporal_encoder(_spatial_out)
        _out, _embed = self.classifier(_temporal_outs)
        _recon = self.decoder(_temporal_outs)

        return _out, _embed, _recon

    def _forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        _spatial_out = self.spatial_encoder(_in)
        _temporal_out, _temporal_outs = self.temporal_encoder(_spatial_out)
        _out, _embed = self.classifier(_temporal_outs)
        _recon = self.decoder(_temporal_outs)

        return _out, _embed, _recon


if __name__ == "__main__":
    import os
    import models.helpers

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    ae = AETimeAlignedResNet(4, 0.5, 30,
                             encoder_planes=(16, 32, 64, 128, 256),
                             decoder_planes=(256, 128, 64, 32, 16))
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
