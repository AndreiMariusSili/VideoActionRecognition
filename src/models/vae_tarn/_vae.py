from typing import Optional, Tuple

import torch as th
from torch import nn

from models.vae_tarn import _classifier as cls, _decoder as de, _encoder as en

VAE_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]


class VAETimeAlignedResNet(nn.Module):
    spatial: nn.Module
    temporal: nn.Module
    aggregation: nn.Module
    classifier: nn.Module

    temporal_in_planes: int
    temporal_out_planes: int
    time_steps: int
    drop_rate: float
    num_classes: int
    vote_type: str

    def __init__(self, time_steps: int, drop_rate: float, num_classes: int, vote_type: str,
                 encoder_planes: Tuple[int, int, int, int, int], decoder_planes: Tuple[int, int, int, int, int]):
        super(VAETimeAlignedResNet, self).__init__()

        self.spatial_encoder_planes = encoder_planes
        self.spatial_decoder_planes = decoder_planes

        self.temporal_in_planes = self.temporal_out_planes = encoder_planes[-1]

        self.time_steps = time_steps
        self.drop_rate = drop_rate
        self.num_classes = num_classes
        self.vote_type = vote_type

        self.spatial_encoder = en.VariationalSpatialResNetEncoder(self.spatial_encoder_planes)
        self.temporal_encoder = en.VariationalTemporalResNetEncoder(self.time_steps,
                                                                    self.temporal_in_planes,
                                                                    self.temporal_out_planes)
        self.decoder = de.VariationalSpatialResNetDecoder(self.temporal_out_planes,
                                                          self.spatial_decoder_planes)
        self.classifier = cls.VariationalTimeAlignedResNetClassifier(self.temporal_out_planes * self.time_steps,
                                                                     self.num_classes,
                                                                     self.drop_rate)

        self.softmax = nn.Softmax(dim=-1)

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

    def forward(self, _in: th.Tensor, inference: bool, num_samples: int) -> VAE_FORWARD:
        if inference:
            return self._inference(_in, num_samples)
        else:
            return self._forward(_in, num_samples)

    def _inference(self, _in: th.Tensor, num_samples: int) -> VAE_FORWARD:
        bs, t, c, h, w = _in.shape

        _spatial_embed = self.spatial_encoder(_in)
        _temporal_latent, _temporal_latents, _means, _vars = self.temporal_encoder(_spatial_embed, num_samples)
        _pred, _final_latent = self.classifier(_temporal_latents)
        _recon = self.decoder(_temporal_latents)

        num_samples = max(1, num_samples)
        if self.vote_type == 'hard':
            _vote = self._hard_vote(_pred.reshape(bs, num_samples, self.num_classes))
        else:
            _vote = self._soft_vote(_pred.reshape(bs, num_samples, self.num_classes))

        return _recon, _pred, _final_latent, _means, _vars, _vote

    def _forward(self, _in: th.Tensor, num_samples: int = 1) -> VAE_FORWARD:
        _spatial_embed = self.spatial_encoder(_in)
        _temporal_latent, _temporal_latents, _means, _vars = self.temporal_encoder(_spatial_embed, num_samples)
        _pred, _class_latent = self.classifier(_temporal_latents)
        _recon = self.decoder(_temporal_latents)

        return _recon, _pred, _class_latent, _means, _vars, None

    def _hard_vote(self, _preds: th.Tensor) -> th.Tensor:
        bs, num_samples, _ = _preds.shape
        _preds = th.argmax(_preds, dim=2, keepdim=True)
        _votes = th.zeros(bs, num_samples, self.num_classes, device=_preds.device)
        # noinspection PyTypeChecker
        _votes = _votes.scatter_(2, _preds, 1).sum(dim=1)

        return _votes

    def _soft_vote(self, _preds: th.Tensor) -> th.Tensor:
        bs, num_samples, _ = _preds.shape
        _preds = self.softmax(_preds)
        _votes = _preds.mean(dim=1)

        return _votes


if __name__ == "__main__":
    import os
    import models.helpers
    import constants as ct

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    vae = VAETimeAlignedResNet(4, 0.5, 30, 'soft',
                               encoder_planes=(16, 32, 64, 128, 256),
                               decoder_planes=(256, 128, 64, 32, 16))
    print(vae)

    print("===INPUT===")
    _in = th.randn((2, 4, 3, 224, 224), dtype=th.float)
    print(_in.shape)

    print("===FORWARD===")
    x, y, z, mu, sig, _ = vae(_in, False, 1)
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"recon":20s}:\t{x.shape}')

    print("===ML INFERENCE===")
    x, y, z, mu, sig, v = vae(_in, True, 0)
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"vote":20s}:\t{v.shape}')
    print(f'{"recon":20s}:\t{x.shape}')

    print("===VARIATIONAL INFERENCE===")
    x, y, z, mu, sig, v = vae(_in, True, ct.VAE_NUM_SAMPLES)
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"vote":20s}:\t{v.shape}')
    print(f'{"recon":20s}:\t{x.shape}')

    print(f'{models.helpers.count_parameters(vae):,}')
