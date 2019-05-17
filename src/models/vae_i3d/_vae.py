from typing import Tuple

import torch as th
from torch import nn

from models.vae_i3d import _common as cm
from models.vae_i3d._classifier import I3DClassifier
from models.vae_i3d._decoder import I3DDecoder
from models.vae_i3d._encoder import I3DEncoder

VAE_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]


class VAEI3D(nn.Module):
    num_classes: int
    dropout_prob: float
    latent_size: int

    def __init__(self, latent_size: int, dropout_prob: float, num_classes: int):
        super(VAEI3D, self).__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.latent_size = latent_size

        self.encoder = I3DEncoder(latent_size, dropout_prob)

        self.flatten = cm.Flatten(latent_size)
        self.rsample = cm.ReparameterizedSample(latent_size)
        self.unflatten = cm.Unflatten(latent_size)

        self.decoder = I3DDecoder(latent_size)
        self.classifier = I3DClassifier(latent_size, num_classes)

    def forward(self, _in: th.Tensor, inference: bool, max_likelihood: bool, num_samples: int) -> VAE_FORWARD:
        if inference:
            return self._inference(_in, max_likelihood, num_samples)
        else:
            return self._forward(_in)

    def _forward(self, _in: th.Tensor):
        bs = _in.shape[0]

        _mean, _log_var = self.encoder(_in)
        _mean, _log_var = self.flatten(_mean), self.flatten(_log_var)
        _latent = self.rsample(_mean, _log_var, 1)
        _latent = self.unflatten(_latent)
        _pred = self.classifier(_latent).view(bs, self.num_classes)
        _recon = self.decoder(_latent)

        return _recon, _pred, _latent, _mean, _log_var

    def _inference(self, _in: th.Tensor, max_likelihood: bool, num_samples: int):
        bs = _in.shape[0]

        _mean, _log_var = self.encoder(_in)
        _mean, _log_var = self.flatten(_mean), self.flatten(_log_var)
        if max_likelihood:
            _latent = self.unflatten(_mean)
            _pred = self.classifier(_latent).view(bs, self.num_classes)
            _recon = self.decoder(_latent)
        else:
            _latent = self.rsample(_mean, _log_var, num_samples)
            _latent = self.unflatten(_latent)
            _pred = self.classifier(_latent).view(bs * num_samples, self.num_classes)
            _recon = None

        return _recon, _pred, _latent, _mean, _log_var


if __name__ == '__main__':
    import os

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    vae = VAEI3D(3, 0.0, 10)
    print(vae)
    _in = th.randn((2, 4, 3, 224, 224), dtype=th.float)
    x, y, z, mu, sig = vae(_in, False, False, 0)
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"recon":20s}:\t{x.shape}')

    x, y, z, mu, sig = vae(_in, True, True, 0)
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"recon":20s}:\t{x.shape}')

    x, y, z, mu, sig = vae(_in, True, False, 3)
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
