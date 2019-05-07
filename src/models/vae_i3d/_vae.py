from typing import Tuple

import torch as th
from torch import nn

from . import _common as cm
from ._classifier import I3DClassifier
from ._decoder import I3DDecoder
from ._encoder import I3DEncoder

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

    def forward(self, _in: th.Tensor, decode: bool = True, ) -> VAE_FORWARD:
        _mean, _log_var = self.encoder(_in)
        _mean, _log_var = self.flatten(_mean), self.flatten(_log_var)
        _latent = self.rsample(_mean, _log_var)
        _latent = self.unflatten(_latent)
        _pred = self.classifier(_latent)
        _recon = self.decoder(_latent)

        return _recon, _pred.view(-1, self.num_classes), _latent, _mean, _log_var

    def inference(self, _in: th.Tensor, max_likelihood: bool, n_samples: int):
        _mean, _log_var = self.encoder(_in)
        _mean, _log_var = self.flatten(_mean), self.flatten(_log_var)
        if max_likelihood:
            _latent = self.unflatten(_mean)
            _pred = self.classifier(_latent).view(-1, self.num_classes)
            _recon = self.decoder(_latent)
        else:
            _latent = []
            for i in range(n_samples):
                z = self.unflatten(self.rsample(_mean, _log_var))
                _latent.append(z)
            _latent = th.cat(tuple(_latent), dim=0)
            _pred = self.classifier(_latent).view(-1, n_samples, self.num_classes)
            _recon = None

        return _recon, _pred, _latent, _mean, _log_var


if __name__ == '__main__':
    import os

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    vae = VAEI3D(1024, 0.0, 10)
    print(vae)
    _in = th.randn((1, 16, 3, 224, 224), dtype=th.float)
    x, y, z, mu, sig = vae.inference(_in)
    print(f'{"recon":20s}:\t{x.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
