from typing import Optional, Tuple

import torch as th
from torch import nn

from models import common as cm
from models.vae_i3d._classifier import I3DClassifier
from models.vae_i3d._decoder import I3DDecoder
from models.vae_i3d._encoder import I3DEncoder

VAE_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]


class VAEI3D(nn.Module):
    num_classes: int
    dropout_prob: float
    latent_size: int
    vote_type: str

    def __init__(self, latent_size: int, dropout_prob: float, num_classes: int, vote_type: str):
        super(VAEI3D, self).__init__()
        assert vote_type in ['soft', 'hard'], f'Unknown vote type: {vote_type}.'

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.latent_size = latent_size
        self.vote_type = vote_type

        self.encoder = I3DEncoder(latent_size, dropout_prob)

        self.flatten = cm.Flatten(latent_size)
        self.rsample = cm.ReparameterizedSample(latent_size)
        self.unflatten = cm.Unflatten(latent_size)

        self.decoder = I3DDecoder(latent_size)
        self.classifier = I3DClassifier(latent_size, num_classes)

        self.softmax = nn.Softmax(dim=-1)

        self._he_init()

    def _he_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                th.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, _in: th.Tensor, inference: bool, num_samples: int) -> VAE_FORWARD:
        if inference:
            return self._inference(_in, num_samples)
        else:
            return self._forward(_in, num_samples)

    def _forward(self, _in: th.Tensor, num_samples: int) -> VAE_FORWARD:
        bs = _in.shape[0]

        _mean, _log_var = self.encoder(_in)
        _mean, _log_var = self.flatten(_mean), self.flatten(_log_var)
        _latent = self.rsample(_mean, _log_var, num_samples)
        _latent = self.unflatten(_latent)
        _pred = self.classifier(_latent).view(bs, num_samples, self.num_classes)
        _recon = self.decoder(_latent)

        _latent = _latent.view(bs, num_samples, self.latent_size, 1, 1, 1)

        return _recon, _pred, _latent, _mean, _log_var, None

    def _inference(self, _in: th.Tensor, num_samples: int) -> VAE_FORWARD:
        bs = _in.shape[0]

        _mean, _log_var = self.encoder(_in)
        _mean, _log_var = self.flatten(_mean), self.flatten(_log_var)
        if num_samples:
            _latent = self.rsample(_mean, _log_var, num_samples)
            _latent = self.unflatten(_latent)
            _pred = self.classifier(_latent).view(bs, num_samples, self.num_classes)
            if self.vote_type == 'hard':
                _vote = self._hard_vote(_pred)
            else:
                _vote = self._soft_vote(_pred)
        else:
            num_samples = 1  # 0 number of samples means 1 ML prediction
            _latent = self.unflatten(_mean)
            _pred = self.classifier(_latent).view(bs, num_samples, self.num_classes)
            if self.vote_type == 'hard':
                _vote = self._hard_vote(_pred)
            else:
                _vote = self._soft_vote(_pred)

        _recon = self.decoder(_latent[::num_samples, :, :, :, :])
        _latent = _latent.view(bs, num_samples, self.latent_size, 1, 1, 1)

        return _recon, _pred, _latent, _mean, _log_var, _vote

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


if __name__ == '__main__':
    import os

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    vae = VAEI3D(3, 0.0, 10, 'soft')
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
    x, y, z, mu, sig, v = vae(_in, True, 5)
    print(f'{"mean":20s}:\t{mu.shape}')
    print(f'{"log_var":20s}:\t{sig.shape}')
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"vote":20s}:\t{v.shape}')
    print(f'{"recon":20s}:\t{x.shape}')
