import typing as tp

import torch as th
from torch import nn

from models.i3d.vae._classifier import I3DClassifier
from models.i3d.vae._decoder import I3DDecoder
from models.i3d.vae._encoder import I3DEncoder

VAE_FORWARD = tp.Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, tp.Optional[th.Tensor]]


class VAEI3D(nn.Module):
    NAME = 'VAEI3D'

    def __init__(self, time_steps: int, latent_planes: int, dropout_prob: float, num_classes: int, vote_type: str):
        super(VAEI3D, self).__init__()
        assert vote_type in ['soft', 'hard'], f'Unknown vote type: {vote_type}.'

        self.time_steps = time_steps
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.latent_planes = latent_planes
        self.vote_type = vote_type

        self.encoder = I3DEncoder(self.latent_planes)
        self.decoder = I3DDecoder(self.latent_planes, self.time_steps)
        self.classifier = I3DClassifier(self.latent_planes, self.dropout_prob, self.num_classes)

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

    def forward(self, _in: th.Tensor, num_samples: int) -> VAE_FORWARD:
        if self.training:
            return self._forward(_in, num_samples)
        else:
            return self._inference(_in, num_samples)

    def _forward(self, _in: th.Tensor, num_samples: int) -> VAE_FORWARD:

        temporal_latents, _mean, _var = self.encoder(_in, num_samples)
        _recon = self.decoder(temporal_latents)
        _pred, _class_latent = self.classifier(temporal_latents)

        return _recon, _pred, temporal_latents, _class_latent, _mean, _var, None

    def _inference(self, _in: th.Tensor, num_samples: int) -> VAE_FORWARD:
        _temporal_latents, _mean, _var = self.encoder(_in, num_samples)
        _recon = self.decoder(_temporal_latents)
        _pred, _class_latent = self.classifier(_temporal_latents)

        if self.vote_type == 'hard':
            _vote = self._hard_vote(_pred)
        else:
            _vote = self._soft_vote(_pred)

        return _recon, _pred, _temporal_latents, _class_latent, _mean, _var, _vote

    def _hard_vote(self, _preds: th.Tensor) -> th.Tensor:
        bs, num_samples, _ = _preds.shape
        _preds = th.argmax(_preds, dim=2, keepdim=True)
        _votes = th.zeros(bs, num_samples, self.num_classes, device=_preds.device)
        _votes = _votes.scatter_(2, _preds, 1).sum(dim=1)

        return _votes

    def _soft_vote(self, _preds: th.Tensor) -> th.Tensor:
        bs, num_samples, _ = _preds.shape
        _preds = self.softmax(_preds)
        _votes = _preds.mean(dim=1)

        return _votes
