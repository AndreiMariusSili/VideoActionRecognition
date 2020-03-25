from typing import Optional, Tuple

import torch as th
from torch import nn

import models.helpers as hp
from models.tarn.vae import _classifier as cls, _spatial_encoder as se, _temporal_encoder as te

GSNN_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, Optional[th.Tensor]]


class GSNNTimeAlignedResNet(nn.Module):
    def __init__(self, time_steps: int,
                 spatial_encoder_planes: Tuple[int, ...], bottleneck_planes: int,
                 classifier_drop_rate: float, class_embed_planes: int, num_classes: int, vote_type: str):
        super(GSNNTimeAlignedResNet, self).__init__()

        self.spatial_encoder_planes = spatial_encoder_planes
        self.spatial_bottleneck_planes = bottleneck_planes

        self.time_steps = time_steps
        self.drop_rate = classifier_drop_rate
        self.num_classes = num_classes
        self.class_embed_planes = class_embed_planes
        self.vote_type = vote_type

        self.class_in_planes = self.spatial_bottleneck_planes * self.time_steps

        self.spatial_encoder = se.VarSpatialResNetEncoder(self.spatial_encoder_planes, self.spatial_bottleneck_planes)
        self.temporal_encoder = te.VarTemporalResNetEncoder(self.time_steps,
                                                            self.spatial_bottleneck_planes)
        self.classifier = cls.VarTimeAlignedResNetClassifier(self.class_in_planes,
                                                             self.class_embed_planes,
                                                             self.drop_rate,
                                                             self.num_classes)

        self.softmax = nn.Softmax(dim=-1)

        hp.he_init(self)

    def forward(self, _in: th.Tensor, num_samples: int) -> GSNN_FORWARD:
        if self.training:
            return self._forward(_in, num_samples)
        else:
            return self._inference(_in, num_samples)

    def _inference(self, _in: th.Tensor, num_samples: int) -> GSNN_FORWARD:
        bs, t, c, h, w = _in.shape

        _spatial_embed = self.spatial_encoder(_in)
        _temporal_latent, _temporal_latents, _means, _vars = self.temporal_encoder(_spatial_embed, num_samples)
        _pred, _class_latent = self.classifier(_temporal_latents)

        num_samples = max(1, num_samples)
        if self.vote_type == 'hard':
            _vote = self._hard_vote(_pred.reshape(bs, num_samples, self.num_classes))
        else:
            _vote = self._soft_vote(_pred.reshape(bs, num_samples, self.num_classes))

        return _pred, _temporal_latents, _class_latent, _means, _vars, _vote

    def _forward(self, _in: th.Tensor, num_samples: int = 1) -> GSNN_FORWARD:
        _spatial_embed = self.spatial_encoder(_in)
        _temporal_latent, _temporal_latents, _means, _vars = self.temporal_encoder(_spatial_embed, num_samples)
        _pred, _class_latent = self.classifier(_temporal_latents)

        return _pred, _temporal_latents, _class_latent, _means, _vars, None

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
