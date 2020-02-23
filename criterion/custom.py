from typing import Tuple

import torch as th
from torch import nn

AE_CRITERION_FORWARD = Tuple[th.Tensor, th.Tensor]
VAE_CRITERION_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor]


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mean: th.Tensor, var: th.Tensor) -> th.Tensor:
        device = mean.device
        numel = mean.numel()

        kld = th.tensor(-0.5).to(device) * (1 + var.log() - mean.pow(2) - var).sum()

        return kld / numel


class AECriterion(nn.Module):
    mse_factor: float
    ce_factor: float

    def __init__(self):
        super(AECriterion, self).__init__()

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, _recon: th.Tensor, _pred: th.Tensor, _in: th.Tensor, _class: th.Tensor) -> AE_CRITERION_FORWARD:
        ce = self.ce(_pred, _class)
        l1 = self.l1(_recon, _in)

        return ce, l1


class VAECriterion(nn.Module):
    kld_factor: float

    def __init__(self):
        super(VAECriterion, self).__init__()

        self.kld_factor = 0.0

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')
        self.kld = KLDivergence()

    def forward(self, _recon: th.Tensor, _pred: th.Tensor, _in: th.Tensor, _class: th.Tensor,
                _mean: th.Tensor, _var: th.Tensor) -> VAE_CRITERION_FORWARD:
        # repeat targets according to number of targets.
        b, s, c = _pred.shape
        _class = _class.reshape(b, 1).repeat(1, s).reshape(b * s)
        _pred = _pred.reshape(b * s, c)
        b, s, t, c, h, w = _recon.shape
        _in = _in.reshape(b, 1, t, c, h, w).repeat(1, s, 1, 1, 1, 1).reshape(b * s, t, c, h, w)
        _recon = _recon.reshape(b * s, t, c, h, w)

        ce = self.ce(_pred, _class)
        l1 = self.l1(_recon, _in)
        kld = self.kld(_mean, _var)

        return ce, l1, kld
