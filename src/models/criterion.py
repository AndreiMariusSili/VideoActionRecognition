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
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, _recon: th.Tensor, _pred: th.Tensor, _in: th.Tensor, _class: th.Tensor) -> AE_CRITERION_FORWARD:
        ce = self.ce(_pred, _class)
        mse = self.mse(_recon, _in)

        return ce, mse


class VAECriterion(nn.Module):
    kld_factor: float

    def __init__(self):
        super(VAECriterion, self).__init__()

        self.kld_factor = 0.0

        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.kld = KLDivergence()

    def forward(self, _recon: th.Tensor, _pred: th.Tensor, _in: th.Tensor, _class: th.Tensor,
                _mean: th.Tensor, _log_var: th.Tensor) -> VAE_CRITERION_FORWARD:
        ce = self.ce(_pred, _class)
        mse = self.mse(_recon, _in)
        kld = self.kld(_mean, _log_var)

        return ce, mse, kld
