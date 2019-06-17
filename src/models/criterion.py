from typing import Tuple

import torch as th
from torch import nn

AE_CRITERION_FORWARD = Tuple[th.Tensor, th.Tensor]
VAE_CRITERION_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor]


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mean: th.Tensor, log_var: th.Tensor) -> th.Tensor:
        device = mean.device
        bs, ls = mean.shape
        kld = th.tensor(-0.5).to(device) * (th.tensor(1.0).to(device) + log_var - mean.pow(2) - log_var.exp()).sum()

        return kld / (bs * ls)


class AECriterion(nn.Module):
    mse_factor: float
    ce_factor: float

    def __init__(self, mse_factor: float, ce_factor: float):
        super(AECriterion, self).__init__()

        self.mse_factor = mse_factor
        self.ce_factor = ce_factor

        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, _recon: th.Tensor, _pred: th.Tensor, _in: th.Tensor, _class: th.Tensor) -> AE_CRITERION_FORWARD:
        mse = self.mse_factor * self.mse(_recon, _in)
        ce = self.ce_factor * self.ce(_pred, _class)

        return mse, ce


class VAECriterion(nn.Module):
    mse_factor: float
    ce_factor: float
    kld_factor: float

    def __init__(self, mse_factor: float, ce_factor: float, kld_factor: float):
        super(VAECriterion, self).__init__()

        self.kld_factor = kld_factor
        self.mse_factor = mse_factor
        self.ce_factor = ce_factor

        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.kld = KLDivergence()

    def forward(self, _recon: th.Tensor, _pred: th.Tensor, _in: th.Tensor, _class: th.Tensor,
                _mean: th.Tensor, _log_var: th.Tensor) -> VAE_CRITERION_FORWARD:
        mse = self.mse_factor * self.mse(_recon, _in)
        ce = self.ce_factor * self.ce(_pred, _class)
        kld = self.kld_factor * self.kld(_mean, _log_var)

        return mse, ce, kld
