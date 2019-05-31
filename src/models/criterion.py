import torch as th
from torch import nn


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mean: th.Tensor, log_var: th.Tensor):
        device = mean.device

        kld = th.tensor(-0.5).to(device) * (th.tensor(1.0).to(device) + log_var - mean.pow(2) - log_var.exp()).mean()

        return kld


class VAECriterion(nn.Module):
    factor: int

    def __init__(self, mse_factor: float, ce_factor: float, kld_factor: float):
        super(VAECriterion, self).__init__()

        self.mse_factor = mse_factor
        self.ce_factor = ce_factor
        self.kld_factor = kld_factor

        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.kld = KLDivergence()

    def forward(self, _recon: th.Tensor, _pred: th.Tensor,
                _in: th.Tensor, _class: th.Tensor,
                _mean: th.Tensor, _log_var: th.Tensor):
        mse = self.mse_factor * self.mse(_recon, _in)
        ce = self.ce_factor * self.ce(_pred, _class)
        kld = self.kld_factor * self.kld(_mean, _log_var)

        return mse, ce, kld
