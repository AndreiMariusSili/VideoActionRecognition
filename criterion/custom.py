from typing import Tuple

import torch as th
from torch import nn

AE_CRITERION_FORWARD = Tuple[th.Tensor, th.Tensor]
GSNN_CRITERION_FORWARD = Tuple[th.Tensor, th.Tensor]
VAE_CRITERION_FORWARD = Tuple[th.Tensor, th.Tensor, th.Tensor]


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mean: th.Tensor, var: th.Tensor) -> th.Tensor:
        device = mean.device

        kld = th.tensor(-0.5).to(device) * (1 + var.log() - mean.pow(2) - var).mean()

        return kld


class AECriterion(nn.Module):
    def __init__(self):
        super(AECriterion, self).__init__()

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, _recon_pred: th.Tensor, _cls_pred: th.Tensor,
                _recon_gt: th.Tensor, _cls_gt: th.Tensor) -> AE_CRITERION_FORWARD:
        ce = self.ce(_cls_pred, _cls_gt)
        l1 = self.l1(_recon_pred, _recon_gt)

        return ce, l1


class GSNNCriterion(nn.Module):
    kld_factor: float

    def __init__(self):
        super(GSNNCriterion, self).__init__()

        self.kld_factor = 0.0

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.kld = KLDivergence()

    def forward(self, _class_pred: th.Tensor, _class_gt: th.Tensor,
                _mean: th.Tensor, _var: th.Tensor) -> GSNN_CRITERION_FORWARD:
        # repeat predictions according to number of targets.
        b, s, c = _class_pred.shape
        _class_gt = _class_gt.reshape(b, 1).repeat(1, s).reshape(b * s)
        _class_pred = _class_pred.reshape(b * s, c)

        ce = self.ce(_class_pred, _class_gt)
        kld = self.kld(_mean, _var)

        return ce, kld


class VAECriterion(nn.Module):
    kld_factor: float

    def __init__(self):
        super(VAECriterion, self).__init__()

        self.kld_factor = 0.0

        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.l1 = nn.SmoothL1Loss(reduction='mean')
        self.kld = KLDivergence()

    def forward(self, _recon_pred: th.Tensor, _class_pred: th.Tensor, _recon_gt: th.Tensor, _class_gt: th.Tensor,
                _mean: th.Tensor, _var: th.Tensor) -> VAE_CRITERION_FORWARD:
        # repeat preds according to number of targets.
        b, s, c = _class_pred.shape
        _class_gt = _class_gt.reshape(b, 1).repeat(1, s).reshape(b * s)
        _class_pred = _class_pred.reshape(b * s, c)
        b, s, t, c, h, w = _recon_pred.shape
        _recon_gt = _recon_gt.reshape(b, 1, t, c, h, w).repeat(1, s, 1, 1, 1, 1).reshape(b * s, t, c, h, w)
        _recon_pred = _recon_pred.reshape(b * s, t, c, h, w)

        ce = self.ce(_class_pred, _class_gt)
        l1 = self.l1(_recon_pred, _recon_gt)
        kld = self.kld(_mean, _var)

        return ce, l1, kld
