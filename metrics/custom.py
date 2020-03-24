from typing import Tuple

import ignite.exceptions as ie
import ignite.metrics as im
import torch as th


class AELoss(im.Metric):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(AELoss, self).__init__(output_transform)
        self.loss_fn = loss_fn
        self.ce = 0
        self.l1 = 0
        self.num_examples = 0

    def reset(self):
        self.ce = 0
        self.l1 = 0
        self.num_examples = 0

    def update(self, output):
        _recon, _pred, _temporal_embeds, _class_embed, _in, _target = output
        ce, mse = self.loss_fn(_recon, _pred, _in, _target)

        n = _pred.shape[0]

        self.ce += ce.item() * n
        self.l1 += mse.item() * n
        self.num_examples += n

    def compute(self):
        if self.num_examples == 0:
            raise ie.NotComputableError('Loss must have at least one example before it can be computed')
        return self.ce / self.num_examples, self.l1 / self.num_examples


class GSNNLoss(im.Metric):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(GSNNLoss, self).__init__(output_transform)
        self.loss_fn = loss_fn
        self.ce = 0
        self.kld = 0
        self.num_examples = 0

    def reset(self):
        self.ce = 0
        self.kld = 0
        self.num_examples = 0

    def prepare(self, output: Tuple[th.Tensor, ...]) -> Tuple[th.Tensor, ...]:  # noqa
        _pred, _temporal_latents, _class_latent, _mean, _var, _in, _target, _ = output

        return _pred, _temporal_latents, _class_latent, _mean, _var, _in, _target

    def update(self, output):
        _pred, _temporal_latents, _class_latent, _mean, _var, _in, _target = self.prepare(output)
        ce, kld = self.loss_fn(_pred, _target, _mean, _var)

        n = _pred.shape[0]

        self.ce += ce.item() * n
        self.kld += kld.item() * n
        self.num_examples += n

    def compute(self) -> Tuple[float, float]:
        if self.num_examples == 0:
            raise ie.NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self.ce / self.num_examples, self.kld / self.num_examples


class VAELoss(im.Metric):
    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(VAELoss, self).__init__(output_transform)
        self.loss_fn = loss_fn
        self.l1 = 0
        self.ce = 0
        self.kld = 0
        self.num_examples = 0

    def reset(self):
        self.l1 = 0
        self.ce = 0
        self.kld = 0
        self.num_examples = 0

    def prepare(self, output: Tuple[th.Tensor, ...]) -> Tuple[th.Tensor, ...]:  # noqa
        _recon, _pred, _temporal_latents, _class_latent, _mean, _var, _in, _target, _ = output

        return _recon, _pred, _temporal_latents, _class_latent, _mean, _var, _in, _target

    def update(self, output):
        _recon, _pred, _temporal_latents, _class_latent, _mean, _var, _in, _target = self.prepare(output)
        ce, l1, kld = self.loss_fn(_recon, _pred, _in, _target, _mean, _var)

        n = _pred.shape[0]

        self.ce += ce.item() * n
        self.l1 += l1.item() * n
        self.kld += kld.item() * n
        self.num_examples += n

    def compute(self) -> Tuple[float, float, float]:
        if self.num_examples == 0:
            raise ie.NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self.ce / self.num_examples, self.l1 / self.num_examples, self.kld / self.num_examples
