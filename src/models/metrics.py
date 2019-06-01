from typing import Tuple

import torch as th
from ignite import metrics
from ignite.exceptions import NotComputableError


class AverageMeter(metrics.Metric):
    """Computes and stores the average and current value"""

    def __init__(self, output_transform=lambda x: x):
        self.sum = 0
        self.num_examples = 0
        super(AverageMeter, self).__init__(output_transform=output_transform)

    def reset(self):
        self.sum = 0
        self.num_examples = 0

    def update(self, output):
        val, n = output

        self.sum += val * n
        self.num_examples += n

    def compute(self):
        if self.num_examples == 0:
            raise NotComputableError('Average must have at least one example before it can be computed')
        return self.sum / self.num_examples


class _VAEBaseClassification(metrics.Metric):

    def __init__(self, output_transform=lambda x: x):
        self._type = None
        self._num_correct = 0
        self._num_examples = 0
        super(_VAEBaseClassification, self).__init__(output_transform=output_transform)

    def _check_shape(self, output):
        y_pred, y = output

        if y.ndimension() > 1 and y.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            y_pred = y_pred.squeeze(dim=1)

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
                             "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                             "but given {} vs {}".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        return y_pred, y

    def reset(self):
        pass

    def update(self, output):
        pass

    def compute(self):
        pass

    def _check_type(self, output):
        y_pred, y = output

        if y.ndimension() + 1 == y_pred.ndimension():
            update_type = "multiclass"
        elif y.ndimension() == y_pred.ndimension():
            update_type = "binary"
            if not th.equal(y, y ** 2):
                raise ValueError("For binary cases, y must be comprised of 0's and 1's.")
        else:
            raise RuntimeError("Invalid shapes of y (shape={}) and y_pred (shape={}), check documentation"
                               " for expected shapes of y and y_pred.".format(y.shape, y_pred.shape))
        if self._type is None:
            self._type = update_type
        else:
            if self._type != update_type:
                raise RuntimeError("update_type has changed from {} to {}.".format(self._type, update_type))


class VAEAccuracyAt1(_VAEBaseClassification):

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def update(self, output):
        _ml_recon, _ml_pred, _mean, _log_var, _var_preds, _in, _target = output

        y_pred, y = self._check_shape((_ml_pred, _target))
        self._check_type((y_pred, y))

        if self._type == "binary":
            indices = th.round(y_pred).type(y.type())
        elif self._type == "multiclass":
            indices = th.argmax(y_pred, dim=1)
        else:
            raise ValueError(f'Unknown type: {self._type}.')

        correct = th.eq(indices, y).view((-1,))
        self._num_correct += th.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class VAEAccuracyAt2(_VAEBaseClassification):

    def reset(self):
        self._num_correct = 0
        self._num_examples = 0

    def prepare(self, output):
        _ml_recon, _ml_pred, _mean, _log_var, _var_preds, _in, _target = output
        batch_size = _target.shape[0]
        num_preds, num_classes = _var_preds.shape
        num_samples = num_preds // batch_size
        _target = _target.view(-1, 1).repeat(1, num_samples).view(-1)
        _var_preds = _var_preds.view(batch_size * num_samples, num_classes)

        return _var_preds, _target

    def update(self, output):
        _var_preds, _target = self.prepare(output)

        y_pred, y = self._check_shape((_var_preds, _target))
        self._check_type((y_pred, y))

        if self._type == "binary":
            indices = th.round(y_pred).type(y.type())
        elif self._type == "multiclass":
            indices = th.argmax(y_pred, dim=1)
        else:
            raise ValueError(f'Unknown type: {self._type}.')

        correct = th.eq(indices, y).view((-1,))
        self._num_correct += th.sum(correct).item()
        self._num_examples += correct.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Accuracy must have at least one example before it can be computed')
        return self._num_correct / self._num_examples


class VAELoss(metrics.Metric):

    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(VAELoss, self).__init__(output_transform)
        self.loss_fn = loss_fn
        self.mse = 0
        self.ce = 0
        self.kld = 0
        self.num_examples = 0

    def reset(self):
        self.mse = 0
        self.ce = 0
        self.kld = 0
        self.num_examples = 0

    def prepare(self, output: Tuple[th.Tensor, ...]) -> Tuple[th.Tensor, ...]:
        _recon, _pred, _latent, _mean, _log_var, _in, _target, _ = output
        _pred = _pred[:, 0, :].squeeze(dim=1)

        return _recon, _pred, _latent, _mean, _log_var, _in, _target

    def update(self, output):
        _recon, _pred, _latent, _mean, _log_var, _input, _target, = self.prepare(output)
        mse, ce, kld = self.loss_fn(_recon, _pred, _input, _target, _mean, _log_var)

        n = _pred.shape[0]

        self.mse += mse.item() * n
        self.ce += ce.item() * n
        self.kld += kld.item() * n
        self.num_examples += n

    def compute(self):
        if self.num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self.mse / self.num_examples, self.ce / self.num_examples, self.kld / self.num_examples


if __name__ == '__main__':
    acc1 = VAEAccuracyAt1()
    acc2 = VAEAccuracyAt2()

    __input = th.randn((2, 4, 3, 224, 224), dtype=th.float)
    ml_recon = th.randn((2, 4, 3, 224, 224), dtype=th.float)
    ml_pred = th.rand((3, 10), dtype=th.float)
    ml_pred[0, 0] = 1.00
    ml_pred[1, 1] = 0.00
    ml_pred[2, 2] = 0.00
    var_preds = th.rand((3, 2, 10), dtype=th.float)
    var_preds[0, 0, 0] = 1.00
    var_preds[0, 1, 0] = 1.00
    var_preds[1, 0, 1] = 1.00
    var_preds[1, 1, 1] = 0.00
    var_preds[2, 0, 2] = 0.00
    var_preds[2, 1, 2] = 0.00
    target = th.tensor([0, 1, 2], dtype=th.long)
    mean = th.randn((3, 1024), dtype=th.float)
    log_var = th.randn((3, 1024), dtype=th.float)

    acc1.update((ml_recon, ml_pred, mean, log_var, var_preds, __input, target))
    print(acc1.compute())

    acc2.update((ml_recon, ml_pred, mean, log_var, var_preds, __input, target))
    print(acc2.compute())
