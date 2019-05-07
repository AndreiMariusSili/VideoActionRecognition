import torch as th
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class _VAEBaseClassification(Metric):

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
        batch_size, num_samples, num_classes = _var_preds.shape
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


class VAELoss(Metric):

    def __init__(self, loss_fn, output_transform=lambda x: x,
                 batch_size=lambda x: x.shape[0], take=0):
        super(VAELoss, self).__init__(output_transform)
        self._loss_fn = loss_fn
        self._batch_size = batch_size
        self._sum = 0
        self._num_examples = 0
        self.take = take

    def reset(self):
        self._sum = 0
        self._num_examples = 0

    def prepare(self, output):
        _ml_recon, _ml_pred, _mean, _log_var, _var_preds, _in, _target = output
        batch_size, num_samples, num_classes = _var_preds.shape
        _var_preds = _var_preds[0:batch_size, 0:num_samples:num_samples, 0:num_classes].view(batch_size, num_classes)

        return _ml_recon, _ml_pred, _mean, _log_var, _var_preds, _in, _target

    def update(self, output):
        _ml_recon, _ml_pred, _mean, _log_var, _var_preds, _in, _target = self.prepare(output)
        average_loss = self._loss_fn(_ml_recon, _var_preds, _in, _target, _mean, _log_var)[self.take]

        if len(average_loss.shape) != 0:
            raise ValueError('loss_fn did not return the average loss')

        N = self._batch_size(_ml_recon)
        self._sum += average_loss.item() * N
        self._num_examples += N

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self._sum / self._num_examples


if __name__ == '__main__':
    acc1 = VAEAccuracyAt1()
    acc2 = VAEAccuracyAt2()

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

    acc1.update((ml_recon, ml_pred, mean, log_var, var_preds, target))
    print(acc1.compute())

    acc2.update((ml_recon, ml_pred, mean, log_var, var_preds, target))
    print(acc2.compute())
