from typing import List, Tuple

import ignite.exceptions as ie
import ignite.metrics as im
import torch as th

import helpers as hp


class AverageMeter(im.Metric):
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
            raise ie.NotComputableError('Average must have at least one example before it can be computed')
        return self.sum / self.num_examples


class VAELoss(im.Metric):

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
            raise ie.NotComputableError(
                'Loss must have at least one example before it can be computed')
        return self.mse / self.num_examples, self.ce / self.num_examples, self.kld / self.num_examples


class AELoss(im.Metric):

    def __init__(self, loss_fn, output_transform=lambda x: x):
        super(AELoss, self).__init__(output_transform)
        self.loss_fn = loss_fn
        self.mse = 0
        self.ce = 0
        self.num_examples = 0

    def reset(self):
        self.mse = 0
        self.ce = 0
        self.num_examples = 0

    def update(self, output):
        _recon, _pred, _embed, _in, _target, = output
        mse, ce = self.loss_fn(_recon, _pred, _in, _target)

        n = _pred.shape[0]

        self.mse += mse.item() * n
        self.ce += ce.item() * n
        self.num_examples += n

    def compute(self):
        if self.num_examples == 0:
            raise ie.NotComputableError('Loss must have at least one example before it can be computed')
        return self.mse / self.num_examples, self.ce / self.num_examples


class MultiLabelIoU(im.Metric):
    """Computes the IoU for group labels."""

    def __init__(self, output_transform=lambda x: x):
        self.IoU = 0
        self.num_examples = 0
        self.lid2gid = hp.read_smth_lid2gid()
        super(MultiLabelIoU, self).__init__(output_transform=output_transform)

    def reset(self):
        self.IoU = 0
        self.num_examples = 0

    def update(self, output: Tuple[th.Tensor, th.Tensor]):
        lids_pred, lids_true = output
        n = lids_true.shape[0]

        gids_true = self._get_true_gids(lids_true)
        gids_pred = self._get_pred_gids(lids_pred, gids_true)

        for gid_true, gid_pred in zip(gids_true, gids_pred):
            self.IoU += len(gid_true.intersection(gid_pred)) / len(gid_true.union(gid_pred))
        self.num_examples += n

    def _get_true_gids(self, lids_true: th.Tensor) -> List[set]:
        gids_true = []

        for i in range(lids_true.shape[0]):
            lid_true = int(lids_true[i].cpu())
            gids_true.append(set(self.lid2gid.loc[lid_true].gid.values))

        return gids_true

    def _get_pred_gids(self, lids_pred: th.Tensor, gids_true: List[set]) -> List[set]:
        gids_pred = []

        for i, true_gid_set in enumerate(gids_true):
            lid_pred = lids_pred[i, :]
            topk_lid_pred = th.topk(lid_pred, k=len(true_gid_set))[1].cpu().numpy()

            pred_gid_set = set()
            for pred in topk_lid_pred:
                pred_gid_set = pred_gid_set.union(self.lid2gid.loc[pred].gid.values)

            gids_pred.append(pred_gid_set)

        return gids_pred

    def compute(self):
        if self.num_examples == 0:
            raise ie.NotComputableError('Average must have at least one example before it can be computed')
        return self.IoU / self.num_examples


if __name__ == '__main__':
    _lid_true = th.tensor([[0], [1], [3]])
    _lid_pred = th.tensor([[10, 9, 8, 7, 6, 5], [0, 1, 2, 3, 4, 5], [0, 1, 5, 4, 3, 4]])
    _iou = MultiLabelIoU()
    _iou.reset()
    _iou.update((_lid_pred, _lid_true))
    print(_iou.compute())
