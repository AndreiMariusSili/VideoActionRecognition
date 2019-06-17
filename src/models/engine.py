from typing import Any, Dict

import torch as th
from ignite import engine as ie

import constants as ct
import models.criterion as mc
import pipeline.transforms as pit

NORMALIZE = pit.TensorNormalize(255)
STANDARDIZE = pit.TensorStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS)


def _prepare_batch(batch, device, non_blocking):
    """Prepare batch for training: pass to a device with options."""
    x, y = batch
    x = ie.convert_tensor(x, device=device, non_blocking=non_blocking)
    y = ie.convert_tensor(y, device=device, non_blocking=non_blocking)

    x = NORMALIZE(x)
    x = STANDARDIZE(x)

    return x, y


def create_cls_trainer(model, optimizer, loss_fn, metrics=None, device=None, non_blocking=True):
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred, embeds = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred.detach(), y.detach()

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_cls_evaluator(model, metrics=None, device=None, non_blocking=True):
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, embeds = model(x)
            return y_pred, y, embeds

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_vae_trainer(model, optimizer, loss_fn: mc.VAECriterion, metrics=None, device=None,
                       non_blocking=True) -> ie.Engine:
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        _recon, _pred, _latent, _mean, _log_var, _ = model(x, inference=False, num_samples=1)
        bs, ns, nc = _pred.shape
        mse, ce, kld = loss_fn(_recon, _pred.view(bs * ns, nc), x, y, _mean, _log_var)
        (mse + ce + kld).backward()
        optimizer.step()
        return _recon.detach(), _pred.detach(), _latent.detach(), _mean.detach(), _log_var.detach(), \
               x.detach(), y.detach(), \
               mse.item(), ce.item(), kld.item()

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_vae_evaluator(model, metrics=None, device=None, non_blocking=True) -> ie.Engine:
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            _recon, _pred, _latent, _mean, _log_var, _vote = model(x, inference=True, num_samples=ct.VAE_NUM_SAMPLES)
            return _recon, _pred, _latent, _mean, _log_var, x, y, _vote

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_ae_trainer(model, optimizer, loss_fn: mc.AECriterion, metrics: Dict[str, Any] = None,
                      device=None, non_blocking=True) -> ie.Engine:
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)

        _pred, _embed, _recon = model(x, inference=False)

        mse, ce = loss_fn(_recon, _pred, x, y)
        (mse + ce).backward()
        optimizer.step()
        return _recon.detach(), _pred.detach(), _embed.detach(), x.detach(), y.detach(), mse.item(), ce.item()

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_ae_evaluator(model, metrics=None, device=None, non_blocking=True) -> ie.Engine:
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            _pred, _embed, _recon = model(x, inference=True)
            return _recon, _pred, _embed, x, y

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine
