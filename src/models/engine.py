import torch as th
from ignite import engine

import constants as ct
from pipeline import TensorNormalize, TensorStandardize

NORMALIZE = TensorNormalize(255)
STANDARDIZE = TensorStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS)


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options."""
    x, y = batch
    x = engine.convert_tensor(x, device=device, non_blocking=non_blocking)
    y = engine.convert_tensor(y, device=device, non_blocking=non_blocking)

    x = NORMALIZE(x)
    x = STANDARDIZE(x)

    return x, y


def create_discriminative_trainer(model, optimizer, loss_fn, device, non_blocking):
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
        return loss.item()

    return engine.Engine(_update)


def create_discriminative_evaluator(model, metrics=None, device=None, non_blocking=False):
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, embeds = model(x)
            return y_pred, y, embeds

    _engine = engine.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_variational_trainer(model, optimizer, loss_fn, device, non_blocking):
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        _recon, _pred, _latent, _mean, _log_var = model(x, inference=False, max_likelihood=False, num_samples=0)
        mse, ce, kld = loss_fn(_recon, _pred, x, y, _mean, _log_var)
        (mse + ce + kld).backward()
        optimizer.step()
        return mse.item(), ce.item(), kld.item()

    return engine.Engine(_update)


def create_variational_evaluator(model, metrics=None, device=None, non_blocking=False):
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            _ml_recon, _ml_pred, _, _mean, _log_var = model(x, inference=True, max_likelihood=True, num_samples=0)
            _, _var_preds, _, _, _ = model(x, inference=True, max_likelihood=False, num_samples=2)

            return _ml_recon, _ml_pred, _mean, _log_var, _var_preds, x, y

    _engine = engine.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine
