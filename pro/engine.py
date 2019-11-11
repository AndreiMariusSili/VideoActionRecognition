import ignite.engine as ie
import torch as th


def prepare_batch(batch, device, non_blocking):
    """Prepare batch for training: pass to a device with options."""
    x, y, _, _ = batch
    x = ie.convert_tensor(x, device=device, non_blocking=non_blocking)
    y = ie.convert_tensor(y, device=device, non_blocking=non_blocking)

    return x, y


def create_cls_trainer(model, optimizer, crt, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create classification trainer."""
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred, embeds = model(x)
        loss = crt(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred.detach(), y.detach()

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_cls_evaluator(model, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create classification evaluator."""
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, embeds = model(x)
            return y_pred, y, embeds

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_ae_trainer(model, optimizer, crt, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create autoencoder trainer."""
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        _pred, _embed, _recon = model(x, inference=False)

        ce, mse = crt(_recon, _pred, x, y)
        (ce + mse).backward()
        optimizer.step()

        return _recon.detach(), _pred.detach(), _embed.detach(), x.detach(), y.detach(), ce.item(), mse.item()

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_ae_evaluator(model, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create autoencoder evaluator."""
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            _pred, _embed, _recon = model(x, inference=True)
            return _recon, _pred, _embed, x, y

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_vae_trainer(model, optimizer, crt, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create variational autoencoder trainer."""
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        _recon, _pred, _latent, _mean, _var, _ = model(x, inference=False, num_samples=1)
        bs, ns, nc = _pred.shape
        ce, mse, kld = crt(_recon, _pred.reshape(bs * ns, nc), x, y, _mean, _var)

        (ce + mse + crt.kld_factor * kld).backward()
        optimizer.step()

        return _recon.detach(), _pred.detach(), _latent.detach(), _mean.detach(), _var.detach(), \
               x.detach(), y.detach(), \
               ce.item(), mse.item(), kld.item(), \
               crt.kld_factor

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_vae_evaluator(model, metrics=None, device=None, num_samples: int = None, non_blocking=True) -> ie.Engine:
    """Create variational autoencoder evaluator."""
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            _recon, _pred, _latent, _mean, _log_var, _vote = model(x, inference=True, num_samples=num_samples)
            return _recon, _pred, _latent, _mean, _log_var, x, y, _vote

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine
