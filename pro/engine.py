import ignite.engine as ie
import torch as th


def prepare_batch(batch, device, non_blocking):
    """Prepare batch for training: pass to a device with options."""
    input_data, class_target_data, recon_target_data, _ = batch
    input_data = ie.convert_tensor(input_data, device=device, non_blocking=non_blocking)
    class_target_data = ie.convert_tensor(class_target_data, device=device, non_blocking=non_blocking)
    recon_target_data = ie.convert_tensor(recon_target_data, device=device, non_blocking=non_blocking)

    return input_data, class_target_data, recon_target_data


def create_cls_trainer(model, optimizer, crt, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create classification trainer."""
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        _in, _cls_gt, _ = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred, temporal_embeds, class_embed = model(_in)
        loss = crt(y_pred, _cls_gt)
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred.detach(), _cls_gt.detach()

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
            _in, _cls_gt, _ = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred, temporal_embeds, class_embed = model(_in)
            return y_pred, _cls_gt, temporal_embeds, class_embed

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_ae_trainer(model, optimizer, crt, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create auto-encoder trainer."""
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        _in, _cls_gt, _recon_gt = prepare_batch(batch, device=device, non_blocking=non_blocking)

        _recon_pred, _cls_pred, _temporal_embeds, _class_embed = model(_in)

        ce, l1 = crt(_recon_pred, _cls_pred, _recon_gt, _cls_gt)
        (ce + l1).backward()

        optimizer.step()

        return (
            _recon_pred.detach(),
            _cls_pred.detach(),
            _temporal_embeds.detach(),
            _class_embed.detach(),
            _in.detach(),
            _cls_gt.detach(),
            ce.item(),
            l1.item()
        )

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_ae_evaluator(model, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create auto-encoder evaluator."""
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            _in, _cls_gt, _recon_gt = prepare_batch(batch, device=device, non_blocking=non_blocking)
            _recon_pred, _cls_pred, _temporal_embeds, _class_embed = model(_in)
            return _recon_pred, _cls_pred, _temporal_embeds, _class_embed, _recon_gt, _cls_gt

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_gsnn_trainer(model, optimizer, crt, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create variational auto-encoder trainer."""
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        _in, _cls_gt, _ = prepare_batch(batch, device=device, non_blocking=non_blocking)
        _cls_pred, _temporal_latents, _class_latent, _mean, _var, _ = model(_in, num_samples=1)
        ce, kld = crt(_cls_pred, _cls_gt, _mean, _var)

        (ce + crt.kld_factor * kld).backward()
        optimizer.step()

        return (
            _cls_pred.detach(),
            _temporal_latents.detach(),
            _class_latent.detach(),
            _mean.detach(),
            _var.detach(),
            _in.detach(),
            _cls_gt.detach(),
            ce.item(),
            kld.item(),
            crt.kld_factor
        )

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_gsnn_evaluator(model, metrics=None, device=None, num_samples: int = None, non_blocking=True) -> ie.Engine:
    """Create variational auto-encoder evaluator."""
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            _in, _cls_gt, _ = prepare_batch(batch, device=device, non_blocking=non_blocking)
            _cls_pred, _temp_lat, _cls_lat, _mean, _var, _vote = model(_in, num_samples=num_samples)
            return _cls_pred, _temp_lat, _cls_lat, _mean, _var, _in, _cls_gt, _vote

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_vae_trainer(model, optimizer, crt, metrics=None, device=th.device('cpu'), non_blocking=True) -> ie.Engine:
    """Create variational auto-encoder trainer."""
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        _in, _cls_gt, _recon_gt = prepare_batch(batch, device=device, non_blocking=non_blocking)
        _recon_pred, _cls_pred, _temporal_latents, _class_latent, _mean, _var, _ = model(_in, num_samples=1)
        ce, l1, kld = crt(_recon_pred, _cls_pred, _recon_gt, _cls_gt, _mean, _var)

        (ce + l1 + crt.kld_factor * kld).backward()
        optimizer.step()

        return (
            _recon_pred.detach(),
            _cls_pred.detach(),
            _temporal_latents.detach(),
            _class_latent.detach(),
            _mean.detach(),
            _var.detach(),
            _in.detach(),
            _cls_gt.detach(),
            ce.item(),
            l1.item(),
            kld.item(),
            crt.kld_factor
        )

    _engine = ie.Engine(_update)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine


def create_vae_evaluator(model, metrics=None, device=None, num_samples: int = None, non_blocking=True) -> ie.Engine:
    """Create variational auto-encoder evaluator."""
    if device:
        model.to(device)

    def _inference(_engine, batch):
        model.eval()
        with th.no_grad():
            _in, _cls_gt, _recon_gt = prepare_batch(batch, device=device, non_blocking=non_blocking)
            _recon_pred, _cls_pred, _temp_lat, _cls_lat, _mean, _var, _vote = model(_in, num_samples=num_samples)
            return _recon_pred, _cls_pred, _temp_lat, _cls_lat, _mean, _var, _in, _cls_gt, _vote

    _engine = ie.Engine(_inference)
    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(_engine, name)

    return _engine
