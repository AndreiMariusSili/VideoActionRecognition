from ignite import engine


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options

    """
    x, y = batch
    return (engine.convert_tensor(x, device=device, non_blocking=non_blocking),
            engine.convert_tensor(y, device=device, non_blocking=non_blocking))


def create_discriminative_trainer(model, optimizer, loss_fn, device, non_blocking):
    if device:
        model.to(device)

    def _update(_engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    return engine.Engine(_update)
