import dataclasses as dc
import timeit

import pandas as pd
import pytest
import torch as th

import helpers as hp
import specs.maps as sm
import specs.models as mo

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

BATCH_SIZE = 2
TIME_STEPS = 4
C, H, W = 3, 224, 224
CLASS_EMBED_PLANES = 1024
NUM_CLASSES = 51
VAE_TEST_NUM_SAMPLES = 3
CLASS_ARGS = [
    ('tadn_class_4', 4), ('tadn_class_8', 8),
    ('tarn_class_4', 4), ('tarn_class_8', 8),
    ('i3d_class_4', 4), ('i3d_class_8', 8),
]
AE_ARGS = [
    ('tarn_ae_4', 4), ('tarn_ae_8', 8),
    ('tarn_flow_4', 4), ('tarn_flow_8', 8),
    ('i3d_ae_4', 4), ('i3d_ae_8', 8),
    ('i3d_flow_4', 4), ('i3d_flow_8', 8),
]
VAE_ARGS = [
    ('tarn_vae_4', 4), ('tarn_vae_8', 8),
    ('i3d_vae_4', 4), ('i3d_vae_8', 8)
]


def _init_model(model_spec: str):
    model_spec = getattr(mo, model_spec)
    model_spec.opts.num_classes = NUM_CLASSES
    opts = dc.asdict(model_spec.opts)
    del opts['batch_size']
    return sm.Models[model_spec.arch].value(**opts), model_spec.opts


@pytest.mark.parametrize(['model_spec', 'time_steps'], CLASS_ARGS)
@th.no_grad()
def test_class_model(model_spec: str, time_steps: int):
    print()
    model, opts = _init_model(model_spec)
    model = model.to('cuda' if th.cuda.is_available() else 'cpu')
    _in = th.randn(BATCH_SIZE, time_steps, C, H, W).to('cuda' if th.cuda.is_available() else 'cpu')
    model.eval()
    model(_in)

    def time_func():
        return model(_in)

    time = timeit.timeit(time_func, number=100) / 100
    pred, temporal_embed, class_embed = model(_in)

    df = pd.DataFrame.from_dict({
        'model': [model.NAME],  # noqa
        'size': [f'{hp.count_parameters(model):,}'],
        'inference_speed': [f'{time:.4f}'],
        'preds': [str(tuple(pred.shape))],
        'embeds': [str(tuple(class_embed.shape))],
    })
    print(df)

    class_embed_planes = CLASS_EMBED_PLANES
    if hasattr(opts, 'class_embed_planes'):
        class_embed_planes = opts.class_embed_planes

    assert pred.shape == (BATCH_SIZE, NUM_CLASSES)
    assert class_embed.shape == (BATCH_SIZE, class_embed_planes)


@pytest.mark.parametrize(['model_spec', 'time_steps'], AE_ARGS)
@th.no_grad()
def test_ae_model(model_spec: str, time_steps: int):
    print()
    model, opts = _init_model(model_spec)
    model = model.to('cuda' if th.cuda.is_available() else 'cpu')
    _in = th.randn(BATCH_SIZE, time_steps, C, H, W).to('cuda' if th.cuda.is_available() else 'cpu')

    recon, pred, temporal_embeds, class_embed = model(_in)

    df = pd.DataFrame.from_dict({
        'model': [model_spec],
        'size': [f'{hp.count_parameters(model):,}'],
        'preds': [str(tuple(pred.shape))],
        'embeds': [str(tuple(class_embed.shape))],
        'recons': [str(tuple(recon.shape))]
    })
    print(df)

    class_embed_planes = CLASS_EMBED_PLANES
    if hasattr(opts, 'class_embed_planes'):
        class_embed_planes = opts.class_embed_planes
        assert pred.shape == (BATCH_SIZE, NUM_CLASSES)
    assert class_embed.shape == (BATCH_SIZE, class_embed_planes)
    assert recon.shape == (BATCH_SIZE, time_steps, 2 if '_flow_' in model_spec else 3, 56, 56)


@pytest.mark.parametrize(['model_spec', 'time_steps'], VAE_ARGS)
@th.no_grad()
def test_vae_model(model_spec: str, time_steps: int):
    print()
    model, opts = _init_model(model_spec)
    model = model.to('cuda' if th.cuda.is_available() else 'cpu')
    _in = th.randn(BATCH_SIZE, time_steps, C, H, W).to('cuda' if th.cuda.is_available() else 'cpu')

    model.train()
    out = model(_in, 1)
    train_recon, train_pred, train_temp_embed, train_class_embed, train_mean, train_variance, train_vote = out
    df = pd.DataFrame.from_dict({
        'mode': ['TRAINING'],
        'model': [model.NAME],
        'size': [f'{hp.count_parameters(model):,}'],
        'preds': [str(tuple(train_pred.shape))],
        'embeds': [str(tuple(train_class_embed.shape))],
        'means': [str(tuple(train_mean.shape))],
        'vars': [str(tuple(train_variance.shape))],
        'recons': [str(tuple(train_recon.shape))],
        'vote': [None],
    })
    print(df)

    model.eval()
    ml_inf_recon, ml_inf_pred, ml_inf_temp_embed, ml_inf_class_embed, ml_inf_mean, ml_inf_variance, ml_inf_vote = model(
        _in, 0)
    df = pd.DataFrame.from_dict({
        'mode': ['ML INFERENCE'],
        'model': [model.NAME],
        'size': [f'{hp.count_parameters(model):,}'],
        'preds': [str(tuple(ml_inf_pred.shape))],
        'embeds': [str(tuple(ml_inf_class_embed.shape))],
        'means': [str(tuple(ml_inf_mean.shape))],
        'vars': [str(tuple(ml_inf_variance.shape))],
        'recons': [str(tuple(ml_inf_recon.shape))],
        'vote': [ml_inf_vote.shape],
    })
    print(df)

    model.eval()
    out = model(_in, 3)
    var_inf_recon, var_inf_pred, var_inf_temp_embed, var_inf_class_embed, var_inf_mean, var_inf_variance, var_inf_vote = out
    df = pd.DataFrame.from_dict({
        'mode': ['VARIATIONAL INFERENCE'],
        'model': [model.NAME],
        'size': [f'{hp.count_parameters(model):,}'],
        'preds': [str(tuple(var_inf_pred.shape))],
        'embeds': [str(tuple(var_inf_class_embed.shape))],
        'means': [str(tuple(var_inf_mean.shape))],
        'vars': [str(tuple(var_inf_variance.shape))],
        'recons': [str(tuple(var_inf_recon.shape))],
        'vote': [var_inf_vote.shape],
    })
    print(df)

    class_embed_planes = CLASS_EMBED_PLANES
    if hasattr(opts, 'class_embed_planes'):
        class_embed_planes = opts.class_embed_planes

    assert train_pred.shape == (BATCH_SIZE, 1, NUM_CLASSES)
    assert train_class_embed.shape == (BATCH_SIZE, 1, class_embed_planes)
    assert train_recon.shape == (BATCH_SIZE, 1, time_steps, C, H, W)

    assert ml_inf_pred.shape == (BATCH_SIZE, 1, NUM_CLASSES)
    assert ml_inf_class_embed.shape == (BATCH_SIZE, 1, class_embed_planes)
    assert ml_inf_recon.shape == (BATCH_SIZE, 1, time_steps, C, H, W)
    assert ml_inf_vote.shape == (BATCH_SIZE, NUM_CLASSES)

    assert var_inf_pred.shape == (BATCH_SIZE, VAE_TEST_NUM_SAMPLES, NUM_CLASSES)
    assert var_inf_class_embed.shape == (BATCH_SIZE, VAE_TEST_NUM_SAMPLES, class_embed_planes)
    assert var_inf_recon.shape == (BATCH_SIZE, VAE_TEST_NUM_SAMPLES, time_steps, C, H, W)
    assert var_inf_vote.shape == (BATCH_SIZE, NUM_CLASSES)
