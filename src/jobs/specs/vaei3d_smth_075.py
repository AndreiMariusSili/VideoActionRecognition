import os

import dataclasses as dc
from torch import cuda, optim

import constants as ct
import pipeline as pipe
from models import criterion, metrics as custom_metrics, options, vae_i3d

NUM_DEVICES = cuda.device_count() if cuda.device_count() > 0 else 1

########################################################################################################################
# DATA BUNCH OPTIONS
########################################################################################################################
db_opts = pipe.DataBunchOptions(
    shape='volume',
    frame_size=224,
)
########################################################################################################################
# TRAIN DATA
########################################################################################################################
train_do = pipe.DataOptions(
    meta_path=ct.SMTH_META_TRAIN,
    cut=0.75,
    setting='train',
)
train_so = pipe.SamplingOptions(
    num_segments=4,
    segment_size=1
)
train_ds_opts = pipe.DataSetOptions(
    do=train_do,
    so=train_so
)
train_dl_opts = pipe.DataLoaderOptions(
    batch_size=128,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# VALID DATA
########################################################################################################################
valid_do = pipe.DataOptions(
    meta_path=ct.SMTH_META_VALID,
    cut=0.75,
    setting='valid',
)
valid_so = pipe.SamplingOptions(
    num_segments=4,
    segment_size=1
)
valid_ds_opts = pipe.DataSetOptions(
    do=valid_do,
    so=valid_so
)
valid_dl_opts = pipe.DataLoaderOptions(
    batch_size=128,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# MODEL AND AUXILIARIES
########################################################################################################################
model_opts = options.VAEI3DOptions(
    latent_size=1024,
    dropout_prob=0.5,
    num_classes=ct.SMTH_NUM_CLASSES
)
optimizer_opts = options.AdamOptimizerOptions(
    lr=0.001
)
trainer_opts = options.TrainerOptions(
    epochs=100,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=criterion.VAECriterion,
    criterion_opts=options.VAECriterionOptions(
        mse_factor=1.0,
        ce_factor=1.0,
        kld_factor=1.0
    )
)
evaluator_opts = options.EvaluatorOptions(
    metrics={
        'acc@1': custom_metrics.VAEAccuracyAt1(),
        'acc@2': custom_metrics.VAEAccuracyAt2(),
        'mse_loss': custom_metrics.VAELoss(criterion.VAECriterion(**dc.asdict(trainer_opts.criterion_opts)), take=0),
        'ce_loss': custom_metrics.VAELoss(criterion.VAECriterion(**dc.asdict(trainer_opts.criterion_opts)), take=1),
        'kld_loss': custom_metrics.VAELoss(criterion.VAECriterion(**dc.asdict(trainer_opts.criterion_opts)), take=2),
        'total_loss': custom_metrics.VAELoss(criterion.VAECriterion(**dc.asdict(trainer_opts.criterion_opts)), take=-1),
    }
)
########################################################################################################################
# RUN
########################################################################################################################
vaei3d_smth_075 = options.RunOptions(
    name='vaei3d_smth_075',
    mode='variational',
    resume=False,
    log_interval=10,
    patience=10,
    model=vae_i3d.VAEI3D,
    model_opts=model_opts,
    data_bunch=pipe.SmthDataBunch,
    db_opts=db_opts,
    train_ds_opts=train_ds_opts,
    valid_ds_opts=valid_ds_opts,
    train_dl_opts=train_dl_opts,
    valid_dl_opts=valid_dl_opts,
    trainer_opts=trainer_opts,
    evaluator_opts=evaluator_opts
)
