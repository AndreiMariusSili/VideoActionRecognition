import copy

import dataclasses as dc
import ignite.metrics as im
from torch import optim

import constants as ct
import pipeline.smth.databunch as smth
from jobs.specs.__dev._smth_dev import *
from models import criterion, metrics, vae_i3d
from options import model_options

train_dl_opts = copy.deepcopy(train_dl_opts)
dev_dl_opts = copy.deepcopy(dev_dl_opts)
valid_dl_opts = copy.deepcopy(valid_dl_opts)
train_dl_opts.batch_size = 128
dev_dl_opts.batch_size = 128
valid_dl_opts.batch_size = 128

########################################################################################################################
# MODEL AND OPTIMIZER
########################################################################################################################
model_opts = model_options.VAEI3DOptions(
    latent_size=1024,
    dropout_prob=0.5,
    num_classes=ct.SMTH_NUM_CLASSES,
    vote_type='soft'
)
optimizer_opts = model_options.AdamOptimizerOptions(
    lr=0.001
)
########################################################################################################################
# TRAINER AND EVALUATOR
########################################################################################################################
mse_loss = metrics.AverageMeter(output_transform=lambda x: (x[-3], x[0].shape[0]))
ce_loss = metrics.AverageMeter(output_transform=lambda x: (x[-2], x[0].shape[0]))
kld_loss = metrics.AverageMeter(output_transform=lambda x: (x[-1], x[0].shape[0]))
total_loss = im.MetricsLambda(lambda x, y, z: x + y + z, mse_loss, ce_loss, kld_loss)
trainer_opts = model_options.TrainerOptions(
    epochs=50,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=criterion.VAECriterion,
    criterion_opts=model_options.VAECriterionOptions(
        mse_factor=1.0,
        ce_factor=1.0,
        kld_factor=0.01
    ),
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[1].squeeze(dim=1), x[6])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1].squeeze(dim=1), x[6])),
        'iou': metrics.MultiLabelIoU(lambda x: (x[1].squeeze(dim=1), x[6])),
        'mse_loss': mse_loss,
        'ce_loss': ce_loss,
        'kld_loss': kld_loss,
        'total_loss': total_loss
    }
)

vae_loss_metric = metrics.VAELoss(criterion.VAECriterion(**dc.asdict(trainer_opts.criterion_opts)))
total_loss = im.MetricsLambda(lambda x: sum(x), vae_loss_metric)
evaluator_opts = model_options.EvaluatorOptions(
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[-1], x[-2])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[-1], x[-2])),
        'iou': metrics.MultiLabelIoU(lambda x: (x[-1], x[-2])),
        'mse_loss': vae_loss_metric[0],
        'ce_loss': vae_loss_metric[1],
        'kld_loss': vae_loss_metric[2],
        'total_loss': total_loss,
    }
)
########################################################################################################################
# RUN
########################################################################################################################
vaei3d_smth_dev = model_options.RunOptions(
    name='vaei3d_smth_dev',
    mode='vae',
    resume=True,
    log_interval=1,
    patience=50,
    model=vae_i3d.VAEI3D,
    model_opts=model_opts,
    data_bunch=smth.SmthDataBunch,
    db_opts=db_opts,
    train_ds_opts=train_ds_opts,
    dev_ds_opts=dev_ds_opts,
    valid_ds_opts=valid_ds_opts,
    train_dl_opts=train_dl_opts,
    dev_dl_opts=dev_dl_opts,
    valid_dl_opts=valid_dl_opts,
    trainer_opts=trainer_opts,
    evaluator_opts=evaluator_opts
)
