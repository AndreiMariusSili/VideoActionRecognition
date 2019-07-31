import copy

import ignite.metrics as im
from torch import optim

import constants as ct
import pipeline.smth.databunch as smth
from jobs.specs.__100._smth_100 import *
from models import criterion, metrics, vae_tarn
from options import model_options

train_dl_opts = copy.deepcopy(train_dl_opts)
dev_dl_opts = copy.deepcopy(dev_dl_opts)
valid_dl_opts = copy.deepcopy(valid_dl_opts)
train_dl_opts.batch_size = 256
dev_dl_opts.batch_size = 256
valid_dl_opts.batch_size = 256

########################################################################################################################
# MODEL AND OPTIMIZER
########################################################################################################################
model_opts = model_options.VAETARNOptions(
    time_steps=4,
    drop_rate=0.0,
    num_classes=ct.SMTH_NUM_CLASSES,
    vote_type='soft',
    encoder_planes=(16, 32, 64, 128, 256),
    decoder_planes=(256, 128, 64, 32, 16),
)
optimizer_opts = model_options.AdamOptimizerOptions(
    lr=0.001
)
########################################################################################################################
# TRAINER AND EVALUATOR
########################################################################################################################
ce_loss = metrics.AverageMeter(output_transform=lambda x: (x[-4], x[0].shape[0]))
mse_loss = metrics.AverageMeter(output_transform=lambda x: (x[-3], x[0].shape[0]))
kld_loss = metrics.AverageMeter(output_transform=lambda x: (x[-2], x[0].shape[0]))
kld_factor = metrics.AverageMeter(output_transform=lambda x: (x[-1], 1))
total_loss = im.MetricsLambda(lambda x, y, z: x + y + z, ce_loss, mse_loss, kld_loss)
trainer_opts = model_options.TrainerOptions(
    epochs=100,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=criterion.VAECriterion,
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[1].squeeze(dim=1), x[6])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1].squeeze(dim=1), x[6])),
        'iou': metrics.MultiLabelIoU(lambda x: (x[1].squeeze(dim=1), x[6])),
        'ce_loss': ce_loss,
        'mse_loss': mse_loss,
        'kld_loss': kld_loss,
        'total_loss': total_loss,
        'kld_factor': kld_factor,
    }
)
vae_loss_metric = metrics.VAELoss(criterion.VAECriterion())
total_loss = im.MetricsLambda(lambda x: sum(x), vae_loss_metric)
evaluator_opts = model_options.EvaluatorOptions(
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[-1], x[-2])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[-1], x[-2])),
        'iou': metrics.MultiLabelIoU(output_transform=lambda x: (x[-1], x[-2])),
        'ce_loss': vae_loss_metric[0],
        'mse_loss': vae_loss_metric[1],
        'kld_loss': vae_loss_metric[2],
        'total_loss': total_loss,
    }
)
########################################################################################################################
# RUN
########################################################################################################################
vaetarn_smth_100 = model_options.RunOptions(
    name='vaetarn_smth_100',
    mode='vae',
    resume=False,
    debug=False,
    log_interval=10,
    model=vae_tarn.VAETimeAlignedResNet,
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
