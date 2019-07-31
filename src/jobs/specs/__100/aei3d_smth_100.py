import copy

import ignite.metrics as im
from torch import optim

import constants as ct
import pipeline.smth.databunch as smth
from jobs.specs.__100._smth_100 import *
from models import ae_i3d, criterion, metrics
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
model_opts = model_options.AEI3DOptions(
    embed_planes=1024,
    dropout_prob=0.0,
    num_classes=ct.SMTH_NUM_CLASSES
)
optimizer_opts = model_options.AdamOptimizerOptions(
    lr=0.001
)
########################################################################################################################
# TRAINER AND EVALUATOR
########################################################################################################################
ce_loss = metrics.AverageMeter(output_transform=lambda x: (x[-2], x[0].shape[0]))
mse_loss = metrics.AverageMeter(output_transform=lambda x: (x[-1], x[0].shape[0]))
total_loss = im.MetricsLambda(lambda x, y: x + y, ce_loss, mse_loss)
trainer_opts = model_options.TrainerOptions(
    epochs=100,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=criterion.AECriterion,
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[1], x[4])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[4])),
        'iou': metrics.MultiLabelIoU(lambda x: (x[1], x[4])),
        'ce_loss': ce_loss,
        'mse_loss': mse_loss,
        'total_loss': total_loss
    }
)
ae_loss_metric = metrics.AELoss(criterion.AECriterion())
total_loss = im.MetricsLambda(lambda x: sum(x), ae_loss_metric)
evaluator_opts = model_options.EvaluatorOptions(
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[1], x[4])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[4])),
        'iou': metrics.MultiLabelIoU(lambda x: (x[1], x[4])),
        'ce_loss': ae_loss_metric[0],
        'mse_loss': ae_loss_metric[1],
        'total_loss': total_loss,
    }
)
########################################################################################################################
# RUN
########################################################################################################################
aei3d_smth_100 = model_options.RunOptions(
    name='aei3d_smth_100',
    mode='ae',
    resume=False,
    debug=False,
    log_interval=10,
    model=ae_i3d.AEI3D,
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
