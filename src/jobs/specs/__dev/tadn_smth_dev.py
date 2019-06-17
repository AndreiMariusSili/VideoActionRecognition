import copy

import ignite.metrics as im
from torch import nn, optim

import constants as ct
import pipeline.smth.databunch as smth
from jobs.specs.__dev._smth_dev import *
from models import metrics, tadn
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
model_opts = model_options.TADNOptions(
    num_classes=ct.SMTH_NUM_CLASSES,
    time_steps=4,
    growth_rate=128,
    drop_rate=0.0,
)
optimizer_opts = model_options.AdamOptimizerOptions(
    lr=0.001
)
########################################################################################################################
# TRAINER AND EVALUATOR
########################################################################################################################
trainer_opts = model_options.TrainerOptions(
    epochs=50,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=nn.CrossEntropyLoss,
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: x[1:3]),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: x[1:3]),
        'iou': metrics.MultiLabelIoU(lambda x: x[1:3]),
        'loss': im.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[1:3])
    }
)
evaluator_opts = model_options.EvaluatorOptions(
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: x[0:2]),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: x[0:2]),
        'iou': metrics.MultiLabelIoU(lambda x: x[0:2]),
        'loss': im.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[0:2])
    }
)
########################################################################################################################
# RUN
########################################################################################################################
tadn_smth_dev = model_options.RunOptions(
    name='tadn_smth_dev',
    mode='class',
    resume=False,
    log_interval=1,
    patience=25,
    model=tadn.TimeAlignedDenseNet,
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
