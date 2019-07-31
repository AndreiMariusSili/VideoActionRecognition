import os

from ignite import metrics
from torch import nn, optim

import constants as ct
from models import i3d
from options import model_options, pipe_options
from pipeline import smth

########################################################################################################################
# DATA BUNCH OPTIONS
########################################################################################################################
db_opts = pipe_options.DataBunchOptions(
    shape='volume',
    frame_size=224
)
########################################################################################################################
# TRAIN DATA
########################################################################################################################
train_do = pipe_options.DataOptions(
    meta_path=ct.SMTH_META_TRAIN,
    cut=0.50,
    setting='train',
)
train_so = pipe_options.SamplingOptions(
    num_segments=4,
    segment_size=1
)
train_ds_opts = pipe_options.DataSetOptions(
    do=train_do,
    so=train_so
)
train_dl_opts = pipe_options.DataLoaderOptions(
    batch_size=128,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# DEV DATA
########################################################################################################################
dev_do = pipe_options.DataOptions(
    meta_path=ct.SMTH_META_DEV,
    cut=0.50,
    setting='valid',
)
dev_so = pipe_options.SamplingOptions(
    num_segments=4,
    segment_size=1
)
dev_ds_opts = pipe_options.DataSetOptions(
    do=dev_do,
    so=dev_so
)
dev_dl_opts = pipe_options.DataLoaderOptions(
    batch_size=128,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# VALID DATA
########################################################################################################################
valid_do = pipe_options.DataOptions(
    meta_path=ct.SMTH_META_VALID,
    cut=0.50,
    setting='valid',
)
valid_so = pipe_options.SamplingOptions(
    num_segments=4,
    segment_size=1
)
valid_ds_opts = pipe_options.DataSetOptions(
    do=valid_do,
    so=valid_so
)
valid_dl_opts = pipe_options.DataLoaderOptions(
    batch_size=128,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# MODEL AND OPTIMIZER
########################################################################################################################
model_opts = model_options.I3DOptions(
    num_classes=ct.SMTH_NUM_CLASSES,
    dropout_prob=0.5
)
optimizer_opts = model_options.AdamOptimizerOptions(
    lr=0.001
)
########################################################################################################################
# TRAINER AND EVALUATOR
########################################################################################################################
trainer_opts = model_options.TrainerOptions(
    epochs=100,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=nn.CrossEntropyLoss,
    metrics={
        'acc@1': metrics.Accuracy(output_transform=lambda x: x[1:3]),
        'acc@2': metrics.TopKCategoricalAccuracy(k=2, output_transform=lambda x: x[1:3]),
        'loss': metrics.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[1:3])
    }
)

evaluator_opts = model_options.EvaluatorOptions(
    metrics={
        'acc@1': metrics.Accuracy(output_transform=lambda x: x[0:2]),
        'acc@2': metrics.TopKCategoricalAccuracy(k=2, output_transform=lambda x: x[0:2]),
        'loss': metrics.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[0:2])
    }
)
########################################################################################################################
# RUN
########################################################################################################################
i3d_smth_050 = model_options.RunOptions(
    name=f'i3d_smth_050',
    mode='class',
    resume=False,
    log_interval=10,
    model=i3d.I3D,
    model_opts=model_opts,
    data_bunch=smth.SmthDataBunch,
    db_opts=db_opts,
    train_ds_opts=train_ds_opts,
    dev_ds_opts=dev_ds_opts,
    valid_ds_opts=dev_ds_opts,
    train_dl_opts=train_dl_opts,
    dev_dl_opts=dev_dl_opts,
    valid_dl_opts=dev_dl_opts,
    trainer_opts=trainer_opts,
    evaluator_opts=evaluator_opts
)
