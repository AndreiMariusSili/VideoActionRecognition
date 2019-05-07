import os

from ignite import metrics
from torch import nn, optim

import constants as ct
import pipeline as pipe
from models import lrcn, options

########################################################################################################################
# DATA BUNCH OPTIONS
########################################################################################################################
db_opts = pipe.DataBunchOptions(
    shape='volume',
    frame_size=224
)
########################################################################################################################
# TRAIN DATA
########################################################################################################################
train_do = pipe.DataOptions(
    meta_path=ct.SMTH_META_TRAIN,
    cut=1.0,
    setting='train',
)
train_so = pipe.SamplingOptions(
    num_segments=4,
    segment_size=4
)
train_ds_opts = pipe.DataSetOptions(
    do=train_do,
    so=train_so
)
train_dl_opts = pipe.DataLoaderOptions(
    batch_size=8,
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
    cut=1.0,
    setting='valid',
)
valid_so = pipe.SamplingOptions(
    num_segments=4,
    segment_size=4
)
valid_ds_opts = pipe.DataSetOptions(
    do=valid_do,
    so=valid_so
)
valid_dl_opts = pipe.DataLoaderOptions(
    batch_size=8,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# MODEL AND AUXILIARIES
########################################################################################################################
model_opts = options.LRCNOptions(
    num_classes=ct.SMTH_NUM_CLASSES,
    freeze_features=False,
    freeze_fusion=False
)
optimizer_opts = options.AdamOptimizerOptions(
    lr=0.001
)
trainer_opts = options.TrainerOptions(
    epochs=100,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=nn.CrossEntropyLoss
)
evaluator_opts = options.EvaluatorOptions(
    metrics={
        'acc@1': metrics.Accuracy(),
        'acc@3': metrics.TopKCategoricalAccuracy(k=3),
        'loss': metrics.Loss(nn.CrossEntropyLoss())
    }
)
########################################################################################################################
# RUN
########################################################################################################################
lrcn_smth_100 = options.RunOptions(
    name=f'lrcn_smth_100',
    mode='discriminative',
    resume=False,
    log_interval=10,
    patience=5,
    model=lrcn.LRCN,
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
