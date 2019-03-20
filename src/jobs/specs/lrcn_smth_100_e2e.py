from torch import optim, nn
from ignite import metrics
import os

from models import options
import pipeline as pipe
from models import lrcn
import constants as ct


########################################################################################################################
# DATA BUNCH OPTIONS
########################################################################################################################
data_bunch_opts = pipe.DataBunchOptions(
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
train_data_set_opts = pipe.DataSetOptions(
    do=train_do,
    so=train_so
)
train_data_loader_opts = pipe.DataLoaderOptions(
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
valid_data_set_opts = pipe.DataSetOptions(
    do=valid_do,
    so=valid_so
)
valid_data_loader_opts = pipe.DataLoaderOptions(
    batch_size=16,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# MODEL AND AUXILIARIES
########################################################################################################################
model_opts = options.LRCNOptions(
    num_classes=10,
    freeze_feature_extractor=False
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
lrcn_smth_100_e2e = options.RunOptions(
    name=f'{ct.SETTING}_lrcn_smth_100_e2e',
    resume=False,
    resume_from=None,
    log_interval=10,
    patience=5,
    model=lrcn.LRCN,
    model_opts=model_opts,
    data_bunch=pipe.SmthDataBunch,
    data_bunch_opts=data_bunch_opts,
    train_data_set_opts=train_data_set_opts,
    valid_data_set_opts=valid_data_set_opts,
    train_data_loader_opts=train_data_loader_opts,
    valid_data_loader_opts=valid_data_loader_opts,
    trainer_opts=trainer_opts,
    evaluator_opts=evaluator_opts
)
