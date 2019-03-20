from torch import optim, nn, cuda
from ignite import metrics

from models import options
import pipeline as pipe
from models import i3d
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
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
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
    batch_size=2,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    drop_last=False
)
########################################################################################################################
# MODEL AND AUXILIARIES
########################################################################################################################
model_opts = options.I3DOptions(
    num_classes=10,
)
optimizer_opts = options.AdamOptimizerOptions(
    lr=0.01
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
dev_i3d_smth = options.RunOptions(
    name=f'dev@{ct.SETTING}@i3d@smth',
    resume=False,
    resume_from=None,
    log_interval=1,
    patience=5,
    model=i3d.I3D,
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