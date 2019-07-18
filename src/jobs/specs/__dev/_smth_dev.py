import os

import constants as ct
from options import pipe_options

__all__ = [
    'db_opts',
    'train_ds_opts', 'train_dl_opts',
    'dev_ds_opts', 'dev_dl_opts',
    'valid_ds_opts', 'valid_dl_opts'
]

# TODO: CHANGE.

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
    cut=1.0,
    setting='eval',
    keep=0.008
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
    batch_size=32,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# DEV DATA
########################################################################################################################
dev_do = pipe_options.DataOptions(
    meta_path=ct.SMTH_META_TRAIN,
    cut=1.0,
    setting='eval',
    keep=0.008
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
    batch_size=32,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# VALID DATA
########################################################################################################################
valid_do = pipe_options.DataOptions(
    meta_path=ct.SMTH_META_TRAIN,
    cut=1.0,
    setting='eval',
    keep=0.008
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
    batch_size=32,
    shuffle=False,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
