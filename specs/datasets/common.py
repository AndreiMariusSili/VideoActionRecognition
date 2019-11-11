"""Holds specifications common across datasets."""
import os

import options.data_options as po

########################################################################################################################
# DATA LOADER OPTIONS
########################################################################################################################
dlo = po.DataLoaderOptions(
    batch_size=128,
    shuffle=True,
    num_workers=os.cpu_count(),
    pin_memory=True,
    drop_last=False
)
########################################################################################################################
# SAMPLING OPTIONS
########################################################################################################################
so = po.SamplingOptions(
    num_segments=4,
    segment_size=1
)
