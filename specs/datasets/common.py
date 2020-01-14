"""Holds specifications common across datasets."""
import torch.multiprocessing as tmp

import options.data_options as po

########################################################################################################################
# DATA LOADER
########################################################################################################################
dlo = po.DataLoaderOptions(
    batch_size=None,
    shuffle=True,
    num_workers=tmp.cpu_count(),
    pin_memory=False,
    drop_last=False
)
########################################################################################################################
# SAMPLING
########################################################################################################################
so_small = po.SamplingOptions(
    num_segments=4,
    segment_size=1
)
so_large = po.SamplingOptions(
    num_segments=8,
    segment_size=2
)
