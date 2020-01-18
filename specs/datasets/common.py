import torch.multiprocessing as mp

import options.data_options as po

########################################################################################################################
# DATA LOADER
########################################################################################################################
dlo = po.DataLoaderOptions(
    batch_size=None,
    shuffle=True,
    num_workers=mp.cpu_count(),
    pin_memory=True,
    drop_last=False,
    timeout=30
)
########################################################################################################################
# SAMPLING
########################################################################################################################
so_4 = po.SamplingOptions(
    num_segments=4,
)
so_16 = po.SamplingOptions(
    num_segments=16,
)
