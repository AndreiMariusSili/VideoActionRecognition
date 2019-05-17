import logging

import numpy as np
import torch as th

logging.basicConfig(level='INFO', format='[%(asctime)-s][%(process)d][%(levelname)s]\t%(message)s')
th.manual_seed(0)
np.random.seed(0)
