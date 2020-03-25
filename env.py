import logging
import random

import numpy as np
import torch as th

import constants as ct

th.manual_seed(ct.RANDOM_STATE)
np.random.seed(ct.RANDOM_STATE)
random.seed(ct.RANDOM_STATE)

logging.basicConfig(level='INFO', format='%(message)s')
logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)
LOGGER = logging.getLogger()
