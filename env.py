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

# #   Monkey patch the ignite framework to allow dataclass serialisation.
# def __safe_getattr__(self, attr):
#     if attr == '__dataclass_fields__':
#         raise AttributeError
#
#     from ignite.metrics import MetricsLambda
#
#     def func(x, *args, **kwargs):
#         return getattr(x, attr)(*args, **kwargs)
#
#     def wrapper(*args, **kwargs):
#         return MetricsLambda(func, self, *args, **kwargs)
#
#     return wrapper
#
#
# ig.metrics.Metric.__getattr__ = __safe_getattr__
