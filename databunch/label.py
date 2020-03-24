import torch as th

import databunch.video_meta as pim


class Label(object):
    def __init__(self, meta: pim.VideoMeta):
        """Initialize a Label object from a row in the meta DataFrame and an integer identifier."""
        self.meta = meta
        self.data = meta.lid

    def to_tensor(self):
        self.data = th.tensor(self.data, dtype=th.int64)

    def to_numpy(self):
        self.data = self.data.numpy()

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'({self.data} {self.meta.label})'

    def __repr__(self):
        return self.__str__()
