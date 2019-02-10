import numpy as np
import skvideo.io

from pipeline import _video_meta


class Video(object):
    data: np.ndarray
    meta: _video_meta.VideoMeta
    cut: int

    def __init__(self, meta: _video_meta.VideoMeta, cut: float):
        """Initialize a Video object from a row in the meta DataFrame."""
        assert 0.0 <= cut <= 1.0, f'Cut should be a value between 0.0 and 1.0. Received: {cut}.'

        self.meta = meta
        self.cut = int(self.meta.length * cut)
        self.data = skvideo.io.vread(meta.path, num_frames=self.cut)

    def show(self):
        """Compile to a Bokeh animation object."""
        pass

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'{self.meta.id} ({"x".join(map(str, self.data.shape))})'

    def __repr__(self):
        return self.__str__()
