import numpy as np

from pipeline._video_meta import VideoMeta


class Label(object):
    data: int
    meta: VideoMeta

    def __init__(self, meta: VideoMeta):
        """Initialize a Label object from a row in the meta DataFrame and an integer identifier."""
        self.meta = meta
        self.data = meta.template_id

    def show(self):
        """Compile to a Bokeh animation object."""
        pass

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'({self.data} {self.meta.template})'

    def __repr__(self):
        return self.__str__()
