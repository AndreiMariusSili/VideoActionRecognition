import bokeh.plotting as bop
import bokeh.models as bom
from typing import List
from glob import glob
from PIL import Image
import numpy as np

from pipeline import video_meta


class Video(object):
    paths: List[str]
    data: List[Image.Image]
    meta: video_meta.VideoMeta
    cut: int

    def __init__(self, meta: video_meta.VideoMeta, cut: float, sample_size: int = None):
        """Initialize a Video object from a row in the meta DataFrame."""
        assert 0.0 <= cut <= 1.0, f'Cut should be a value between 0.0 and 1.0. Received: {cut}.'

        self.meta = meta
        self.cut = int(round(self.meta.length * cut))
        self.sample_size = sample_size
        self.paths = self.__get_frame_paths()
        self.data = self.__get_frame_data()

    def __get_frame_paths(self):
        """Get a cut of the entire video."""
        all_frame_paths = np.array(sorted(glob(f'{self.meta.jpeg_path}/*.jpeg')))
        cut_frame_paths = all_frame_paths[0:self.cut]

        if self.sample_size is not None:
            sample_indices = np.linspace(0, self.cut, self.sample_size, False).round().astype(np.uint8)
            cut_frame_paths = cut_frame_paths[sample_indices]

        return cut_frame_paths

    def __get_frame_data(self):
        return [Image.open(path) for path in self.paths]

    def show(self, fig: bop.Figure, source: bom.ColumnDataSource) -> None:
        """Compile to a Bokeh animation object."""
        fig.image_rgba(image=self.meta.id, source=source, x=0, y=0, dw=224, dh=224)
        fig.tools = []
        fig.toolbar.logo = None
        fig.toolbar_location = None
        fig.axis.visible = False
        fig.xgrid.grid_line_color = None
        fig.ygrid.grid_line_color = None
        fig.outline_line_color = None

    def __str__(self):
        """Representation as (id, {dimensions})"""
        return f'Video {self.meta.id} ({"x".join(map(str, (len(self.data), *self.data[0].size)))})'

    def __repr__(self):
        return self.__str__()
