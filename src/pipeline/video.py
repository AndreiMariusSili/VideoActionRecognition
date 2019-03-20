import bokeh.plotting as bop
import bokeh.models as bom
from typing import List, Optional
from glob import glob
from PIL import Image
import numpy as np

from pipeline import video_meta


class Video(object):
    meta: video_meta.VideoMeta
    cut: int
    setting: str
    num_segments: int
    segment_sample_size: int
    paths: List[str]
    data: List[Image.Image]

    def __init__(self, meta: video_meta.VideoMeta, cut: float, setting: str,
                 num_segments: Optional[int], segment_sample_size: Optional[int]):
        """Initialize a Video object from a row in the meta DataFrame."""
        assert 0.0 <= cut <= 1.0, f'Cut should be a value between 0.0 and 1.0. Received: {cut}.'
        assert setting in ['train', 'valid'], f'setting should be either "train" or "valid".'
        assert bool(num_segments) == bool(segment_sample_size), 'Specify both number of segments and segment size.'

        self.meta = meta
        self.cut = int(round(self.meta.length * cut))
        self.setting = setting
        self.num_segments = num_segments
        self.segment_sample_size = segment_sample_size
        self.paths = self.__get_frame_paths()
        self.data = self.__get_frame_data()

    def __get_frame_paths(self):
        """Get a cut of the entire video."""
        all_frame_paths = np.array(sorted(glob(f'{self.meta.jpeg_path}/*.jpeg')))
        cut_frame_paths = all_frame_paths[0:self.cut]

        if self.num_segments is not None:
            if self.setting == 'train':
                cut_frame_paths = self.__random_sample_segments(cut_frame_paths)
            else:
                cut_frame_paths = self.__fixed_sample_segments(cut_frame_paths)

        return cut_frame_paths

    def __random_sample_segments(self, cut_frame_paths: np.ndarray):
        """Split the video into segments and uniformly random sample from each segment. If the segment is smaller than
        the sample size, sample with replacement to duplicate some frames."""
        segments = np.array_split(cut_frame_paths, self.num_segments)
        segments = [segment for segment in segments if segment.size > 0]
        sample = []
        for segment in segments:
            try:
                sample.append(np.sort(np.random.choice(segment, self.segment_sample_size, replace=False)))
            except ValueError:
                sample.append(np.sort(np.random.choice(segment, self.segment_sample_size, replace=True)))

        return np.array(sample).reshape(-1)

    def __fixed_sample_segments(self, cut_frame_paths: np.ndarray):
        """Sample frames at roughly equal intervals from the video.
        If the video is too short, some frames will be duplicated."""
        size = self.segment_sample_size * self.num_segments
        # self.cut -1 to prevent rounding leading to out of bounds index.
        sample_indices = np.linspace(0, self.cut - 1, size, True).round().astype(np.uint8)

        return cut_frame_paths[sample_indices]

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
