import pathlib as pl
from glob import glob
from typing import List, Optional, Union

import bokeh.models as bom
import bokeh.plotting as bop
import numpy as np
from PIL import Image
from skvideo import io

from databunch import video_meta


class Video(object):
    meta: video_meta.VideoMeta
    cut: int
    setting: str
    num_segments: int
    segment_sample_size: int
    indices: np.ndarray
    data: Union[List[Image.Image], np.ndarray]

    def __init__(self, meta: video_meta.VideoMeta, root_path: pl.Path, read_jpeg: bool, cut: float, setting: str,
                 num_segments: Optional[int], segment_sample_size: Optional[int]):
        """Initialize a Video object from a row in the meta DataFrame."""
        assert 0.0 <= cut <= 1.0, f'Cut should be a value between 0.0 and 1.0. Received: {cut}.'
        assert setting in ['train', 'eval'], f'Unknown setting: {setting}.'
        assert bool(num_segments) == bool(segment_sample_size), 'Specify both number of segments and segment size.'

        self.meta = meta
        self.root_path = root_path
        self.read_jpeg = read_jpeg
        self.cut = int(round(self.meta.length * cut))
        self.setting = setting
        self.num_segments = num_segments
        self.segment_sample_size = segment_sample_size

        if self.num_segments is not None and self.num_segments > self.cut:
            self.num_segments = self.cut

        self.indices = self.__get_frame_indices()
        self.data = self.__get_frame_data()

    def __get_frame_indices(self):
        """Get a cut of the entire video."""
        if self.read_jpeg:
            path = self.root_path / self.meta.image_path
            all_frames = np.array(sorted(glob(f'{path}/*.jpeg')))
            cut_frames = all_frames[0:self.cut]
        else:
            all_frames = np.arange(self.meta.length, dtype=np.int)
            cut_frames = all_frames[0:self.cut]

        if self.num_segments is not None:
            if self.setting == 'train':
                cut_frames = self.__random_sample_segments(cut_frames)
            else:
                cut_frames = self.__fixed_sample_segments(cut_frames)

        return cut_frames

    def __random_sample_segments(self, cut_frame_indices: np.ndarray):
        """Split the video into segments and uniformly random sample from each segment. If the segment is smaller than
        the sample size, sample with replacement to duplicate some frames."""
        segments = np.array_split(cut_frame_indices, self.num_segments)
        segments = [segment for segment in segments if segment.size > 0]
        sample = []
        for segment in segments:
            try:
                sample.append(np.sort(np.random.choice(segment, self.segment_sample_size, replace=False)))
            except ValueError:
                sample.append(np.sort(np.random.choice(segment, self.segment_sample_size, replace=True)))

        return np.array(sample).reshape(-1)

    def __fixed_sample_segments(self, cut_frame_paths: np.ndarray):
        """Sample frames at roughly equal intervals from the video. If the video is too short, some frames will
        be duplicated. self.cut -1 to prevent rounding leading to out of bounds index."""
        size = self.segment_sample_size * self.num_segments
        sample_indices = np.linspace(0, self.cut - 1, size, True).round().astype(np.int)

        return cut_frame_paths[sample_indices]

    def __get_frame_data(self):
        if self.read_jpeg:
            return [Image.open(path) for path in self.indices]
        else:
            video_path = self.root_path / self.meta.video_path
            video = io.vread(video_path.as_posix(), num_frames=self.cut)
            return [Image.fromarray(frame) for frame in video[self.indices]]

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
        if isinstance(self.data, np.ndarray):
            return f'Video {self.meta.id} ({"x".join(map(str, (len(self.data), *self.data[0].shape)))})'
        else:
            return f'Video {self.meta.id} ({"x".join(map(str, (len(self.data), *self.data[0].size)))})'

    def __repr__(self):
        return self.__str__()
