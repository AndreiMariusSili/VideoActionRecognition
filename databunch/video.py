import glob
import pathlib as pl
import typing as t

import cv2
import numpy as np

import databunch.video_meta as vm


class Video(object):
    def __init__(self, meta: vm.VideoMeta, root_path: pl.Path, read_jpeg: bool,
                 cut: float, setting: str, num_segments: int):
        assert 0.0 <= cut <= 1.0, f'Cut should be a value between 0.0 and 1.0. Received: {cut}.'
        assert setting in ['train', 'eval'], f'Unknown setting: {setting}.'

        self.meta = meta
        self.root_path = root_path
        self.read_jpeg = read_jpeg
        self.cut = int(round(self.meta.length * cut)) - 1
        self.setting = setting
        self.num_segments = num_segments

        self.cut_locs = self._cut_locs()
        self.subsample_locs = self._subsample_locs()
        if self.read_jpeg:
            self.data = self._image_data()
        else:
            self.data = self._video_data()

    def _cut_locs(self) -> np.ndarray:
        all_locs = np.arange(self.meta.length)
        cut_locs = all_locs[0:self.cut]

        return np.array(cut_locs)

    def _subsample_locs(self) -> np.ndarray:
        if self.setting == 'train':
            subsample_locs = self._random_subsample()
        else:
            subsample_locs = self._fixed_subsample()

        return subsample_locs

    def _random_subsample(self) -> np.ndarray:
        segments = [segment for segment in np.array_split(self.cut_locs, self.num_segments) if segment.size > 0]
        segments = self._pad_with_last_segment(segments)
        sample = [np.random.choice(segment, replace=False) for segment in segments]

        return np.array(sample)

    def _fixed_subsample(self) -> np.ndarray:
        segments = [segment for segment in np.array_split(self.cut_locs, self.num_segments) if segment.size > 0]
        segments = self._pad_with_last_segment(segments)
        sample = [segment[len(segment) // 2] for segment in segments]

        return np.array(sample)

    def _pad_with_last_segment(self, segments: t.List[np.ndarray]) -> t.List[np.ndarray]:
        segment_padding = self.num_segments - len(segments)

        return segments + [segments[-1]] * segment_padding

    def _image_data(self) -> t.List[np.ndarray]:
        dir_path = glob.escape((self.root_path / self.meta.image_path).as_posix())
        paths = np.sort(np.array(glob.glob(f'{dir_path}/*.jpeg')))
        subsample_paths = paths[self.subsample_locs]
        data = []
        for path in subsample_paths:
            data.append(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))

        return data

    def _video_data(self) -> t.List[np.ndarray]:
        cap = cv2.VideoCapture(str(self.root_path / self.meta.video_path))
        current_frame_loc = 0
        data = []
        for subsample_loc in self.subsample_locs:
            while current_frame_loc < subsample_loc:
                current_frame_loc += 1
                cap.grab()
            current_frame_loc += 1
            data.append(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
        cap.release()

        return data

    def __str__(self):
        return f'Video {self.meta.id} ({"x".join(map(str, (len(self.data), *self.data[0].shape)))})'

    def __repr__(self):
        return self.__str__()
