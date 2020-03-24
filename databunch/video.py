import glob
import pathlib as pl
import typing as t

import cv2
import cv2.optflow as optf  # noqa
import imgaug.augmenters as ia
import numpy as np
import torch as th
import torchvision.transforms.functional as F  # noqa

import databunch.video_meta as vm


class Video(object):
    def __init__(self, meta: vm.VideoMeta, root_path: pl.Path, read_jpeg: bool,
                 cut: float, setting: str, num_segments: int, flow: bool, aug_seq: ia.Sequential):
        assert 0.0 <= cut <= 1.0, f'Cut should be a value between 0.0 and 1.0. Received: {cut}.'
        assert setting in ['train', 'eval'], f'Unknown setting: {setting}.'

        self.meta = meta
        self.root_path = root_path
        self.read_jpeg = read_jpeg
        self.cut = int(round(self.meta.length * cut)) - 1
        self.setting = setting
        self.num_segments = num_segments
        self.flow = flow
        self.aug_seq = aug_seq

        self.flow_algo = cv2.optflow.createOptFlow_Farneback()
        self.cut_locs = self._cut_locs()
        self.subsample_locs = self._subsample_locs()
        self.data = self._get_data()
        self.data = self._do_augment()
        self.recon = self._get_recon()

    def _get_data(self):
        if self.read_jpeg:
            data = self._image_data()
        else:
            data = self._video_data()

        return data

    def _do_augment(self):
        det_aug_seq = self.aug_seq.to_deterministic()

        return [det_aug_seq.augment_image(frame) for frame in self.data]

    def _get_recon(self):
        data = ia.Resize(56)(images=self.data)
        recon = self._flow_data(data) if self.flow else data

        return recon

    def _cut_locs(self) -> np.ndarray:
        all_locs = np.arange(self.meta.length)  # disregard first and last frames, as they may be blank.
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

    def _flow_data(self, data: t.List[np.ndarray]) -> t.List[np.ndarray]:
        _flow = []
        for i in range(len(data) - 1):
            first, second = data[i], data[i + 1]
            first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
            second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
            frame_flow = np.clip(self.flow_algo.calc(first, second, None), -20, 20)
            _flow.append(frame_flow)

        return _flow

    def to_tensor(self):
        self.data = [F.to_tensor(frame) for frame in self.data]
        self.recon = [F.to_tensor(flow) for flow in self.recon]

    def to_numpy(self):
        if isinstance(self.data, list):
            for i in range(len(self.data)):
                if isinstance(self.data[i], th.Tensor):
                    self.data[i] = self.data[i].cpu().numpy()
        if isinstance(self.recon, list):
            for i in range(len(self.recon)):
                if isinstance(self.recon[i], th.Tensor):
                    self.recon[i] = self.recon[i].cpu().numpy()  # noqa
        self.data = np.transpose(np.stack(self.data, axis=0), axes=(0, 2, 3, 1))
        self.recon = np.transpose(np.stack(self.recon, axis=0), axes=(0, 2, 3, 1))

    def __str__(self):
        return f'Video {self.meta.id} ({"x".join(map(str, (len(self.data), *self.data[0].shape)))})'

    def __repr__(self):
        return self.__str__()
