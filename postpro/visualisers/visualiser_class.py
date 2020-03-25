import typing as t

import numpy as np
import torch as th

import databunch.video as dv
import postpro.visualisers._visualiser_base as _base


class ClassVisualiser(_base.BaseVisualiser):
    def recons(self, video: np.ndarray) -> t.List[np.ndarray]:
        raise RuntimeError('Reconstruction not available for classification visualiser.')

    def preds(self, video: dv.Video) -> t.List[th.Tensor]:
        video_data = th.tensor(video.data[np.newaxis, :, :, :], dtype=th.float32, device=self.device)
        pred, *_ = self.model(video_data)

        return [pred.squeeze()]
