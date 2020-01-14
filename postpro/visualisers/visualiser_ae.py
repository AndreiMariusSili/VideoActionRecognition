import typing as t

import numpy as np
import torch as th

import databunch.video as dv
import postpro.visualisers._visualiser_base as _base


class AEVisualiser(_base.BaseVisualiser):
    def preds(self, video: dv.Video) -> t.List[th.Tensor]:
        video_data = th.tensor(video.data[np.newaxis, :, :, :], dtype=th.float32, device=self.device)
        _, pred, *_ = self.model(video_data)

        return [pred.squeeze()]

    def recons(self, video: dv.Video) -> t.List[th.Tensor]:
        video_data = th.tensor(video.data[np.newaxis, :, :, :], dtype=th.float32, device=self.device)
        recon, *_ = self.model(video_data)

        return [recon.squeeze()]
