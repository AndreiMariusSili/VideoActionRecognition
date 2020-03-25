import typing as t

import numpy as np
import torch as th

import constants as ct
import databunch.video as dv
import postpro.visualisers._visualiser_base as _base


class VAEVisualiser(_base.BaseVisualiser):
    def preds(self, video: dv.Video) -> t.List[th.Tensor]:
        video_data = th.tensor(video.data[np.newaxis, :, :, :], dtype=th.float32, device=self.device)
        _, preds, *_ = self.model(video_data, ct.VAE_NUM_SAMPLES_TEST)
        return [pred.squeeze() for pred in th.split(preds, 1, dim=1)]

    def recons(self, video: dv.Video) -> t.List[th.Tensor]:
        video_data = th.tensor(video.data[np.newaxis, :, :, :], dtype=th.float32, device=self.device)
        recons, *_ = self.model(video_data, True, ct.VAE_NUM_SAMPLES_TEST)

        return [recon.squeeze() for recon in th.split(recons, 1, dim=1)]
