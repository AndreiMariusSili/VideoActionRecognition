import typing as t

import ignite.engine as ie
import numpy as np
import sklearn.decomposition as skd
import sklearn.manifold as skm
import torch as th
import torch.nn.functional as func

import constants as ct
import options.experiment_options as eo
import postpro.evaluators._evaluator_base as _base
import pro.engine as pe
import specs.maps as sm


class GSNNEvaluator(_base.BaseEvaluator):
    def __init__(self, opts: eo.ExperimentOptions, local_rank: int):
        super(GSNNEvaluator, self).__init__(opts, local_rank)
        assert self.opts.model.type == 'gsnn'

    def _init_evaluators(self) -> t.Tuple[ie.Engine, ie.Engine, ie.Engine]:
        evaluator_metrics = sm.Metrics[self.opts.evaluator.metrics].value

        train_evaluator = pe.create_gsnn_evaluator(self.model, evaluator_metrics, self.device, ct.VAE_NUM_SAMPLES_TEST)
        dev_evaluator = pe.create_gsnn_evaluator(self.model, evaluator_metrics, self.device, ct.VAE_NUM_SAMPLES_TEST)
        test_evaluator = pe.create_gsnn_evaluator(self.model, evaluator_metrics, self.device, ct.VAE_NUM_SAMPLES_TEST)

        return train_evaluator, dev_evaluator, test_evaluator

    def _get_model_outputs(self, x: th.Tensor) -> t.Dict[str, th.Tensor]:
        """Get outputs of interest from a model."""
        energy, temporal_embed, class_embed, mean, var, vote = self.model(x, ct.VAE_NUM_SAMPLES_TEST)
        conf = func.softmax(energy, dim=-1)

        return {
            'temporal_embeds': temporal_embed.cpu().numpy(),
            'class_embeds': class_embed.cpu().numpy(),
            'confs': conf.cpu().numpy(),
            'votes': vote.cpu().numpy()
        }

    def _get_projections(self, embed_name: str, sample_embeds: np.ndarray) -> np.ndarray:
        """Get tsne projections."""
        if embed_name == 'class_embeds':
            n, s, c = sample_embeds.shape
            sample_embeds = sample_embeds.reshape([n * s, c])
            pca_ndim = min(n, c, 64)
            sample_pca = skd.PCA(pca_ndim, random_state=ct.RANDOM_STATE).fit_transform(sample_embeds)
            sample_tsne = skm.TSNE(2, verbose=1, random_state=ct.RANDOM_STATE).fit_transform(sample_pca)
            sample_tsne = sample_tsne.reshape([n, s, 2])
        elif embed_name == 'temporal_embeds':
            if sample_embeds.ndim == 6:  # sequence models
                n, s, _t, c, h, w = sample_embeds.shape
                sample_embeds = sample_embeds.reshape([n * s, _t, c * h * w])
            else:  # hierarchical models
                _t = 1
                n, s, c, h, w = sample_embeds.shape
                sample_embeds = sample_embeds.reshape([n * s, _t, c * h * w])
            pca_ndim = min(n, c * h * w, 64)
            sample_pca = np.empty([n * s, _t, pca_ndim])
            sample_tsne = np.empty([n * s, _t, 2])
            for i in range(_t):
                sample_pca[:, i, :] = skd.PCA(pca_ndim, random_state=ct.RANDOM_STATE).fit_transform(
                    sample_embeds[:, i, :])
                sample_tsne[:, i, :] = skm.TSNE(2, verbose=1, random_state=ct.RANDOM_STATE) \
                    .fit_transform(sample_pca[:, i, :])
            sample_tsne = sample_tsne.reshape([n, s, _t, 2])
        else:
            raise ValueError(f'Unknown embed name: {embed_name}.')

        return sample_tsne
