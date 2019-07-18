import copy
import dataclasses as dc
import pathlib as pl
import pickle as pkl
import traceback
from glob import glob
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import sklearn.manifold as skm
import torch as th
from ignite import engine
from torch import cuda, distributed, nn

import constants as ct
import helpers as hp
import models.engine as me
import options.model_options as mo
import pipeline.smth.databunch as smth
import pipeline.transforms as pit
from env import logging

NORMALIZE = pit.TensorNormalize(255)
STANDARDIZE = pit.TensorStandardize(ct.IMAGE_NET_MEANS, ct.IMAGE_NET_STDS)


class Evaluation(object):
    local_rank: int
    mode: str
    valid_metrics: dict
    train_metrics: dict
    metrics: dict
    run_dir: pl.Path
    run_opts: mo.RunOptions
    rank: int
    world_size: int
    main_proc: bool
    distributed: bool
    device: Any
    best_ckpt: str
    model: nn.Module
    data_bunch: smth.SmthDataBunch
    train_results: pd.DataFrame
    valid_results: pd.DataFrame
    results: pd.DataFrame

    def __init__(self, spec: mo.RunOptions, local_rank: int):
        self.local_rank = local_rank
        self.mode = spec.mode

        self.train_metrics = spec.evaluator_opts.metrics
        self.dev_metrics = spec.evaluator_opts.metrics
        self.valid_metrics = spec.evaluator_opts.metrics

        self.name = spec.name

        self.cut = f'__{self.name.split("_").pop()}'
        self.run_dir = ct.SMTH_RUN_DIR / self.cut / self.name
        self.run_opts = copy.deepcopy(spec)

        self.rank, self.world_size, self.main_proc, self.distributed = self._init_distributed()
        self.logger = self._init_logging()

        self.device = self._init_device()
        self.best_ckpt = self._get_best_ckpt()
        self.model = self._init_model()
        self.train_evaluator, self.dev_evaluator, self.valid_evaluator = self._init_evaluator()
        self._load_data_bunch()
        self.verbose = True

        self.metrics = {
            'train_acc@1': [],
            'train_acc@5': [],
            'train_iou': [],
            'dev_acc@1': [],
            'dev_acc@5': [],
            'dev_iou': [],
            'valid_acc@1': [],
            'valid_acc@5': [],
            'valid_iou': [],
        }
        if (ct.SMTH_RUN_DIR / self.cut / 'results.json').exists():
            self.results = pd.read_json((ct.SMTH_RUN_DIR / self.cut / 'results.json').as_posix(), orient='index')
        else:
            self.results = pd.DataFrame(None, None, ['train_acc@1', 'train_acc@5', 'train_iou',
                                                     'dev_acc@1', 'dev_acc@5', 'dev_iou',
                                                     'valid_acc@1', 'valid_acc@5', 'valid_iou'])

    def _init_distributed(self) -> Tuple[int, int, bool, bool]:
        if distributed.is_available() and self.local_rank != -1:
            if cuda.is_available():
                distributed.init_process_group(backend='nccl', init_method='env://')
            else:
                distributed.init_process_group(backend='gloo', init_method='env://')
            rank = distributed.get_rank()
            world_size = distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size, rank == 0, world_size > 0

    def _init_logging(self):
        """Get a logger that outputs to console and file. Only log messages if on the main process."""
        logger = logging.getLogger()
        if not self.main_proc:
            logger.handlers = []

        return logger

    def _init_device(self) -> Any:
        """Set the device according to CPU vs. single process CUDA vs. distributed CUDA."""
        if self.world_size > 1:
            device = th.device(f'cuda:{self.rank}' if cuda.is_available() else f'cpu')
        else:
            device = th.device(f'cuda' if cuda.is_available() else f'cpu')

        return device

    def _get_best_ckpt(self) -> str:
        """Get the path to the best checkpoint."""
        best_models = glob((self.run_dir / 'best_model_*.pth').as_posix())
        if len(best_models) > 1:
            raise ValueError('More than one best model available. Remove old versions.')
        return best_models.pop()

    def _init_model(self) -> nn.Module:
        """Load the state dict of a model and sends to available device(s)."""
        model = self.run_opts.model(**dc.asdict(self.run_opts.model_opts))
        model.eval()
        model.to(self.device)
        model.load_state_dict(th.load(self.best_ckpt, map_location=self.device))
        if distributed.is_available() and self.local_rank != -1:
            if cuda.is_available():
                if self.world_size > 1:
                    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank)
                else:
                    model = nn.parallel.DistributedDataParallel(model)
            else:
                model = nn.parallel.DistributedDataParallelCPU(model)

        return model

    def _init_evaluator(self) -> Tuple[engine.Engine, engine.Engine, engine.Engine]:
        """Initialize the trainer and evaluator engines."""
        if self.mode == 'class':
            train_evaluator = me.create_cls_evaluator(self.model, self.train_metrics, self.device, True)
            dev_evaluator = me.create_cls_evaluator(self.model, self.dev_metrics, self.device, True)
            valid_evaluator = me.create_cls_evaluator(self.model, self.valid_metrics, self.device, True)
        elif self.mode == 'ae':
            train_evaluator = me.create_ae_evaluator(self.model, self.train_metrics, self.device, True)
            dev_evaluator = me.create_ae_evaluator(self.model, self.dev_metrics, self.device, True)
            valid_evaluator = me.create_ae_evaluator(self.model, self.valid_metrics, self.device, True)
        else:
            train_evaluator = me.create_vae_evaluator(self.model, self.train_metrics, self.device, True)
            dev_evaluator = me.create_vae_evaluator(self.model, self.dev_metrics, self.device, True)
            valid_evaluator = me.create_vae_evaluator(self.model, self.valid_metrics, self.device, True)

        return train_evaluator, dev_evaluator, valid_evaluator

    def _load_data_bunch(self) -> None:
        """Load the data bunch."""

        # flag use of distributed sampler
        self.run_opts.db_opts.distributed = self.world_size > 1

        # make sure batch and workers are distributed well across worlds.
        self.run_opts.train_dl_opts.batch_size = self.run_opts.train_dl_opts.batch_size // self.world_size
        self.run_opts.train_dl_opts.num_workers = self.run_opts.train_dl_opts.num_workers // self.world_size
        self.run_opts.dev_dl_opts.batch_size = self.run_opts.dev_dl_opts.batch_size // self.world_size
        self.run_opts.dev_dl_opts.num_workers = self.run_opts.dev_dl_opts.num_workers // self.world_size
        self.run_opts.valid_dl_opts.batch_size = self.run_opts.valid_dl_opts.batch_size // self.world_size
        self.run_opts.valid_dl_opts.num_workers = self.run_opts.valid_dl_opts.num_workers // self.world_size

        self.data_bunch = self.run_opts.data_bunch(db_opts=self.run_opts.db_opts,
                                                   train_ds_opts=self.run_opts.train_ds_opts,
                                                   dev_ds_opts=self.run_opts.dev_ds_opts,
                                                   valid_ds_opts=self.run_opts.valid_ds_opts,
                                                   train_dl_opts=self.run_opts.train_dl_opts,
                                                   dev_dl_opts=self.run_opts.dev_dl_opts,
                                                   valid_dl_opts=self.run_opts.valid_dl_opts)
        self._prepare_results_set('train')._prepare_results_set('dev')._prepare_results_set('valid')

    def _prepare_results_set(self, split: str) -> 'Evaluation':
        """Create a results set from the split metadata."""
        assert split in ['train', 'dev', 'valid'], f'Unknown split: {split}.'

        if split == 'train':
            results = self.data_bunch.train_set.meta.copy()
        elif split == 'dev':
            results = self.data_bunch.dev_set.meta.copy()
        else:
            results = self.data_bunch.valid_set.meta.copy()
        # add prediction columns
        results['top1_conf'] = None
        results['top1_pred'] = None
        results['top2_conf_1'] = None
        results['top2_conf_2'] = None
        results['top2_pred_1'] = None
        results['top2_pred_2'] = None
        results['proj_x1'] = None
        results['proj_x2'] = None
        results['preds'] = None

        if split == 'train':
            self.train_results = results
        elif split == 'dev':
            self.dev_results = results
        else:
            self.valid_results = results

        return self

    def evaluate_split(self, split: str) -> 'Evaluation':
        """Get top1 and top2 results for a split and store in DataFrame."""
        assert split in ['train', 'dev', 'valid'], f'Unknown split: {split}.'
        self._log(f'Starting {split} evaluation...')
        if split == 'train':
            dataset = self.data_bunch.train_set
            loader = self.data_bunch.train_loader
            options = self.data_bunch.train_dl_opts
            results = self.train_results
            evaluator = self.train_evaluator
        elif split == 'dev':
            dataset = self.data_bunch.dev_set
            loader = self.data_bunch.dev_loader
            options = self.data_bunch.dev_dl_opts
            results = self.dev_results
            evaluator = self.dev_evaluator
        else:
            dataset = self.data_bunch.valid_set
            loader = self.data_bunch.valid_loader
            options = self.data_bunch.valid_dl_opts
            results = self.valid_results
            evaluator = self.valid_evaluator

        self._calculate_metrics(evaluator, loader, split)
        if self.mode == 'class':
            self._calculate_class_results(dataset, loader, options, results, split)
        elif self.mode == 'ae':
            self._calculate_ae_results(dataset, loader, options, results, split)
        else:
            self._calculate_vae_results(dataset, loader, options, results, split)

        if self.main_proc:
            results = hp.read_smth_results(self.run_dir / f'results_{split}_{self.rank}.tar')
            for rank in range(1, self.world_size):
                results = results.combine_first(hp.read_smth_results(self.run_dir / f'results_{split}_{rank}.tar'))
            cols = ['top1_conf', 'top1_pred', 'top2_conf_1', 'top2_conf_2', 'top2_pred_1', 'top2_pred_2', 'preds']
            for col in cols:
                assert len(results[results[col].isnull()]) == 0, f'{col}, {len(results[results[col].isnull()])}'
            results.to_pickle((self.run_dir / f'results_{split}.tar').as_posix(), compression=None)

        return self

    def _calculate_class_results(self, dataset, loader, options, results, split):
        """Calculates the predictions and embeddings over a dataset."""
        if distributed.is_available():
            distributed.barrier()

        with th.no_grad():
            dataset.evaluating = True
            softmax = nn.Softmax(dim=-1)

            batch_size = options.batch_size * self.world_size
            total = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)

            ids, targets, preds, conf1s, top1s, conf2s, top2s, embeds = [], [], [], [], [], [], [], []
            for i, (video_data, video_labels, videos, labels) in enumerate(loader):
                x = video_data.to(device=self.device, non_blocking=True)
                x = NORMALIZE(x)
                x = STANDARDIZE(x)

                ids.extend([video.meta.id for video in videos])
                targets.extend([label.data for label in labels])

                pred, embed = self.model(x)
                pred_probs = softmax(pred)
                conf1, top1 = pred_probs.max(dim=-1)
                conf2, top2 = pred_probs.topk(2, dim=-1)

                preds.append(pred_probs)
                conf1s.append(conf1)
                top1s.append(top1)
                conf2s.append(conf2)
                top2s.append(top2)
                embeds.append(embed)

                if self.verbose:
                    self._log(f'[Batch: {i + 1}/{total}]')

        preds = th.cat(tuple(preds), dim=0).cpu().numpy()
        conf1s = th.cat(tuple(conf1s), dim=0).cpu().numpy()
        top1s = th.cat(tuple(top1s), dim=0).cpu().numpy()
        conf2s = th.cat(tuple(conf2s), dim=0).cpu().numpy()
        top2s = th.cat(tuple(top2s), dim=0).cpu().numpy()
        embed_ids, embeds = self._stratified_sample_latents(ids, targets, embeds)
        try:
            tsne = skm.TSNE()
            embed_projections = tsne.fit_transform(embeds)
        except ValueError:
            embed_projections = np.array([100, 100])

        results.loc[ids, 'preds'] = [pkl.dumps(pred) for pred in preds]
        results.loc[ids, 'top1_conf'] = conf1s
        results.loc[ids, 'top1_pred'] = top1s
        results.loc[ids, ['top2_conf_1', 'top2_conf_2']] = conf2s.reshape(-1, 2)
        results.loc[ids, ['top2_pred_1', 'top2_pred_2']] = top2s.reshape(-1, 2)
        results.loc[embed_ids, ['proj_x1', 'proj_x2']] = embed_projections
        results.to_pickle((self.run_dir / f'results_{split}_{self.rank}.tar').as_posix(), compression=None)

        if distributed.is_available():
            distributed.barrier()

    def _calculate_ae_results(self, dataset, loader, options, results, split):
        """Calculates the predictions and embeddings over a dataset."""
        if distributed.is_available():
            distributed.barrier()

        with th.no_grad():
            dataset.evaluating = True
            softmax = nn.Softmax(dim=-1)

            batch_size = options.batch_size * self.world_size
            total = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)

            ids, targets, preds, conf1s, top1s, conf2s, top2s, embeds = [], [], [], [], [], [], [], []
            for i, (video_data, video_labels, videos, labels) in enumerate(loader):
                x = video_data.to(device=self.device, non_blocking=True)
                x = NORMALIZE(x)
                x = STANDARDIZE(x)

                ids.extend([video.meta.id for video in videos])
                targets.extend([label.data for label in labels])

                pred, embed, _ = self.model(x, True)
                pred_probs = softmax(pred)
                conf1, top1 = pred_probs.max(dim=-1)
                conf2, top2 = pred_probs.topk(2, dim=-1)

                preds.append(pred_probs)
                conf1s.append(conf1)
                top1s.append(top1)
                conf2s.append(conf2)
                top2s.append(top2)
                embeds.append(embed)

                if self.verbose:
                    self._log(f'[Batch: {i + 1}/{total}]')

        preds = th.cat(tuple(preds), dim=0).cpu().numpy()
        conf1s = th.cat(tuple(conf1s), dim=0).cpu().numpy()
        top1s = th.cat(tuple(top1s), dim=0).cpu().numpy()
        conf2s = th.cat(tuple(conf2s), dim=0).cpu().numpy()
        top2s = th.cat(tuple(top2s), dim=0).cpu().numpy()
        embed_ids, embeds = self._stratified_sample_latents(ids, targets, embeds)
        try:
            tsne = skm.TSNE()
            embed_projections = tsne.fit_transform(embeds)
        except ValueError:
            embed_projections = np.array([100, 100])

        results.loc[ids, 'preds'] = [pkl.dumps(pred) for pred in preds]
        results.loc[ids, 'top1_conf'] = conf1s
        results.loc[ids, 'top1_pred'] = top1s
        results.loc[ids, ['top2_conf_1', 'top2_conf_2']] = conf2s.reshape(-1, 2)
        results.loc[ids, ['top2_pred_1', 'top2_pred_2']] = top2s.reshape(-1, 2)
        results.loc[embed_ids, ['proj_x1', 'proj_x2']] = embed_projections
        results.to_pickle((self.run_dir / f'results_{split}_{self.rank}.tar').as_posix(), compression=None)

        if distributed.is_available():
            distributed.barrier()

    def _calculate_vae_results(self, dataset, loader, options, results, split):
        if distributed.is_available():
            distributed.barrier()

        with th.no_grad():
            dataset.evaluating = True

            batch_size = options.batch_size * self.world_size
            total = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)

            ids, targets, preds, conf1s, top1s, conf2s, top2s, latents = [], [], [], [], [], [], [], []
            for i, (video_data, video_labels, videos, labels) in enumerate(loader):
                x = video_data.to(device=self.device, non_blocking=True)
                x = NORMALIZE(x)
                x = STANDARDIZE(x)

                ids.extend([video.meta.id for video in videos])
                targets.extend([label.data for label in labels])

                _recon, _pred, _latent, _mean, _var, _vote = self.model(x, True, ct.VAE_NUM_SAMPLES)

                bs, ns, nc = _pred.shape

                conf1, top1 = _vote.max(dim=-1)

                conf2, top2 = _vote.topk(2, dim=-1)

                preds.append(_vote)
                conf1s.append(conf1)
                top1s.append(top1)
                conf2s.append(conf2)
                top2s.append(top2)

                _latent = _latent.mean(dim=1).reshape(bs, -1)
                latents.append(_latent)

                if self.verbose:
                    self._log(f'[Batch: {i + 1}/{total}]')

        preds = th.cat(tuple(preds), dim=0).cpu().numpy()
        conf1s = th.cat(tuple(conf1s), dim=0).cpu().numpy()
        top1s = th.cat(tuple(top1s), dim=0).cpu().numpy()
        conf2s = th.cat(tuple(conf2s), dim=0).cpu().numpy()
        top2s = th.cat(tuple(top2s), dim=0).cpu().numpy()
        latent_ids, latents = self._stratified_sample_latents(ids, targets, latents)
        try:
            tsne = skm.TSNE()
            latent_projections = tsne.fit_transform(latents)
        except ValueError:
            latent_projections = np.array([100, 100])

        results.loc[ids, 'preds'] = [pkl.dumps(pred) for pred in preds]
        results.loc[ids, 'top1_conf'] = conf1s
        results.loc[ids, 'top1_pred'] = top1s
        results.loc[ids, ['top2_conf_1', 'top2_conf_2']] = conf2s.reshape(-1, 2)
        results.loc[ids, ['top2_pred_1', 'top2_pred_2']] = top2s.reshape(-1, 2)
        results.loc[latent_ids, ['proj_x1', 'proj_x2']] = latent_projections
        results.to_pickle((self.run_dir / f'results_{split}_{self.rank}.tar').as_posix(), compression=None)

        if distributed.is_available():
            distributed.barrier()

    def _calculate_metrics(self, evaluator: engine.Engine, loader: Any, split: str):
        """Calculate metrics using an evaluator engine."""
        if distributed.is_available():
            distributed.barrier()
        evaluator.run(loader)
        if distributed.is_available():
            distributed.barrier()

        self._aggregate_metrics(evaluator, split)

    def _aggregate_metrics(self, _engine: engine.Engine, split: str) -> None:
        """Gather evaluation metrics. Performs a reduction step if in distributed setting."""
        assert split in ['train', 'dev', 'valid'], f'Unknown split: {split}.'
        if self.distributed:
            names = []
            values = []

            for key, value in _engine.state.metrics.items():
                names.append(f'{split}_{key}')
                values.append(value)
            values = th.tensor(values, requires_grad=False, device=self.device)
            distributed.all_reduce(values)
            values /= self.world_size
            self.metrics.update(zip(names, values.detach().cpu().numpy()))
        else:
            for key, value in _engine.state.metrics.items():
                self.metrics[f'{split}_{key}'] = value

    def _stratified_sample_latents(self, ids: List[int], targets: List[int], latents: List[th.Tensor]) -> Tuple[
        np.ndarray, np.ndarray]:
        """Sample latents uniformly across across classes."""
        latents = [latent.cpu().numpy() for latent in latents]
        latents = [row for batch in latents for row in batch]
        df = pd.DataFrame(zip(ids, targets, latents), columns=['ids', 'targets', 'latents'])

        samples = []
        for target in df.targets.unique():
            sample = df[df.targets == target][0:ct.TSNE_SAMPLE_SIZE]
            samples.append(sample)
        sample = pd.concat(samples, axis=0, verify_integrity=True)

        return sample.ids.values, np.array([latent for latent in sample.latents])

    def _to_score(self, path: pl.Path) -> float:
        """Get the score of a checkpoint from its filename."""
        return float(path.as_posix().replace('.pth', '').split('=').pop())

    def start(self):
        """Evaluates and stores result of a model."""
        # noinspection PyBroadException
        try:
            self.evaluate_split('train').evaluate_split('dev').evaluate_split('valid')

            if distributed.is_available():
                distributed.barrier()

            if self.main_proc:
                self.metrics.update(source=[self.name])
                row = pd.DataFrame(data=self.metrics).set_index('source')
                if self.name in self.results.index:
                    self.results.update(row)
                else:
                    self.results = self.results.append(row, ignore_index=False, verify_integrity=True, sort=True)
                self.results.to_json((ct.SMTH_RUN_DIR / self.cut / 'results.json').as_posix(), orient='index')

                self._log('Evaluation Completed. Sending notification.')
                hp.notify('good', self.run_opts.name, f"""Finished evaluation job.""")
                self._log('Done.')
        except Exception:
            if self.main_proc:
                fields = [
                    {
                        'title': 'Error',
                        'value': traceback.format_exc(),
                        'short': False
                    }
                ]
                hp.notify('bad', self.run_opts.name, f"""Error occurred during evaluation.""", fields)
                raise

    def _log(self, msg: str):
        self.logger.info(msg)
