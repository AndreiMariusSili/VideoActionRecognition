import abc
import collections as cl
import copy
import dataclasses as dc
import glob
import typing as t

import ignite.engine as ie
import numpy as np
import torch as th
import torch.cuda as cuda
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data as tud
import tqdm

import constants as ct
import databunch.databunch as db
import logger as lg
import options.experiment_options as eo
import specs.maps as sm

RESULTS = t.Tuple[t.Dict[str, np.ndarray], np.ndarray]
TSNE = t.Tuple[t.Dict[str, np.ndarray], np.ndarray]


class BaseEvaluator(abc.ABC):
    def __init__(self, opts: eo.ExperimentOptions, local_rank: int):
        self.opts = opts
        self.local_rank = local_rank

        self.rank, self.world_size = self._init_distributed()
        self.device = th.device(f'cuda' if cuda.is_available() else f'cpu')
        self.data_bunch, self.opts.model.opts.num_classes = self._init_databunch()
        self.best_ckpt = self._get_best_ckpt()
        self.model = self._init_model()
        self.train_evaluator, self.dev_evaluator, self.test_evaluator = self._init_evaluators()

        self.logger = lg.ExperimentLogger(
            self.opts,
            True,
            self.opts.trainer.metrics,
            self.opts.evaluator.metrics
        )
        self.logger.attach_pbar(self.train_evaluator)
        self.logger.init_log()

    def _init_distributed(self) -> t.Tuple[int, int]:
        """Create distributed setup."""
        if dist.is_available() and self.local_rank != -1:
            dist.init_process_group(backend='nccl')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert world_size == 1, 'Invalid distributed init. Should be one process to many gpus.'
        else:
            rank = 0
            world_size = 1
        return rank, world_size

    def _get_best_ckpt(self) -> str:
        """Get the path to the best checkpoint."""
        best_models = list(glob.glob((ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'best_model_*.pth').as_posix()))
        if len(best_models) > 1:
            raise ValueError('More than one best model available. Remove old versions.')
        return best_models.pop()

    def _init_model(self) -> nn.Module:
        """Initialize, resume model. One process, multi-gpu."""
        opts = dc.asdict(copy.deepcopy(self.opts.model.opts))
        del opts['batch_size']
        model = sm.Models[self.opts.model.arch].value(**opts).to(self.device)
        print(f'Loading model from {self.best_ckpt}...')
        model.load_state_dict(th.load(self.best_ckpt, map_location=self.device))

        if self.opts.overfit:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    module.momentum = 1.0

        if self.local_rank != -1:
            model = nn.parallel.DistributedDataParallel(model)

        model.eval()

        return model

    @abc.abstractmethod
    def _init_evaluators(self) -> t.Tuple[ie.Engine, ie.Engine, ie.Engine]:
        """Initialize the evaluator engines."""

    def _init_databunch(self) -> t.Tuple[db.VideoDataBunch, int]:
        """Load the data bunch. All options are already set in the runner."""
        self.opts.databunch.train_dso.setting = 'eval'
        self.opts.databunch.dev_dso.setting = 'eval'
        self.opts.databunch.test_dso.setting = 'eval'
        self.opts.databunch.dlo.shuffle = False
        self.opts.databunch.distributed = self.local_rank != -1
        data_bunch = db.VideoDataBunch(db_opts=self.opts.databunch)

        return data_bunch, len(data_bunch.lids)

    def evaluate_split(self, split: str) -> 'BaseEvaluator':
        """Calculate results for a split and log."""
        assert split in ['train', 'dev', 'test'], f'Unknown split: {split}.'
        self.logger.log(f'Evaluating {split} split...')
        (ct.WORK_ROOT / self.opts.run_dir / split).mkdir(parents=True, exist_ok=True)

        if split == 'train':
            loader = self.data_bunch.train_loader
            evaluator = self.train_evaluator
        elif split == 'dev':
            loader = self.data_bunch.dev_loader
            evaluator = self.dev_evaluator
        else:
            loader = self.data_bunch.test_loader
            evaluator = self.test_evaluator

        metrics = self._calculate_metrics(evaluator, loader, split)
        self.logger.log_metrics(metrics)
        self.logger.persist_metrics(metrics, split)

        # Not computing predictions and tsnes for now since we don't use them anywhere.
        # outs, ids = self._calculate_results(loader, split)
        # outs, tsne_ids = self._calculate_tsnes(outs, ids, split)
        # self.logger.persist_outs(outs, tsne_ids, split)

        return self

    def _calculate_metrics(self, evaluator: ie.Engine, loader: tud.DataLoader, split: str) -> t.Dict[str, float]:
        """Calculate metrics using an evaluator engine."""
        evaluator.run(loader)
        self._aggregate_metrics(evaluator)

        return {f'{split}_{k}': float(v) for k, v in evaluator.state.metrics.items()}

    def _calculate_results(self, loader: tud.DataLoader, split: str) -> RESULTS:
        """Calculate extra results: predictions, embeddings, etc."""
        ids, targets = [], []
        outs = cl.defaultdict(list)

        pbar = tqdm.tqdm(total=len(loader.dataset), leave=True)
        with th.no_grad():
            for i, (_in, _cls_gt, _recon_gt, videos) in enumerate(loader):
                x = _in.to(device=self.device, non_blocking=True)

                ids.extend([video.id for video in videos])

                batch_outs = self._get_model_outputs(x)
                for k, v in batch_outs.items():
                    outs[k].append(v)

                pbar.update(x.shape[0])
        pbar.clear()
        pbar.close()

        ids, outs = np.array(ids), dict(outs)
        for k, v in outs.items():
            outs[k] = np.concatenate(v, axis=0)

        return outs, ids

    @abc.abstractmethod
    def _get_model_outputs(self, x: th.Tensor) -> t.Dict[str, th.Tensor]:
        """Get outputs of interest from a model."""

    @abc.abstractmethod
    def _get_projections(self, embed_name: str, sample_embeds: np.ndarray) -> np.ndarray:
        """Get tsne projections."""

    def _calculate_tsnes(self, outs: t.Dict[str, np.ndarray], ids: np.ndarray, split: str) -> TSNE:
        """Calculate TSNE projections for embeddings. Samples from train set because it's too big."""
        outs_proj = {}

        if split == 'train' and not self.opts.debug:
            sample = np.random.choice(len(ids), min(ct.TSNE_TRAIN_SAMPLE_SIZE, len(ids)), replace=False)
        else:
            sample = np.arange(0, len(ids))

        sample_ids = ids[sample]
        for name, out in outs.items():
            if 'class_embeds' in name:
                sample_embeds = out[sample]
                sample_tsne = self._get_projections(name, sample_embeds)
                outs_proj[f'{name}_tsne'] = sample_tsne
            else:
                outs_proj[name] = out

        return outs_proj, sample_ids

    def _aggregate_metrics(self, _engine: ie.Engine) -> None:
        """Gather evaluation metrics. Performs a reduction step if in distributed setting."""
        local_names, global_names, values = [], [], []
        for key, value in _engine.state.metrics.items():
            local_names.append(key)
            values.append(value)

        values = th.tensor(values, requires_grad=False, device=self.device)
        _engine.state.metrics.update(zip(local_names, values.detach().cpu().numpy()))

    def start(self):
        """Evaluates and stores result of a model."""
        try:
            self.evaluate_split('train').evaluate_split('dev').evaluate_split('test')
            self.logger.persist_experiment()
            self.logger.close()
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            self.logger.close()
            if dist.is_initialized():
                dist.destroy_process_group()
            raise e
