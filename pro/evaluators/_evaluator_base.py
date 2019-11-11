import abc
import copy
import dataclasses as dc
import glob
import pathlib as pth
import typing as typ

import ignite.engine as ie
import pandas as pd
import sklearn.manifold as skm
import torch as th
import torch.utils.data as tud
import tqdm
from torch import cuda as cuda, distributed as dist, multiprocessing as mp, nn as nn

import constants as ct
import databunch.databunch as db
import helpers as hp
import options.data_options as do
import options.experiment_options as eo
import pro.logger as pl


class BaseEvaluator(abc.ABC):
    CONF_COLS = ['conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5']
    PRED_COLS = ['pred_1', 'pred_2', 'pred_3', 'pred_4', 'pred_5']
    PROJ_COLS = ['proj_1', 'proj_2']
    TSNE = skm.TSNE()
    TSNE_SAMPLE_SIZE = ct.TSNE_SAMPLE_SIZE

    def __init__(self, opts: eo.ExperimentOptions, local_rank: int):
        self.opts = opts
        self.local_rank = local_rank

        self.rank, self.world_size, self.main_proc, self.distributed = self._init_distributed()
        self.run_dir = self._init_run()
        self.logger = pl.ExperimentLogger(self.opts, self.main_proc, self.run_dir,
                                          self.opts.trainer.metrics, self.opts.evaluator.metrics)

        self.device = self._init_device()
        self.data_bunch, self.opts.model_opts.num_classes = self._init_databunch()
        self.best_ckpt = self._get_best_ckpt()
        self.model = self._init_model()
        self.criterion = self._init_criterion()
        self.train_evaluator, self.dev_evaluator, self.test_evaluator = self._init_evaluators()

        self.train_results, self.dev_results, self.test_results = self._prepare_results()

    def _init_distributed(self) -> typ.Tuple[int, int, bool, bool]:
        """Create distributed setup."""
        if dist.is_available() and self.local_rank != -1:
            dist.init_process_group(backend='nccl', init_method='env://')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size, rank == 0, world_size > 1

    def _init_run(self) -> pth.Path:
        """Create the run directory and store the run options."""
        run_dir = ct.RUNS_ROOT / self.opts.name
        run_dir.mkdir(parents=True, exist_ok=True)

        return run_dir

    def _init_device(self) -> typ.Any:
        if self.distributed:
            device = th.device(f'cuda:{self.rank}' if cuda.is_available() else f'cpu')
        else:
            device = th.device(f'cuda' if cuda.is_available() else f'cpu')

        return device

    def _get_best_ckpt(self) -> str:
        """Get the path to the best checkpoint."""
        best_models = list(glob.glob((self.run_dir / 'best_model_*.pth').as_posix()))
        if len(best_models) > 1:
            raise ValueError('More than one best model available. Remove old versions.')
        return best_models.pop()

    def _init_model(self) -> nn.Module:
        """Initialize, resume model. One process, multi-gpu."""
        opts = dc.asdict(copy.deepcopy(self.opts.model_opts))
        model = opts.pop('arch')(**opts).to(self.device)
        self.logger.log(f'Loading model from {self.best_ckpt}...')
        model.load_state_dict(th.load(self.best_ckpt, map_location=self.device))

        model = nn.parallel.DistributedDataParallel(model)

        if self.opts.debug:
            for module in model.modules():
                if type(module) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                    module.track_running_stats = False
                elif type(module) in [nn.Dropout, nn.Dropout2d, nn.Dropout3d]:
                    module.p = 0.0

        model.eval()

        return model

    def _init_criterion(self):
        """Initialize loss function to correct device."""
        criterion = self.opts.trainer.criterion(**dc.asdict(self.opts.trainer.criterion_opts))

        return criterion.to(self.device)

    @abc.abstractmethod
    def _init_evaluators(self) -> typ.Tuple[ie.Engine, ie.Engine, ie.Engine]:
        """Initialize the evaluator engines."""

    def _init_databunch(self) -> typ.Tuple[db.VideoDataBunch, int]:
        """Load the data bunch."""

        # flag use of distributed sampler; set start method to prevent deadlocks.
        self.opts.databunch_opts.distributed = self.world_size > 1
        if self.distributed:
            mp.set_start_method('forkserver')

        # make sure batch and workers are distributed well across worlds. -1 for the main process
        self.opts.databunch_opts.dlo.batch_size //= self.world_size
        self.opts.databunch_opts.dlo.num_workers //= self.world_size
        self.opts.databunch_opts.dlo.num_workers -= 1

        self.opts.databunch_opts.dlo.shuffle = False
        data_bunch = db.VideoDataBunch(db_opts=self.opts.databunch_opts)

        return data_bunch, len(data_bunch.lids)

    def _prepare_results(self) -> typ.List[pd.DataFrame]:
        """Create a results set from the split metadata."""
        split_results = []
        for split in ['train', 'dev', 'test']:
            results = getattr(self.data_bunch, f'{split}_set').meta.copy().set_index('id', verify_integrity=True)

            results[self.CONF_COLS] = None
            results[self.PRED_COLS] = None
            results[self.PROJ_COLS] = None

            split_results.append(results)

        return split_results

    def evaluate_split(self, split: str) -> 'BaseEvaluator':
        """Calculate results for a split and log."""
        assert split in ['train', 'dev', 'test'], f'Unknown split: {split}.'
        self.logger.log(f'Starting {split} evaluation...')

        options = self.data_bunch.dbo.dlo
        if split == 'train':
            loader = self.data_bunch.train_loader
            results = self.train_results
            evaluator = self.train_evaluator
        elif split == 'dev':
            loader = self.data_bunch.dev_loader
            results = self.dev_results
            evaluator = self.dev_evaluator
        else:
            loader = self.data_bunch.test_loader
            results = self.test_results
            evaluator = self.test_evaluator

        self._calculate_metrics(evaluator, loader)
        self._calculate_results(evaluator, loader, options, results, split)
        self._calculate_tsne_projections(results, split)

        # if self.mode == 'class':
        #     self._calculate_class_results(dataset, loader, options, results, split)
        # elif self.mode == 'ae':
        #     self._calculate_ae_results(dataset, loader, options, results, split)
        # else:
        #     self._calculate_vae_results(dataset, loader, options, results, split)

        return self

    def _calculate_metrics(self, evaluator: ie.Engine, loader: tud.DataLoader):
        """Calculate metrics using an evaluator engine."""
        evaluator.run(loader)
        self._aggregate_metrics(evaluator)
        self._augment_metrics(evaluator)
        self.logger.log_experiment_metrics(evaluator.state.metrics)

    def _calculate_results(self, evaluator: ie.Engine, loader: tud.DataLoader, options: do.DataBunchOptions,
                           results: pd.DataFrame, split: str):
        """Calculate extra results: predictions, embeddings, etc."""
        loader.dataset.evaluating = True
        ids, targets, embeds, energies, confs = [], [], [], [], []
        pbar = tqdm.tqdm(total=len(loader.dataset))

        softmax = nn.Softmax(dim=-1)
        with th.no_grad():
            for i, (video_data, video_labels, videos, labels) in enumerate(loader):
                x = video_data.to(device=self.device, non_blocking=True)

                ids.extend([video.meta.id for video in videos])
                targets.extend([label.data for label in labels])

                energy, embed, *_ = self.model(x)
                conf = softmax(energy)

                embeds.append(embed.cpu())
                energies.append(energy.cpu())
                confs.append(conf.cpu())

                pbar.update(x.shape[0])

        confs = th.cat(tuple(confs), dim=0).numpy()
        conf5, pred5 = confs.topk(5, dim=-1)
        results.loc[ids, self.CONF_COLS] = conf5.reshape(-1, 5)
        results.loc[ids, self.PRED_COLS] = pred5.reshape(-1, 5)

        embeds = th.cat(tuple(embeds), dim=0).numpy()
        energies = th.cat(tuple(energies), dim=0).numpy()

        embeds = pd.DataFrame(data=embeds, index=results.index)
        energies = pd.DataFrame(data=energies, index=results.index)

        embeds.to_parquet((self.run_dir / f'embeds_{split}.pqt').as_posix(), engine='pyarrow', index=True)
        energies.to_parquet((self.run_dir / f'energies_{split}.pqt').as_posix(), engine='pyarrow', index=True)
        results.to_parquet((self.run_dir / f'results_{split}.pqt').as_posix(), engine='pyarrow', index=True)

    def _calculate_tsne_projections(self, results: pd.DataFrame, split: str):
        results = pd.read_parquet((self.run_dir / f'results_{split}.pqt').as_posix(), engine='pyarrow')
        embeds = pd.read_parquet((self.run_dir / f'embeds_{split}.pqt').as_posix(), engine='pyarrow')
        targets = results.loc[embeds.index, 'lid'].values

        embeds_sample = self._stratified_sample_latents(embeds, targets)

        proj_sample = self.TSNE.fit_transform(embeds_sample)
        proj_sample = pd.DataFrame(data=proj_sample, index=embeds_sample.index)
        proj_sample.to_parquet((self.run_dir / f'tsne_{split}.pqt').as_posix())

    def _aggregate_metrics(self, _engine: ie.Engine) -> None:
        """Gather evaluation metrics. Performs a reduction step if in distributed setting."""
        local_names, global_names, values = [], [], []
        for key, value in _engine.state.metrics.items():
            local_names.append(key)
            values.append(value)

        values = th.tensor(values, requires_grad=False, device=self.device)
        if self.distributed:
            dist.all_reduce(values)
            values /= self.world_size
        _engine.state.metrics.update(zip(local_names, values.detach().cpu().numpy()))

    def _augment_metrics(self, _engine: ie.Engine) -> None:
        """Augment metrics with various markers."""
        _engine.state.metrics.update({
            'model': [self.opts.name],
            'size': hp.count_parameters(self.model)
        })

    def _stratified_sample_latents(self, embeds, targets):
        """Sample latents uniformly across across classes."""
        df = embeds.join(targets, how='inner')

        samples = []
        for target in df.targets.unique():
            sample = df[df.targets == target][0:self.TSNE_SAMPLE_SIZE]
            samples.append(sample)
        sample = pd.concat(samples, axis=0, verify_integrity=True)

        return sample

    def start(self):
        """Evaluates and stores result of a model."""
        try:
            self.evaluate_split('train').evaluate_split('dev').evaluate_split('test')
        except Exception as e:
            self.logger.close()
            raise e
