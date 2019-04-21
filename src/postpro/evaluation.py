import pathlib as pl
import traceback
from glob import glob
from typing import Any, Tuple

import dataclasses as dc
import pandas as pd
import torch as th
from torch import cuda, distributed, nn

import constants as ct
import helpers as hp
import models.options as mo
from env import logging
from pipeline import smth


class Evaluation(object):
    local_rank: int
    run_dir: pl.Path
    run_opts: mo.RunOptions
    rank: int
    world_size: int
    device: object
    best_ckpt: str
    model: nn.Module
    data_bunch: smth.SmthDataBunch
    train_results: pd.DataFrame
    valid_results: pd.DataFrame

    def __init__(self, spec: mo.RunOptions, local_rank: int):
        self.local_rank = local_rank
        self.run_dir = ct.SMTH_RUN_DIR / spec.name
        self.run_opts = spec
        self.rank, self.world_size = self._init_distributed()
        self.device = self._init_device()
        self.best_ckpt = self._get_best_ckpt()
        self.model = self._init_model()
        self._load_data_bunch()

    def _init_distributed(self) -> Tuple[int, int]:
        """Initialize the process pool if distributed is available."""
        if distributed.is_available():
            distributed.init_process_group(backend='nccl', init_method='env://')
            rank = distributed.get_rank()
            world_size = distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        return rank, world_size

    def _init_device(self) -> Any:
        """Set the device according to CPU vs. single process CUDA vs. distributed CUDA."""
        if self.world_size > 1:
            device = th.device(f'cuda:{self.rank}' if cuda.is_available() else f'cpu')
        else:
            device = th.device(f'cuda' if cuda.is_available() else f'cpu')

        return device

    def _get_best_ckpt(self) -> str:
        """Get the path to the best checkpoint."""
        return glob((self.run_dir / 'best_model_*.pth').as_posix()).pop()

    def _init_model(self) -> nn.Module:
        """Load the state dict of a model and sends to available device(s)."""
        model = self.run_opts.model(**dc.asdict(self.run_opts.model_opts))
        model.to(self.device)
        model.load_state_dict(th.load(self.best_ckpt, map_location=self.device))
        if distributed.is_available() and self.local_rank != -1:
            if cuda.is_available():
                if self.world_size > 1:
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank)
                else:
                    model = nn.parallel.DistributedDataParallel(model)
            else:
                model = nn.parallel.DistributedDataParallelCPU(model)

        return model.to(self.device)

    def _load_data_bunch(self) -> None:
        """Load the data bunch."""

        # flag use of distributed sampler
        self.run_opts.db_opts.distributed = self.world_size > 1

        # make sure batch and workers and distributed well across worlds.
        self.run_opts.train_dl_opts.batch_size = self.run_opts.train_dl_opts.batch_size // self.world_size * ct.NUM_DEVICES
        self.run_opts.train_dl_opts.num_workers = self.run_opts.train_dl_opts.num_workers // self.world_size
        self.run_opts.valid_dl_opts.batch_size = self.run_opts.valid_dl_opts.batch_size // self.world_size * ct.NUM_DEVICES
        self.run_opts.valid_dl_opts.num_workers = self.run_opts.valid_dl_opts.num_workers // self.world_size

        self.data_bunch = self.run_opts.data_bunch(self.run_opts.db_opts,
                                                   self.run_opts.train_ds_opts,
                                                   self.run_opts.valid_ds_opts,
                                                   self.run_opts.train_dl_opts,
                                                   self.run_opts.valid_dl_opts)
        self._prepare_results_set('train')._prepare_results_set('valid')

    def _prepare_results_set(self, split: str) -> 'Evaluation':
        """Create a results set from the split metadata."""
        assert split in ['train', 'valid'], f'Unknown split: {split}.'

        if split == 'train':
            results = self.data_bunch.train_set.meta.copy()
        else:
            results = self.data_bunch.valid_set.meta.copy()
        # add prediction columns
        results['top1_conf'] = None
        results['top1_pred'] = None
        results['top2_conf_1'] = None
        results['top2_conf_2'] = None
        results['top2_pred_1'] = None
        results['top2_pred_2'] = None

        if split == 'train':
            self.train_results = results
        else:
            self.valid_results = results

        return self

    def _to_score(self, path: pl.Path) -> float:
        """Get the score of a checkpoint from its filename."""
        return float(path.as_posix().replace('.pth', '').split('=').pop())

    def _evaluate_split(self, split: str) -> 'Evaluation':
        """Get top1 and top2 results for a split an store in DataFrame."""
        assert split in ['train', 'valid'], f'Unknown split: {split}.'
        if self.rank == 0:
            logging.info(f'Starting {split} evaluation...')
        if split == 'train':
            dataset = self.data_bunch.train_set
            loader = self.data_bunch.train_loader
            options = self.data_bunch.train_dl_opts
            results = self.train_results
        else:
            dataset = self.data_bunch.valid_set
            loader = self.data_bunch.valid_loader
            options = self.data_bunch.valid_dl_opts
            results = self.valid_results
        with th.no_grad():
            self.model.eval()
            dataset.evaluating = True
            softmax = nn.Softmax(dim=-1)
            total = len(dataset) // options.batch_size

            for i, (video_data, video_labels, videos, labels) in enumerate(loader):
                x = video_data.to(device=self.device, non_blocking=True)
                e = self.model(x)
                p = softmax(e)
                conf1, top1 = p.max(dim=-1)
                conf2, top2 = p.topk(2, dim=-1)
                ids = [video.meta.id for video in videos]
                results.loc[ids, 'top1_conf'] = conf1.cpu().numpy()
                results.loc[ids, 'top1_pred'] = top1.cpu().numpy()
                results.loc[ids, ['top2_conf_1', 'top2_conf_2']] = conf2.cpu().numpy()
                results.loc[ids, ['top2_pred_1', 'top2_pred_2']] = top2.cpu().numpy()

                del e
                del p
                del conf1
                del conf2
                del top1
                del top2

                if self.rank == 0:
                    logging.info(f'[Batch: {i}/{total}]')
        results.to_json(self.run_dir / f'results_{split}_{self.rank}.json', orient='records')

        if distributed.is_available():
            distributed.barrier()

        if self.rank == 0:
            results = hp.read_smth_results(self.run_dir / f'results_{split}_{self.rank}.json')
            for rank in range(1, self.world_size):
                results = results.combine_first(hp.read_smth_results(self.run_dir / f'results_{split}_{rank}.json'))
            cols = ['top1_conf', 'top1_pred', 'top2_conf_1', 'top2_conf_2', 'top2_pred_1', 'top2_pred_2']
            for col in cols:
                assert len(results[results[col].isnull()]) == 0, f'{col}, {len(results[results[col].isnull()])}'
            results.to_json(self.run_dir / f'results_{split}.json', orient='records')

        return self

    def start(self):
        """Evaluates and stores result of a model."""
        # noinspection PyBroadException
        try:
            self._evaluate_split('train')._evaluate_split('valid')

            if distributed.is_available():
                distributed.barrier()

            if self.rank == 0:
                logging.info('Evaluation Completed. Sending notification.')
                hp.notify('good', self.run_opts.name, f"""Finished evaluation job.""")
                logging.info(f'Done.')
        except Exception:
            fields = [
                {
                    'title': 'Error',
                    'value': traceback.format_exc(),
                    'short': False
                }
            ]
            hp.notify('bad', self.run_opts.name, f"""Error occurred during evaluation.""", fields)
            raise
