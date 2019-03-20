import dataclasses as dc
from glob import glob
import pathlib as pl
from torch import nn, distributed, cuda
import pandas as pd
import torch as th
import numpy as np
from tqdm import tqdm

import models.options as mo
from pipeline import smth
from env import logging
import constants as ct


class Evaluation(object):
    rank: int
    data_bunch: smth.SmthDataBunch
    model: nn.Module
    best_ckpt: str
    device: object
    run_opts: mo.RunOptions
    run_dir: pl.Path
    train_results: pd.DataFrame
    valid_results: pd.DataFrame

    def __init__(self, spec: mo.RunOptions):
        self._init_distributed()
        self.run_dir = ct.RUN_DIR / spec.name
        self.run_opts = spec
        self.device = th.device(f'cuda:{self.rank}' if th.cuda.is_available() else f'cpu:{self.rank}')
        self._get_best_ckpt()
        self._init_model()
        self._load_data_bunch()

    def _init_distributed(self):
        """Initialize the process pool if distributed is available."""
        if distributed.is_available():
            if cuda.is_available():
                distributed.init_process_group(backend='nccl', init_method='env://')
            else:
                distributed.init_process_group(backend='gloo', init_method='env://')
            self.rank = distributed.get_rank()
        else:
            self.rank = 0

    def _get_best_ckpt(self) -> None:
        """Get the path to the best checkpoint."""
        model_paths = sorted(glob((self.run_dir / 'run_model_*.pth').as_posix()))
        scores = np.array(map(self._to_score, model_paths))
        best_idx = scores.argmin()
        self.best_ckpt = model_paths[best_idx]

    def _init_model(self) -> None:
        """Load the state dict of a model and sends to available device(s)."""
        self.model = self.run_opts.model(**dc.asdict(self.run_opts.model_opts))
        self.model = self.model.to(self.device)
        with open(self.best_ckpt, 'rb') as file:
            self.model.load_state_dict(th.load(file, map_location='cpu'))
        if distributed.is_available():
            if cuda.is_available():
                self.model = nn.parallel.DistributedDataParallel(self.model)
            else:
                self.model = nn.parallel.DistributedDataParallelCPU(self.model)

    def _load_data_bunch(self) -> None:
        """Load the data bunch."""
        self.run_opts.data_bunch_opts.distributed = distributed.is_available()
        self.data_bunch = self.run_opts.data_bunch(self.run_opts.data_bunch_opts,
                                                   self.run_opts.train_data_set_opts,
                                                   self.run_opts.valid_data_set_opts,
                                                   self.run_opts.train_data_loader_opts,
                                                   self.run_opts.valid_data_loader_opts)
        self._prepare_results_set('train')._prepare_results_set('valid')

    def _prepare_results_set(self, split: str) -> 'Evaluation':
        """Create a results set from the split metadata."""
        assert split in ['train', 'valid'], f'Unknown split: {split}.'

        if split == 'train':
            results = self.data_bunch.train_set.meta.copy()
        else:
            results = self.data_bunch.valid_set.meta.copy()
        results['top1_conf'] = None
        results['top1_pred'] = None
        results['top3_conf_1'] = None
        results['top3_conf_2'] = None
        results['top3_conf_3'] = None
        results['top3_pred_1'] = None
        results['top3_pred_3'] = None
        results['top3_pred_2'] = None

        if split == 'train':
            self.train_results = results
        else:
            self.valid_results = results

        return self

    def _to_score(self, path: pl.Path) -> float:
        """Get the score of a checkpoint from its filename."""
        return float(path.as_posix().replace('.pth', '').split('=').pop())

    def start(self):
        """Evaluates and stores result of a model."""
        self._evaluate_split('train')._evaluate_split('valid')

    def _evaluate_split(self, split: str) -> 'Evaluation':
        """Get top1 and top3 results for a split an store in DataFrame."""
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

            for video_data, video_labels, videos, data in tqdm(loader, total=total):
                x = video_data.to(self.device, non_blocking=True)
                e = self.model(x)
                p = softmax(e)
                conf1, top1 = p.max(dim=-1)
                conf3, top3 = p.topk(3, dim=-1)
                ids = [video.meta.id for video in videos]
                results.loc[ids, 'top1_conf'] = conf1.cpu().numpy()
                results.loc[ids, 'top1_pred'] = top1.cpu().numpy()
                results.loc[ids, ['top3_conf_1', 'top3_conf_2', 'top3_conf_3']] = conf3.cpu().numpy()
                results.loc[ids, ['top3_pred_1', 'top3_pred_2', 'top3_pred_3']] = top3.cpu().numpy()
        assert len(results[results['top1_conf'].isnull()]) == 0
        results.to_json(self.run_dir / f'results_{split}.json', orient='records')
        if self.rank == 0:
            logging.info('Done.')

        return self
