import dataclasses as dc
from glob import glob
import pathlib as pl
from torch import nn
import pandas as pd
import torch as th
import numpy as np
from tqdm import tqdm

import models.options as mo
from pipeline import smth
from env import logging
import constants as ct


class Evaluation(object):
    data_bunch: smth.SmthDataBunch
    model: nn.Module
    best_ckpt: str
    device: object
    run_opts: mo.RunOptions
    run_dir: pl.Path
    results: pd.DataFrame

    def __init__(self, run_dir: pl.Path, run_opts: mo.RunOptions):
        self.run_dir = ct.RUN_DIR / run_dir
        self.run_opts = run_opts
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self._get_best_ckpt()
        self._load_model()
        self._load_data_bunch()

    def _get_best_ckpt(self) -> None:
        """Get the path to the best checkpoint."""
        model_paths = sorted(glob((self.run_dir / 'model_*.pth').as_posix()))
        scores = np.array(map(self._to_score, model_paths))
        best_idx = scores.argmin()
        self.best_ckpt = model_paths[best_idx]

    def _load_model(self) -> None:
        """Load the state dict of a model and sends to available device(s)."""
        self.model = self.run_opts.model(**dc.asdict(self.run_opts.model_opts))
        with open(self.best_ckpt, 'rb') as file:
            self.model.load_state_dict(th.load(file, map_location='cpu'))
            self.model.to(self.device)

    def _load_data_bunch(self):
        """Load the data bunch."""
        self.data_bunch = self.run_opts.data_bunch(self.run_opts.data_bunch_opts,
                                                   self.run_opts.data_set_opts,
                                                   self.run_opts.data_loader_opts)
        self.results = self.data_bunch.valid_set.meta.copy()
        self.results['top1_conf'] = None
        self.results['top1_pred'] = None
        self.results['top3_conf_1'] = None
        self.results['top3_conf_2'] = None
        self.results['top3_conf_3'] = None
        self.results['top3_pred_1'] = None
        self.results['top3_pred_2'] = None
        self.results['top3_pred_3'] = None

    def _to_score(self, path: pl.Path) -> float:
        """Get the score of a checkpoint from its filename."""
        return float(path.as_posix().replace('.pth', '').split('=').pop())

    def start(self):
        """Evaluates and stores result of a model."""
        logging.info('Starting evaluation...')
        with th.no_grad():
            self.model.eval()
            self.data_bunch.valid_set.presenting = True
            softmax = nn.Softmax(dim=0)
            total = len(self.data_bunch.valid_set) // self.data_bunch.dl_opts.batch_size

            for videos, labels, metas, indices in tqdm(self.data_bunch.valid_loader, total=total):
                x = videos.to(self.device, non_blocking=True)
                e = self.model(x)
                p = softmax(e)
                conf1, top1 = p.max(dim=-1)
                conf3, top3 = p.topk(3, dim=-1)

                self.results.iloc[indices, -8] = conf1.cpu().numpy()
                self.results.iloc[indices, -7] = top1.cpu().numpy()
                self.results.iloc[indices, -6:-3:] = conf3.cpu().numpy()
                self.results.iloc[indices, -3:] = top3.cpu().numpy()
        assert len(self.results[self.results['top1_conf'].isnull()]) == 0
        self.results.to_json(self.run_dir / 'results.json', orient='records')
        logging.info('Done.')
