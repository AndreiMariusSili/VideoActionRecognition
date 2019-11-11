import dataclasses as dc
import logging as log
import pathlib as pth
import typing as tp

import ignite.contrib.handlers.tensorboard_logger as tbl
import ignite.contrib.handlers.tqdm_logger as tql
import ignite.engine as ie
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.utils.tensorboard as ttb

import databunch.databunch as db
import helpers as hp
import options.experiment_options


class ExperimentLogger(object):

    def __init__(self, opts: options.experiment_options.ExperimentOptions, main_proc: bool, run_dir: pth.Path,
                 train_metrics: tp.Dict[str, tp.Any], dev_metrics: tp.Dict[str, tp.Any]):
        self.opts = opts
        self.main_proc = main_proc
        self.run_dir = run_dir
        self.train_metrics = train_metrics
        self.dev_metrics = dev_metrics
        self.train_pbar, self.dev_pbar = self._init_train_pbar()
        self.tb_logger = tbl.TensorboardLogger(log_dir=(self.run_dir / 'logs').as_posix())
        self.summary = ttb.SummaryWriter((self.run_dir / 'logs').as_posix(), flush_secs=60)

    def _init_train_pbar(self) -> tp.Optional[tql.ProgressBar]:
        if not self.main_proc:
            return

        train_pbar = tql.ProgressBar(persist=True)

        return train_pbar

    def attach_train_pbar(self, engine: ie.Engine):
        if not self.main_proc:
            return

        self.train_pbar.attach(engine, 'all')

    def init_handlers(self, trainer: ie.Engine, evaluator: ie.Engine, model: nn.Module, optimizer):
        if not self.main_proc:
            return

        self.tb_logger.attach(trainer,
                              log_handler=tbl.OutputHandler(
                                  tag='training',
                                  metric_names='all'),
                              event_name=ie.Events.ITERATION_COMPLETED)
        self.tb_logger.attach(trainer,
                              log_handler=tbl.OptimizerParamsHandler(
                                  optimizer,
                                  tag='training'),
                              event_name=ie.Events.ITERATION_COMPLETED)

        self.tb_logger.attach(trainer,
                              log_handler=tbl.OutputHandler(tag='train',
                                                            metric_names='all'),
                              event_name=ie.Events.EPOCH_COMPLETED)
        self.tb_logger.attach(evaluator,
                              log_handler=tbl.OutputHandler(tag='dev',
                                                            metric_names='all',
                                                            global_step_transform=tbl.global_step_from_engine(trainer)),
                              event_name=ie.Events.EPOCH_COMPLETED)

        if self.opts.debug:
            self.tb_logger.attach(trainer, log_handler=tbl.WeightsHistHandler(model, tag='debug'),
                                  event_name=ie.Events.ITERATION_COMPLETED)
            self.tb_logger.attach(trainer, log_handler=tbl.WeightsScalarHandler(model, tag='debug'),
                                  event_name=ie.Events.ITERATION_COMPLETED)
            self.tb_logger.attach(trainer, log_handler=tbl.GradsHistHandler(model, tag='debug'),
                                  event_name=ie.Events.EPOCH_COMPLETED)
            self.tb_logger.attach(trainer, log_handler=tbl.GradsScalarHandler(model, tag='debug'),
                                  event_name=ie.Events.ITERATION_COMPLETED)
            self.tb_logger.attach(trainer, log_handler=tbl.GradsHistHandler(model, tag='debug'),
                                  event_name=ie.Events.EPOCH_COMPLETED)

    def init_log(self, data_bunch: db.VideoDataBunch,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Adam,
                 lr_scheduler: lrs.MultiStepLR):
        self.log(f'Initializing run {self.opts.name}.')
        self.log(str(data_bunch))
        self.log(f'{hp.count_parameters(model):,}')
        self.log(str(criterion))
        self.log(str(optimizer))
        self.log(str(lr_scheduler.state_dict()))

        #
        _in = th.randn([2,
                        self.opts.databunch_opts.train_dso.so.num_segments,
                        3,
                        self.opts.databunch_opts.frame_size,
                        self.opts.databunch_opts.frame_size], dtype=th.float)
        self.summary.add_graph(model, _in)

    def log_dev_metrics(self, engine: ie.Engine):
        if self.main_proc:
            metrics = ', '.join(f'dev_{key}={value:.4f}' for key, value in engine.state.metrics.items())
            # self.summary.add_hparams(hparam_dict={}, metric_dict=engine.state.metrics)
            self.log(metrics)

    def log(self, msg: str):
        if self.main_proc:
            log.info(msg)

    def log_experiment_metrics(self, metric_dict: tp.Dict[str, tp.Any]):
        if not self.main_proc:
            return

        self.summary.add_hparams(hparam_dict=hp.flatten_dict(dc.asdict(self.opts)), metric_dict=metric_dict)

    def close(self):
        if self.main_proc:
            self.train_pbar.close()
            self.dev_pbar.close()
            self.tb_logger.close()
