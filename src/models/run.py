from glob import glob
from typing import Any, Union, Dict, Type, Tuple
from torch import nn, cuda, distributed
from ignite import engine, handlers
import dataclasses as dc
import pathlib as pl
import torch as th
import math
import json
import csv
import os

import models.options as mo
from env import logging
import pipeline as pipe
import constants as ct


class Run(object):
    distributed: bool
    rank: int
    iterations: Union[float, int]
    log_interval: int
    train_evaluator: engine.Engine
    valid_evaluator: engine.Engine
    trainer: engine.Engine
    device: object
    metrics: Dict[str, Any]
    data_bunch: pipe.SmthDataBunch
    criterion: nn.CrossEntropyLoss
    optimizer: Union[th.optim.Adam, th.optim.SGD, th.optim.RMSprop]
    model: nn.Module
    epochs: int
    resume: bool
    name: str
    run_dir: pl.Path

    logger: logging.Logger

    def __init__(self, opts: mo.RunOptions):
        self.__init_distributed()
        self.__init_record(opts)
        self.logger = self.__init_logging()
        self.logger.info(f'Initializing run {opts.name}.')

        self.name = opts.name
        self.resume = opts.resume
        self.epochs = opts.trainer_opts.epochs
        self.patience = opts.patience
        self.log_interval = opts.log_interval

        self.metrics = opts.evaluator_opts.metrics
        self.device = th.device(f'cuda:{self.rank}' if th.cuda.is_available() else f'cpu')
        self.model = self.__init_model(opts.model, opts.model_opts)
        self.logger.info(self.model)
        self.optimizer = self.__init_optimizer(opts.trainer_opts.optimizer, opts.trainer_opts.optimizer_opts)
        self.logger.info(self.optimizer)
        self.criterion = opts.trainer_opts.criterion(**dc.asdict(opts.trainer_opts.criterion_opts)).to(self.device)
        self.logger.info(self.criterion)
        opts.data_bunch_opts.distributed = distributed.is_available()
        self.data_bunch = opts.data_bunch(opts.data_bunch_opts,
                                          opts.train_data_set_opts,
                                          opts.valid_data_set_opts,
                                          opts.train_data_loader_opts,
                                          opts.valid_data_loader_opts)
        self.logger.info(self.data_bunch)

        self.trainer, self.train_evaluator, self.valid_evaluator = self.__init_trainer_evaluator()

        train_examples = len(self.data_bunch.train_set)
        batch_size = self.data_bunch.train_dl_opts.batch_size
        self.iterations = math.ceil(train_examples / batch_size) * self.epochs

        self.__init_handlers()

    def __init_distributed(self):
        if distributed.is_available():
            if cuda.is_available():
                distributed.init_process_group(backend='nccl', init_method='env://')
            else:
                distributed.init_process_group(backend='gloo', init_method='env://')
            self.rank = distributed.get_rank()
        else:
            self.rank = 0

    def __init_record(self, opts: mo.RunOptions):
        self.run_dir = ct.RUN_DIR / opts.name
        os.makedirs(self.run_dir.as_posix(), exist_ok=True)
        with open((self.run_dir / 'options.json').as_posix(), 'w') as file:
            json.dump(dc.asdict(opts), file, indent=True, sort_keys=True, default=str)

    def __init_logging(self, ):
        """Get a logger that outputs to console and file. Only log messages if on the main process."""
        logger = logging.getLogger()
        if self.rank == 0:
            formatter = logging.Formatter('[%(asctime)-s][%(process)d][%(levelname)s]\t%(message)s')
            file_handler = logging.FileHandler(self.run_dir / f'training.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            logger.handlers = []

        return logger

    def __init_model(self, model: nn.Module, opts: Any) -> nn.Module:
        model = model(**dc.asdict(opts))
        model = model.to(self.device)
        if self.resume:
            model_path = glob((self.run_dir / 'run_model_*').as_posix()).pop()
            self.logger.info(f'Loading model from {model_path}.')
            model.load_state_dict(th.load(model_path, map_location=self.device))
        if distributed.is_available():
            if cuda.is_available():
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank)
            else:
                model = nn.parallel.DistributedDataParallelCPU(model)

        return model

    def __init_optimizer(self, optimizer: Type[th.optim.Optimizer], optimizer_opts: Any) -> th.optim.Optimizer:
        _optimizer = optimizer(self.model.parameters(), **dc.asdict(optimizer_opts))
        if self.resume:
            optimizer_path = glob((self.run_dir / 'run_optimizer_*').as_posix()).pop()
            self.logger.info(f'Loading optimizer from {optimizer_path}.')
            _optimizer.load_state_dict(th.load(optimizer_path, map_location=self.device))

        return _optimizer

    def __init_trainer_evaluator(self) -> Tuple[engine.Engine, engine.Engine, engine.Engine]:
        trainer = engine.create_supervised_trainer(self.model, self.optimizer, self.criterion, self.device, True)
        train_evaluator = engine.create_supervised_evaluator(self.model, self.metrics, self.device, True)
        valid_evaluator = engine.create_supervised_evaluator(self.model, self.metrics, self.device, True)

        return trainer, train_evaluator, valid_evaluator

    def __init_handlers(self) -> None:
        # all file handling is done only in the main process.
        if self.rank == 0:
            self.__init_checkpoint_handler()
            self.__init_stats_recorder()
        self.__init_iter_timer_handler()
        self.__init_early_stopping_handler()

        self.trainer.add_event_handler(engine.Events.STARTED, self._on_training_started)
        self.trainer.add_event_handler(engine.Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(engine.Events.ITERATION_COMPLETED, self._on_iteration_completed)
        self.trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.train_evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.valid_evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)

    def __init_iter_timer_handler(self) -> None:
        self.iter_timer = handlers.Timer(average=False)
        self.iter_timer.attach(self.trainer,
                               start=engine.Events.ITERATION_STARTED,
                               step=engine.Events.ITERATION_COMPLETED)

    def __init_checkpoint_handler(self) -> None:
        require_empty = not self.resume
        checkpoint_handler = handlers.ModelCheckpoint(dirname=self.run_dir.as_posix(), filename_prefix='run',
                                                      n_saved=1, require_empty=require_empty, save_as_state_dict=True,
                                                      score_function=self.acc_3, score_name='acc@3')
        checkpoint_args = {
            'model': self.model.module if hasattr(self.model, 'module') else self.model,
            'optimizer': self.optimizer,
        }
        self.valid_evaluator.add_event_handler(engine.Events.COMPLETED, checkpoint_handler, checkpoint_args)

    def __init_early_stopping_handler(self) -> None:
        handler = handlers.EarlyStopping(patience=self.patience, score_function=self.acc_3, trainer=self.trainer)
        self.valid_evaluator.add_event_handler(engine.Events.COMPLETED, handler)

    def __init_stats_recorder(self) -> None:
        fieldnames = ['train_loss', 'valid_loss', 'train_acc@1', 'valid_acc@1', 'train_acc@3', 'valid_acc@3']
        self.csv_file = open((self.run_dir / 'stats.csv').as_posix(), 'a+')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames)
        if not self.resume:
            self.csv_writer.writeheader()

    def _on_training_started(self, _engine: engine.Engine) -> None:
        if self.resume:
            self.logger.info(f'Loading trainer state from {(self.run_dir / "trainer_state.json").as_posix()}')
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'r') as file:
                state = json.load(file)
                _engine.state.iteration = state['iteration']
                _engine.state.epoch = state['epoch']

    def _on_epoch_started(self, _engine: engine.Engine) -> None:
        self.logger.info('TRAINING.')
        if self.data_bunch.train_sampler is not None:
            self.data_bunch.train_sampler.set_epoch(_engine.state.epoch)

    def _on_iteration_completed(self, _engine: engine.Engine) -> None:
        iteration = (_engine.state.iteration - 1) % len(self.data_bunch.train_set) + 1
        if iteration % self.log_interval == 0:
            self.logger.info(f'[Batch: {_engine.state.iteration:04d}/{self.iterations:4d}]'
                             f'[Iteration Time: {self.iter_timer.value():6.2f}s]'
                             f'[Batch Loss: {_engine.state.output:8.4f}]')

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        metrics = {}

        self.logger.info('EVALUATION TRAIN.')
        self.train_evaluator.run(self.data_bunch.train_loader)
        for key, value in self.train_evaluator.state.metrics.items():
            metrics[f'train_{key}'] = value
        msg = (f'[Acc@1: {metrics["train_acc@1"]:.4f}]'
               f'[Acc@3: {metrics["train_acc@3"]:.4f}]'
               f'[Loss: {metrics["train_loss"]:.4f}]')
        self.logger.info(msg)

        self.logger.info('EVALUATION VALID.')
        self.valid_evaluator.run(self.data_bunch.valid_loader)
        for key, value in self.valid_evaluator.state.metrics.items():
            metrics[f'valid_{key}'] = value
        msg = (f'[Acc@1: {metrics["valid_acc@1"]:.4f}]'
               f'[Acc@3: {metrics["valid_acc@3"]:.4f}]'
               f'[Loss: {metrics["valid_loss"]:.4f}]')
        self.logger.info(msg)

        # only write stats and state to file if on main process.
        if self.rank == 0:
            self.csv_writer.writerow(metrics)
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'w') as file:
                state = {
                    'iteration': _engine.state.iteration,
                    'epoch': _engine.state.epoch,
                }
                json.dump(state, file, indent=True)

    def _on_exception_raised(self, _engine: engine.Engine, exception: Exception) -> None:
        if self.rank == 0:
            self.csv_file.close()

        raise exception

    def val_loss(self, _engine: engine.Engine) -> float:
        return -round(_engine.state.metrics['loss'], 4)

    def acc_3(self, _engine: engine.Engine) -> float:
        return round(_engine.state.metrics['acc@3'], 4)

    def run(self) -> None:
        self.trainer.run(self.data_bunch.train_loader, max_epochs=self.epochs)
        if self.rank == 0:
            self.csv_file.close()
