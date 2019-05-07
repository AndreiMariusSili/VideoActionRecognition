import csv
import json
import math
import os
import pathlib as pl
import traceback
from glob import glob
from typing import Any, Dict, TextIO, Tuple, Union

import dataclasses as dc
import torch as th
from ignite import engine, handlers
from torch import cuda, distributed, nn, optim

import constants as ct
import helpers as hp
import models.engine as me
import models.options as mo
import pipeline as pipe
from env import logging


class Run(object):
    rank: int
    word_size: int
    run_dir: pl.Path
    logger: logging.Logger
    device: Any
    notification_sent: bool
    name: str
    mode: str
    resume: bool
    epochs: int
    patience: int
    log_interval: int
    model: nn.Module
    criterion: nn.CrossEntropyLoss
    optimizer: Union[th.optim.Adam, th.optim.SGD, th.optim.RMSprop]
    lr_scheduler: optim.lr_scheduler.ReduceLROnPlateau
    data_bunch: pipe.SmthDataBunch
    metrics: Dict[str, Any]
    trainer: engine.Engine
    train_evaluator: engine.Engine
    valid_evaluator: engine.Engine
    iterations: Union[float, int]
    csv_file: TextIO
    csv_writer: csv.DictWriter
    iter_timer: handlers.Timer

    def __init__(self, opts: mo.RunOptions):
        self.rank, self.world_size = self._init_distributed()
        self.run_dir = self._init_record(opts)
        self.logger = self._init_logging()
        self._log(f'Initializing run {opts.name}.')
        self.device = self._init_device()

        assert opts.mode in ['discriminative', 'variational']

        self.notification_sent = False
        self.name = opts.name
        self.mode = opts.mode
        self.resume = opts.resume
        self.epochs = opts.trainer_opts.epochs
        self.patience = opts.patience
        self.log_interval = opts.log_interval

        self.model = self._init_model(opts.model, opts.model_opts)
        self._log(self.model)
        self.optimizer = self._init_optimizer(opts.trainer_opts.optimizer, opts.trainer_opts.optimizer_opts)
        self._log(self.optimizer)
        self.criterion = opts.trainer_opts.criterion(**dc.asdict(opts.trainer_opts.criterion_opts)).to(self.device)
        self._log(self.criterion)
        self.lr_scheduler = self._init_lr_scheduler()
        self._log(self.lr_scheduler)
        self.data_bunch = self._init_data_bunch(opts)
        self._log(self.data_bunch)

        self.metrics = opts.evaluator_opts.metrics
        self.trainer, self.train_evaluator, self.valid_evaluator = self._init_trainer_evaluator()
        train_examples = len(self.data_bunch.train_set)
        # multiply back by world size to get global batch size since this is changed in _init_data_bunch
        batch_size = self.data_bunch.train_dl_opts.batch_size * self.world_size
        self.iterations = math.ceil(train_examples / batch_size) * self.epochs
        self.csv_file, self.csv_writer, self.iter_timer = self._init_handlers()

    def _init_distributed(self) -> Tuple[int, int]:
        if distributed.is_available():
            if cuda.is_available():
                distributed.init_process_group(backend='nccl', init_method='env://')
            else:
                distributed.init_process_group(backend='gloo', init_method='env://')
            rank = distributed.get_rank()
            world_size = distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size

    def _init_record(self, opts: mo.RunOptions) -> pl.Path:
        """Create the run directory and store the run options."""
        run_dir = ct.SMTH_RUN_DIR / opts.name
        os.makedirs(run_dir.as_posix(), exist_ok=True)
        with open((run_dir / 'options.json').as_posix(), 'w') as file:
            json.dump(dc.asdict(opts), file, indent=True, sort_keys=True, default=str)

        return run_dir

    def _init_logging(self, ):
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

    def _init_device(self) -> Any:
        if self.world_size > 1:
            device = th.device(f'cuda:{self.rank}' if cuda.is_available() else f'cpu')
        else:
            device = th.device(f'cuda' if cuda.is_available() else f'cpu')

        return device

    def _init_model(self, model: nn.Module, opts: Any) -> nn.Module:
        model = model(**dc.asdict(opts))
        model.to(self.device)
        if self.resume:
            model_path = glob((self.run_dir / 'latest_model_*').as_posix()).pop()
            self._log(f'Loading model from {model_path}.')
            model.load_state_dict(th.load(model_path, map_location=self.device))
        if distributed.is_available():
            if cuda.is_available():
                if self.world_size > 1:
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank)
                else:
                    model = nn.parallel.DistributedDataParallel(model)
            else:
                model = nn.parallel.DistributedDataParallelCPU(model)

        return model

    def _init_optimizer(self, optimizer: Any, optimizer_opts: Any) -> Any:
        optimizer = optimizer(self.model.parameters(), **dc.asdict(optimizer_opts))
        if self.resume:
            optimizer_path = glob((self.run_dir / 'latest_optimizer_*').as_posix()).pop()
            self._log(f'Loading optimizer from {optimizer_path}.')
            optimizer.load_state_dict(th.load(optimizer_path, map_location=self.device))

        return optimizer

    def _init_lr_scheduler(self):
        """Initialize a LR scheduler that reduces the LR when there was no improvement for some epochs."""
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode='max',
                                                            factor=0.1,
                                                            patience=self.patience // 2,
                                                            verbose=self.rank == 0)
        if self.resume:
            lr_scheduler_path = glob((self.run_dir / 'latest_lr_scheduler_*').as_posix()).pop()
            self._log(f'Loading LR scheduler from {lr_scheduler_path}.')
            lr_scheduler.load_state_dict(th.load(lr_scheduler_path, map_location=self.device))

        return lr_scheduler

    def _init_data_bunch(self, opts: mo.RunOptions) -> pipe.SmthDataBunch:
        # flag use of distributed sampler
        opts.db_opts.distributed = self.world_size > 1

        # make sure batch and workers are distributed well across worlds.
        opts.train_dl_opts.batch_size = opts.train_dl_opts.batch_size // self.world_size * ct.NUM_DEVICES
        opts.train_dl_opts.num_workers = opts.train_dl_opts.num_workers // self.world_size
        opts.valid_dl_opts.batch_size = opts.valid_dl_opts.batch_size // self.world_size * ct.NUM_DEVICES
        opts.valid_dl_opts.num_workers = opts.valid_dl_opts.num_workers // self.world_size

        return opts.data_bunch(opts.db_opts,
                               opts.train_ds_opts,
                               opts.valid_ds_opts,
                               opts.train_dl_opts,
                               opts.valid_dl_opts)

    def _init_trainer_evaluator(self) -> Tuple[engine.Engine, engine.Engine, engine.Engine]:
        """Initialize the trainer and evaluator engines."""
        if self.mode == 'discriminative':
            trainer = me.create_discriminative_trainer(self.model, self.optimizer, self.criterion, self.device, True)
            train_evaluator = me.create_discriminative_evaluator(self.model, self.metrics, self.device, True)
            valid_evaluator = me.create_discriminative_evaluator(self.model, self.metrics, self.device, True)
        else:
            trainer = me.create_variational_trainer(self.model, self.optimizer, self.criterion, self.device, True)
            train_evaluator = me.create_variational_evaluator(self.model, self.metrics, self.device, True)
            valid_evaluator = me.create_variational_evaluator(self.model, self.metrics, self.device, True)

        return trainer, train_evaluator, valid_evaluator

    def _init_handlers(self) -> Tuple[TextIO, csv.DictWriter, handlers.Timer]:
        """Initialize the handlers of engine events. All file handling in done only in the main process. """
        file, writer = None, None
        if self.rank == 0:
            self._init_checkpoint_handlers()
            file, writer = self._init_stats_recorder()
        self._init_early_stopping_handler()
        iter_timer = self._init_iter_timer_handler()

        self.trainer.add_event_handler(engine.Events.STARTED, self._on_training_started)
        self.trainer.add_event_handler(engine.Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(engine.Events.ITERATION_COMPLETED, self._on_iteration_completed)
        self.trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.trainer.add_event_handler(engine.Events.COMPLETED, self._on_training_completed)

        self.train_evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.valid_evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)

        return file, writer, iter_timer

    def _init_iter_timer_handler(self) -> handlers.Timer:
        """Initialize a timer for each batch processing time."""
        iter_timer = handlers.Timer(average=False)
        iter_timer.attach(self.trainer,
                          start=engine.Events.ITERATION_STARTED,
                          step=engine.Events.ITERATION_COMPLETED)
        return iter_timer

    def _init_checkpoint_handlers(self) -> None:
        """Initialize a handler that will store the state dict of the model ,optimizer and scheduler for the best
        model according to accuracy@2 metric."""
        require_empty = not self.resume
        best_checkpoint_handler = handlers.ModelCheckpoint(dirname=self.run_dir.as_posix(), filename_prefix='best',
                                                           n_saved=1, require_empty=require_empty,
                                                           save_as_state_dict=True,
                                                           score_function=self._acc_2, score_name='acc@2')
        checkpoint_args = {
            'model': self.model.module if hasattr(self.model, 'module') else self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }
        latest_checkpoint_handler = handlers.ModelCheckpoint(dirname=self.run_dir.as_posix(), filename_prefix='latest',
                                                             n_saved=1, require_empty=require_empty,
                                                             save_as_state_dict=True, save_interval=1)
        if self.resume:
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'r') as file:
                state = json.load(file)
                best_checkpoint_handler._iteration = state['epoch']
                latest_checkpoint_handler._iteration = state['epoch']
        self.valid_evaluator.add_event_handler(engine.Events.COMPLETED, best_checkpoint_handler, checkpoint_args)
        self.valid_evaluator.add_event_handler(engine.Events.COMPLETED, latest_checkpoint_handler, checkpoint_args)

    def _init_early_stopping_handler(self) -> None:
        """Initialize a handler that will stop the engine run if no improvement in accuracy@2 has happened for a
        number of epochs."""
        early_stopper = handlers.EarlyStopping(patience=self.patience, score_function=self._acc_2, trainer=self.trainer)
        self.valid_evaluator.add_event_handler(engine.Events.COMPLETED, early_stopper)

    def _init_stats_recorder(self) -> Tuple[TextIO, csv.DictWriter]:
        """Open a file and initialize a dict writer to persist training statistics."""
        if self.mode == 'discriminative':
            fieldnames = ['train_loss', 'valid_loss',
                          'train_acc@1', 'valid_acc@1',
                          'train_acc@2', 'valid_acc@2']
        else:
            fieldnames = [
                'train_mse_loss', 'train_ce_loss', 'train_kld_loss', 'train_total_loss',
                'valid_mse_loss', 'valid_ce_loss', 'valid_kld_loss', 'valid_total_loss',
                'train_acc@1', 'train_acc@2',
                'valid_acc@1', 'valid_acc@2'
            ]
        csv_file = open((self.run_dir / 'stats.csv').as_posix(), 'a+')
        csv_writer = csv.DictWriter(csv_file, fieldnames)
        if not self.resume:
            csv_writer.writeheader()
        return csv_file, csv_writer

    def _on_training_started(self, _engine: engine.Engine) -> None:
        """Event handler for start of training. Evaluates the model on the dataset."""
        if self.resume:
            self._log(f'Loading trainer state from {(self.run_dir / "trainer_state.json").as_posix()}')
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'r') as file:
                state = json.load(file)
                _engine.state.iteration = state['iteration']
                _engine.state.epoch = state['epoch']
        else:
            self._evaluate(_engine)

    def _on_training_completed(self, _engine: engine.Engine) -> None:
        """Event handler for end of training. Sends slack notification."""
        if self.rank == 0 and not self.notification_sent:
            self.logger.info('Training Completed. Sending notification.')
            text = f'Finished training job.'
            train_acc_1 = round(self.train_evaluator.state.metrics["acc@1"], 4)
            valid_acc_1 = round(self.valid_evaluator.state.metrics["acc@1"], 4)
            train_acc_2 = round(self.train_evaluator.state.metrics["acc@2"], 4)
            valid_acc_2 = round(self.valid_evaluator.state.metrics["acc@2"], 4)
            if self.mode == 'discriminative':
                train_loss = round(self.train_evaluator.state.metrics["loss"], 4)
                valid_loss = round(self.valid_evaluator.state.metrics["loss"], 4)
            else:
                metrics = self.train_evaluator.state.metrics
                losses = [metrics["mse_loss"], metrics["ce_loss"], metrics["kld_loss"]]
                train_loss = round(sum(losses), 4)
                metrics = self.valid_evaluator.state.metrics
                losses = [metrics["mse_loss"], metrics["ce_loss"], metrics["kld_loss"]]
                valid_loss = round(sum(losses), 4)

            fields = [
                {
                    'title': 'Epoch',
                    'value': self.trainer.state.epoch,
                    'short': False
                },
                {
                    'title': 'Accuracy@1 (Train / Valid)',
                    'value': f'{train_acc_1} / {valid_acc_1}',
                    'short': False
                },
                {
                    'title': 'Accuracy@2 (Train / Valid)',
                    'value': f'{train_acc_2} / {valid_acc_2}',
                    'short': False
                },
                {
                    'title': 'Loss (Train / Valid)',
                    'value': f'{train_loss} / {valid_loss}',
                    'short': False
                },
            ]
            hp.notify('good', self.name, text, fields)
            self.logger.info(f'Done.')
            self.notification_sent = True

    def _on_epoch_started(self, _engine: engine.Engine) -> None:
        """Event handler for start of epoch. Sets epoch of distributed train sampler if necessary."""
        self._log('TRAINING.')
        if self.data_bunch.train_sampler is not None:
            self.data_bunch.train_sampler.set_epoch(_engine.state.epoch)

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        """Event handler for end of epoch. Evaluates model on training and validation set. Schedules LR."""
        self._evaluate(_engine)
        self._schedule_lr()

    def _on_iteration_completed(self, _engine: engine.Engine) -> None:
        """Event handler for end of batch processing. Logs statistics once in a while."""
        iteration = (_engine.state.iteration - 1) % len(self.data_bunch.train_set) + 1
        if iteration % self.log_interval == 0:
            if self.mode == 'discriminative':
                self._log(f'[Batch: {_engine.state.iteration:04d}/{self.iterations:04d}]'
                          f'[Iteration Time: {self.iter_timer.value():6.2f}s]'
                          f'[Batch Loss: {_engine.state.output:8.4f}]')
            else:
                self._log(f'[Batch: {_engine.state.iteration:04d}/{self.iterations:04d}]'
                          f'[Iteration Time: {self.iter_timer.value():6.2f}s]'
                          f'[MSE Loss: {_engine.state.output[0]:8.4f}]'
                          f'[CE Loss: {_engine.state.output[1]:8.4f}]'
                          f'[KLD Loss: {_engine.state.output[2]:8.4f}]')

    def _on_exception_raised(self, _engine: engine.Engine, exception: Exception) -> None:
        """Event handler for raised exception. Performs cleanup. Sends slack notification."""
        if distributed.is_available():
            distributed.barrier()

        if self.rank == 0:
            self.csv_file.close()
            if not self.notification_sent:
                text = f'Error occurred during training.'
                fields = [
                    {
                        'title': 'Epoch',
                        'value': self.trainer.state.epoch,
                        'short': False
                    },
                    {
                        'title': 'Error',
                        'value': traceback.format_exc(),
                        'short': False
                    }
                ]
                hp.notify('bad', self.name, text, fields)
                self.notification_sent = True

        raise exception

    def _evaluate(self, _engine: engine.Engine) -> None:
        """Evaluate on train and validation set."""
        metrics = {}

        self._log('EVALUATION TRAIN.')
        self.train_evaluator.run(self.data_bunch.train_loader)
        for key, value in self.train_evaluator.state.metrics.items():
            metrics[f'train_{key}'] = value
        if self.mode == 'discriminative':
            msg = (f'[Acc@1: {metrics["train_acc@1"]:.4f}]'
                   f'[Acc@2: {metrics["train_acc@2"]:.4f}]'
                   f'[Loss: {metrics["train_loss"]:.4f}]')
        else:
            all_losses = [metrics["train_mse_loss"], metrics["train_ce_loss"], metrics["train_kld_loss"]]
            msg = (f'[Acc@1: {metrics["train_acc@1"]:.4f}]'
                   f'[Acc@2: {metrics["train_acc@2"]:.4f}]'
                   f'[MSE Loss: {metrics["train_mse_loss"]:.4f}]'
                   f'[CE Loss: {metrics["train_ce_loss"]:.4f}]'
                   f'[KLD Loss: {metrics["train_kld_loss"]:.4f}]'
                   f'[Total Loss: {sum(all_losses):.4f}]')
        self._log(msg)

        self._log('EVALUATION VALID.')
        self.valid_evaluator.run(self.data_bunch.valid_loader)
        for key, value in self.valid_evaluator.state.metrics.items():
            metrics[f'valid_{key}'] = value
        if self.mode == 'discriminative':
            msg = (f'[Acc@1: {metrics["valid_acc@1"]:.4f}]'
                   f'[Acc@2: {metrics["valid_acc@2"]:.4f}]'
                   f'[Loss: {metrics["valid_loss"]:.4f}]')
        else:
            all_losses = [metrics["valid_mse_loss"], metrics["valid_ce_loss"], metrics["valid_kld_loss"]]
            msg = (f'[Acc@1: {metrics["valid_acc@1"]:.4f}]'
                   f'[Acc@2: {metrics["valid_acc@2"]:.4f}]'
                   f'[MSE Loss: {metrics["valid_mse_loss"]:.4f}]'
                   f'[CE Loss: {metrics["valid_ce_loss"]:.4f}]'
                   f'[KLD Loss: {metrics["valid_kld_loss"]:.4f}]'
                   f'[Total Loss: {sum(all_losses):.4f}]')
        self._log(msg)

        # only write stats and state to file if on main process.
        if self.rank == 0:
            self.csv_writer.writerow(metrics)
            self.csv_file.flush()
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'w') as file:
                state = {
                    'iteration': _engine.state.iteration,
                    'epoch': _engine.state.epoch,
                }
                json.dump(state, file, indent=True)

    def _schedule_lr(self, ):
        """Take a scheduler step according to accuracy@2."""
        self.lr_scheduler.step(self.valid_evaluator.state.metrics['acc@2'])

    def _negative_val_loss(self, _engine: engine.Engine) -> float:
        """Return opposite val loss for model evaluation purposes."""
        return -round(_engine.state.metrics['loss'], 4)

    def _acc_2(self, _engine: engine.Engine) -> float:
        """Return accuracy@2 for model evaluation purposes."""
        return round(_engine.state.metrics['acc@2'], 4)

    def _log(self, msg: Any):
        """Log message. Flush file handler."""
        self.logger.info(msg)
        if self.rank == 0:
            self.logger.handlers[0].flush()

    def run(self) -> None:
        """Start a run."""
        if distributed.is_available():
            distributed.barrier()

        self.trainer.run(self.data_bunch.train_loader, max_epochs=self.epochs)

        if distributed.is_available():
            distributed.barrier()

        if self.rank == 0:
            self.csv_file.close()
