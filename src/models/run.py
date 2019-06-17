import copy
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
import options.model_options as mo
import pipeline.smth.databunch as smth
from env import logging


class Run(object):
    opts: mo.RunOptions
    rank: int
    word_size: int
    run_dir: pl.Path
    cut: str
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
    data_bunch: smth.SmthDataBunch
    valid_metrics: Dict[str, Any]
    trainer: engine.Engine
    evaluator: engine.Engine
    iterations: Union[float, int]
    csv_file: TextIO
    csv_writer: csv.DictWriter
    iter_timer: handlers.Timer

    def __init__(self, opts: mo.RunOptions):
        assert opts.mode in ['class', 'ae', 'vae']

        self.notification_sent = False
        if opts.mode == 'class':
            self.name = opts.name
        elif opts.mode == 'ae':
            self.name = f'{opts.name}@' \
                f'{opts.trainer_opts.criterion_opts.mse_factor}_' \
                f'{opts.trainer_opts.criterion_opts.ce_factor}'
        else:
            self.name = f'{opts.name}@' \
                f'{opts.trainer_opts.criterion_opts.mse_factor}_' \
                f'{opts.trainer_opts.criterion_opts.ce_factor}_' \
                f'{opts.trainer_opts.criterion_opts.kld_factor}'
        self.mode = opts.mode
        self.resume = opts.resume
        self.epochs = opts.trainer_opts.epochs
        self.patience = opts.patience
        self.log_interval = opts.log_interval
        self.opts = copy.deepcopy(opts)

        self.rank, self.world_size = self._init_distributed()
        self.run_dir, self.cut = self._init_record(opts)
        self.logger = self._init_logging()
        self._log(f'Initializing run {opts.name}.')
        self.device = self._init_device()

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

        self.train_metrics = opts.trainer_opts.metrics
        self.dev_metrics = opts.evaluator_opts.metrics
        self.trainer, self.evaluator = self._init_trainer_evaluator()
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

    def _init_record(self, opts: mo.RunOptions) -> Tuple[pl.Path, str]:
        """Create the run directory and store the run options."""
        cut = f'__{self.name.split("@").pop(0).split("_").pop()}'
        run_dir = ct.SMTH_RUN_DIR / cut / self.name

        os.makedirs(run_dir.as_posix(), exist_ok=True)
        with open((run_dir / 'options.json').as_posix(), 'w') as file:
            json.dump(dc.asdict(opts), file, indent=True, sort_keys=True, default=str)

        return run_dir, cut

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
            latest_models = glob((self.run_dir / 'latest_model_*.pth').as_posix())
            if len(latest_models) > 1:
                raise ValueError('More than one latest model available. Remove old versions.')
            
            model_path = latest_models.pop()
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
                                                            factor=0.5,
                                                            patience=ct.LR_PATIENCE,
                                                            verbose=True)
        if self.resume:
            lr_scheduler_path = glob((self.run_dir / 'latest_lr_scheduler_*').as_posix()).pop()
            self._log(f'Loading LR scheduler from {lr_scheduler_path}.')
            lr_scheduler.load_state_dict(th.load(lr_scheduler_path, map_location=self.device))

        return lr_scheduler

    def _init_data_bunch(self, opts: mo.RunOptions) -> smth.SmthDataBunch:
        # flag use of distributed sampler
        opts.db_opts.distributed = self.world_size > 1

        # make sure batch and workers are distributed well across worlds.
        opts.train_dl_opts.batch_size = opts.train_dl_opts.batch_size // self.world_size
        opts.train_dl_opts.num_workers = opts.train_dl_opts.num_workers // self.world_size
        opts.dev_dl_opts.batch_size = opts.dev_dl_opts.batch_size // self.world_size
        opts.dev_dl_opts.num_workers = opts.dev_dl_opts.num_workers // self.world_size
        opts.valid_dl_opts.batch_size = opts.valid_dl_opts.batch_size // self.world_size
        opts.valid_dl_opts.num_workers = opts.valid_dl_opts.num_workers // self.world_size

        return opts.data_bunch(db_opts=opts.db_opts,
                               train_ds_opts=opts.train_ds_opts,
                               dev_ds_opts=opts.dev_ds_opts,
                               valid_ds_opts=opts.valid_ds_opts,
                               train_dl_opts=opts.train_dl_opts,
                               dev_dl_opts=opts.dev_dl_opts,
                               valid_dl_opts=opts.valid_dl_opts)

    def _init_trainer_evaluator(self) -> Tuple[engine.Engine, engine.Engine]:
        """Initialize the trainer and evaluator engines."""
        if self.mode == 'class':
            trainer = me.create_cls_trainer(self.model, self.optimizer, self.criterion, self.train_metrics, self.device)
            evaluator = me.create_cls_evaluator(self.model, self.dev_metrics, self.device)
        elif self.mode == 'ae':
            trainer = me.create_ae_trainer(self.model, self.optimizer, self.criterion, self.train_metrics, self.device)
            evaluator = me.create_ae_evaluator(self.model, self.dev_metrics, self.device)
        else:
            trainer = me.create_vae_trainer(self.model, self.optimizer, self.criterion, self.train_metrics, self.device)
            evaluator = me.create_vae_evaluator(self.model, self.dev_metrics, self.device)

        return trainer, evaluator

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

        self.evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)

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
                                                           score_function=self._acc_1, score_name='acc@1')
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
        self.evaluator.add_event_handler(engine.Events.COMPLETED, best_checkpoint_handler, checkpoint_args)
        self.evaluator.add_event_handler(engine.Events.COMPLETED, latest_checkpoint_handler, checkpoint_args)

    def _init_early_stopping_handler(self) -> None:
        """Initialize a handler that will stop the engine run if no improvement in accuracy@2 has happened for a
        number of epochs."""
        early_stopper = handlers.EarlyStopping(patience=self.patience, score_function=self._acc_1, trainer=self.trainer)
        self.evaluator.add_event_handler(engine.Events.COMPLETED, early_stopper)

    def _init_stats_recorder(self) -> Tuple[TextIO, csv.DictWriter]:
        """Open a file and initialize a dict writer to persist training statistics."""
        if self.mode == 'class':
            fieldnames = [
                'train_acc@1', 'dev_acc@1',
                'train_acc@5', 'dev_acc@5',
                'train_iou', 'dev_iou',
                'train_loss', 'dev_loss',
            ]
        elif self.mode == 'ae':
            fieldnames = [
                'train_acc@1', 'train_acc@5', 'train_iou',
                'dev_acc@1', 'dev_acc@5', 'dev_iou',
                'train_mse_loss', 'train_ce_loss', 'train_total_loss',
                'dev_mse_loss', 'dev_ce_loss', 'dev_total_loss'
            ]
        else:
            fieldnames = [
                'train_acc@1', 'train_acc@5', 'train_iou',
                'dev_acc@1', 'dev_acc@5', 'dev_iou',
                'train_mse_loss', 'train_ce_loss', 'train_kld_loss', 'train_total_loss',
                'dev_mse_loss', 'dev_ce_loss', 'dev_kld_loss', 'dev_total_loss',
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

    def _on_training_completed(self, _engine: engine.Engine) -> None:
        """Event handler for end of training. Sends slack notification."""
        if self.rank == 0 and not self.notification_sent:
            self.logger.info('Training Completed. Sending notification.')
            text = f'Finished training job.'
            train_acc_1 = round(self.trainer.state.metrics["acc@1"], 4)
            dev_acc_1 = round(self.evaluator.state.metrics["acc@1"], 4)
            train_acc_5 = round(self.trainer.state.metrics["acc@5"], 4)
            dev_acc_5 = round(self.evaluator.state.metrics["acc@5"], 4)
            train_iou = round(self.trainer.state.metrics["iou"], 4)
            dev_iou = round(self.evaluator.state.metrics["iou"], 4)
            if self.mode == 'class':
                train_loss = round(self.trainer.state.metrics["loss"], 4)
                valid_loss = round(self.evaluator.state.metrics["loss"], 4)

                fields = [
                    {
                        'title': 'Epoch',
                        'value': self.trainer.state.epoch,
                        'short': False
                    },
                    {
                        'title': 'Accuracy@1 (Train / Valid)',
                        'value': f'{train_acc_1} / {dev_acc_1}',
                        'short': False
                    },
                    {
                        'title': 'Accuracy@2 (Train / Valid)',
                        'value': f'{train_acc_5} / {dev_acc_5}',
                        'short': False
                    },
                    {
                        'title': 'IoU (Train / Valid)',
                        'value': f'{train_iou} / {dev_iou}',
                        'short': False
                    },
                    {
                        'title': 'Loss (Train / Valid)',
                        'value': f'{train_loss} / {valid_loss}',
                        'short': False
                    },
                ]
            elif self.mode == 'ae':
                train_metrics = self.trainer.state.metrics
                valid_metrics = self.evaluator.state.metrics

                fields = [
                    {
                        'title': 'Epoch',
                        'value': self.trainer.state.epoch,
                        'short': False
                    },
                    {
                        'title': 'Accuracy@1 (Train / Valid)',
                        'value': f'{train_acc_1} / {dev_acc_1}',
                        'short': False
                    },
                    {
                        'title': 'Accuracy@2 (Train / Valid)',
                        'value': f'{train_acc_5} / {dev_acc_5}',
                        'short': False
                    },
                    {
                        'title': 'IoU (Train / Valid)',
                        'value': f'{train_iou} / {dev_iou}',
                        'short': False
                    },
                    {
                        'title': 'MSE Loss (Train / Valid)',
                        'value': f'{round(train_metrics["mse_loss"], 4)} / {round(valid_metrics["mse_loss"], 4)}',
                        'short': False
                    },
                    {
                        'title': 'CE Loss (Train / Valid)',
                        'value': f'{round(train_metrics["ce_loss"], 4)} / {round(valid_metrics["ce_loss"], 4)}',
                        'short': False
                    },
                    {
                        'title': 'Total Loss (Train / Valid)',
                        'value': f'{round(train_metrics["total_loss"], 4)} / {round(valid_metrics["total_loss"], 4)}',
                        'short': False
                    },
                ]
            else:
                train_metrics = self.trainer.state.metrics
                valid_metrics = self.evaluator.state.metrics

                fields = [
                    {
                        'title': 'Epoch',
                        'value': self.trainer.state.epoch,
                        'short': False
                    },
                    {
                        'title': 'Accuracy@1 (Train / Valid)',
                        'value': f'{train_acc_1} / {dev_acc_1}',
                        'short': False
                    },
                    {
                        'title': 'Accuracy@2 (Train / Valid)',
                        'value': f'{train_acc_5} / {dev_acc_5}',
                        'short': False
                    },
                    {
                        'title': 'IoU (Train / Valid)',
                        'value': f'{train_iou} / {dev_iou}',
                        'short': False
                    },
                    {
                        'title': 'MSE Loss (Train / Valid)',
                        'value': f'{round(train_metrics["mse_loss"], 4)} / {round(valid_metrics["mse_loss"], 4)}',
                        'short': False
                    },
                    {
                        'title': 'CE Loss (Train / Valid)',
                        'value': f'{round(train_metrics["ce_loss"], 4)} / {round(valid_metrics["ce_loss"], 4)}',
                        'short': False
                    },
                    {
                        'title': 'KLD Loss (Train / Valid)',
                        'value': f'{round(train_metrics["kld_loss"], 4)} / {round(valid_metrics["kld_loss"], 4)}',
                        'short': False
                    },
                    {
                        'title': 'Total Loss (Train / Valid)',
                        'value': f'{round(train_metrics["total_loss"], 4)} / {round(valid_metrics["total_loss"], 4)}',
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
            if self.mode == 'class':
                self._log(f'[Batch: {_engine.state.iteration:04d}/{self.iterations:04d}]'
                          f'[Iteration Time: {self.iter_timer.value():6.2f}s]'
                          f'[Batch Loss: {_engine.state.output[0]:8.4f}]')
            elif self.mode == 'ae':
                self._log(f'[Batch: {_engine.state.iteration:04d}/{self.iterations:04d}]'
                          f'[Iteration Time: {self.iter_timer.value():6.2f}s]'
                          f'[MSE Loss: {_engine.state.output[-2]:8.4f}]'
                          f'[CE Loss: {_engine.state.output[-1]:8.4f}]')
            else:
                self._log(f'[Batch: {_engine.state.iteration:04d}/{self.iterations:04d}]'
                          f'[Iteration Time: {self.iter_timer.value():6.2f}s]'
                          f'[MSE Loss: {_engine.state.output[-3]:8.4f}]'
                          f'[CE Loss: {_engine.state.output[-2]:8.4f}]'
                          f'[KLD Loss: {_engine.state.output[-1]:8.4f}]')

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
        for key, value in self.trainer.state.metrics.items():
            metrics[f'train_{key}'] = value
        if self.mode == 'class':
            msg = (f'[Acc@1: {metrics["train_acc@1"]:.4f}]'
                   f'[Acc@5: {metrics["train_acc@5"]:.4f}]'
                   f'[IoU: {metrics["train_iou"]:.4f}]'
                   f'[Loss: {metrics["train_loss"]:.4f}]')
        elif self.mode == 'ae':
            msg = (f'[Acc@1: {metrics["train_acc@1"]:.4f}]'
                   f'[Acc@5: {metrics["train_acc@5"]:.4f}]'
                   f'[IoU: {metrics["train_iou"]:.4f}]'
                   f'[MSE Loss: {metrics["train_mse_loss"]:.4f}]'
                   f'[CE Loss: {metrics["train_ce_loss"]:.4f}]'
                   f'[Total Loss: {metrics["train_total_loss"]:.4f}]')
        else:
            msg = (f'[Acc@1: {metrics["train_acc@1"]:.4f}]'
                   f'[Acc@5: {metrics["train_acc@5"]:.4f}]'
                   f'[IoU: {metrics["train_iou"]:.4f}]'
                   f'[MSE Loss: {metrics["train_mse_loss"]:.4f}]'
                   f'[CE Loss: {metrics["train_ce_loss"]:.4f}]'
                   f'[KLD Loss: {metrics["train_kld_loss"]:.4f}]'
                   f'[Total Loss: {metrics["train_total_loss"]:.4f}]')
        self._log(msg)

        self._log('EVALUATION DEV.')
        self.evaluator.run(self.data_bunch.dev_loader)
        for key, value in self.evaluator.state.metrics.items():
            metrics[f'dev_{key}'] = value
        if self.mode == 'class':
            msg = (f'[Acc@1: {metrics["dev_acc@1"]:.4f}]'
                   f'[Acc@5: {metrics["dev_acc@5"]:.4f}]'
                   f'[IoU: {metrics["dev_iou"]:.4f}]'
                   f'[Loss: {metrics["dev_loss"]:.4f}]')
        elif self.mode == 'ae':
            msg = (f'[Acc@1: {metrics["dev_acc@1"]:.4f}]'
                   f'[Acc@5: {metrics["dev_acc@5"]:.4f}]'
                   f'[IoU: {metrics["dev_iou"]:.4f}]'
                   f'[MSE Loss: {metrics["dev_mse_loss"]:.4f}]'
                   f'[CE Loss: {metrics["dev_ce_loss"]:.4f}]'
                   f'[Total Loss: {metrics["dev_total_loss"]:.4f}]')
        else:
            msg = (f'[Acc@1: {metrics["dev_acc@1"]:.4f}]'
                   f'[Acc@5: {metrics["dev_acc@5"]:.4f}]'
                   f'[IoU: {metrics["dev_iou"]:.4f}]'
                   f'[MSE Loss: {metrics["dev_mse_loss"]:.4f}]'
                   f'[CE Loss: {metrics["dev_ce_loss"]:.4f}]'
                   f'[KLD Loss: {metrics["dev_kld_loss"]:.4f}]'
                   f'[Total Loss: {metrics["dev_total_loss"]:.4f}]')
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
        self.lr_scheduler.step(self.evaluator.state.metrics['acc@1'])

    def _negative_val_loss(self, _engine: engine.Engine) -> float:
        """Return opposite val loss for model evaluation purposes."""
        return -round(_engine.state.metrics['loss'], 4)

    def _acc_1(self, _engine: engine.Engine) -> float:
        """Return accuracy@1 for model evaluation purposes."""
        return round(_engine.state.metrics['acc@1'], 4)

    def _acc_2(self, _engine: engine.Engine) -> float:
        """Return accuracy@2 for model evaluation purposes."""
        return round(_engine.state.metrics['acc@5'], 4)

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
