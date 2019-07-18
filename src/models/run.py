import copy
import csv
import dataclasses as dc
import json
import math
import os
import pathlib as pl
import traceback
from glob import glob
from typing import Any, Dict, TextIO, Tuple, Union

import torch as th
from ignite import engine, handlers
from ignite.contrib.handlers.tensorboard_logger import *
from torch import cuda, distributed, nn, optim

import constants as ct
import helpers as hp
import models.engine as me
import options.model_options as mo
import pipeline.smth.databunch as smth
from env import logging


class Run(object):
    opts: mo.RunOptions
    local_rank: int
    rank: int
    word_size: int
    main_proc: bool
    distributed: bool
    run_dir: pl.Path
    cut: str
    logger: logging.Logger
    device: Any
    notification_sent: bool
    name: str
    mode: str
    resume: bool
    debug: bool
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
    tb_logger: TensorboardLogger
    csv_file: TextIO
    csv_writer: csv.DictWriter
    iter_timer: handlers.Timer
    metrics: Dict[str, Any]

    def __init__(self, opts: mo.RunOptions, local_rank: int):
        assert opts.mode in ['class', 'ae', 'vae']
        self.local_rank = local_rank
        self.notification_sent = False
        self.name = opts.name
        self.mode = opts.mode
        self.resume = opts.resume
        self.debug = opts.debug
        self.epochs = opts.trainer_opts.epochs
        self.patience = opts.patience
        self.log_interval = opts.log_interval
        self.opts = copy.deepcopy(opts)

        self.rank, self.world_size, self.main_proc, self.distributed = self._init_distributed()
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
        self.iterations = math.ceil(train_examples / batch_size)
        self.csv_file, self.csv_writer, self.iter_timer, self.tb_logger = self._init_handlers()

        self.metrics = {}

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

    def _init_record(self, opts: mo.RunOptions) -> Tuple[pl.Path, str]:
        """Create the run directory and store the run options."""
        cut = f'__{self.name.split("_").pop()}'
        run_dir = ct.SMTH_RUN_DIR / cut / self.name

        os.makedirs(run_dir.as_posix(), exist_ok=True)
        with open((run_dir / 'options.json').as_posix(), 'w') as file:
            json.dump(dc.asdict(opts), file, indent=True, sort_keys=True, default=str)

        return run_dir, cut

    def _init_logging(self, ):
        """Get a logger that outputs to console and file. Only log messages if on the main process."""
        logger = logging.getLogger()
        if self.main_proc:
            formatter = logging.Formatter('[%(asctime)-s][%(process)d][%(levelname)s]\t%(message)s')
            file_handler = logging.FileHandler(self.run_dir / f'training.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            logger.handlers = []

        return logger

    def _init_device(self) -> Any:
        if self.distributed:
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
        if distributed.is_available() and self.local_rank != -1:
            if cuda.is_available():
                if self.distributed:
                    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                    model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank,
                                                                find_unused_parameters=True)
                else:
                    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
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
                                                            verbose=self.main_proc)
        if self.resume:
            lr_scheduler_path = glob((self.run_dir / 'latest_lr_scheduler_*').as_posix()).pop()
            self._log(f'Loading LR scheduler from {lr_scheduler_path}.')
            lr_scheduler.load_state_dict(th.load(lr_scheduler_path, map_location=self.device))

        return lr_scheduler

    def _init_data_bunch(self, opts: mo.RunOptions) -> smth.SmthDataBunch:
        # flag use of distributed sampler
        opts.db_opts.distributed = self.distributed

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

    def _init_handlers(self) -> Tuple[TextIO, csv.DictWriter, handlers.Timer, TensorboardLogger]:
        """Initialize the handlers of engine events. All file handling in done only in the main process. """
        tb_logger = None

        self.evaluator.add_event_handler(engine.Events.COMPLETED, self._aggregate_metrics, split='dev')
        self.evaluator.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)

        self._init_checkpoint_handlers()
        self._init_early_stopping_handler()
        file, writer = self._init_stats_recorder()
        iter_timer = self._init_iter_timer_handler()

        self.trainer.add_event_handler(engine.Events.STARTED, self._on_training_started)
        self.trainer.add_event_handler(engine.Events.EPOCH_STARTED, self._on_epoch_started)
        self.trainer.add_event_handler(engine.Events.ITERATION_COMPLETED, self._on_iteration_completed)
        self.trainer.add_event_handler(engine.Events.EPOCH_COMPLETED, self._on_epoch_completed)
        self.trainer.add_event_handler(engine.Events.EXCEPTION_RAISED, self._on_exception_raised)
        self.trainer.add_event_handler(engine.Events.COMPLETED, self._on_training_completed)

        if self.debug:
            tb_logger = TensorboardLogger(log_dir=self.run_dir.as_posix())
            tb_logger.attach(self.trainer, log_handler=GradsScalarHandler(self.model),
                             event_name=engine.Events.ITERATION_COMPLETED)
            tb_logger.attach(self.trainer, log_handler=GradsHistHandler(self.model),
                             event_name=engine.Events.EPOCH_COMPLETED)

        return file, writer, iter_timer, tb_logger

    def _init_iter_timer_handler(self) -> handlers.Timer:
        """Initialize a timer for each batch processing time."""
        iter_timer = handlers.Timer(average=False)
        iter_timer.attach(self.trainer,
                          start=engine.Events.ITERATION_STARTED,
                          step=engine.Events.ITERATION_COMPLETED)
        return iter_timer

    def _init_checkpoint_handlers(self) -> None:
        """Initialize a handler that will store the state dict of the model ,optimizer and scheduler for the best
        and latest models."""
        if self.main_proc:
            require_empty = not self.resume
            checkpoint_args = {
                'model': self.model.module if hasattr(self.model, 'module') else self.model,
                'optimizer': self.optimizer,
                'lr_scheduler': self.lr_scheduler
            }
            best_checkpoint_handler = handlers.ModelCheckpoint(dirname=self.run_dir.as_posix(), filename_prefix='best',
                                                               n_saved=1, require_empty=require_empty,
                                                               save_as_state_dict=True,
                                                               score_function=self._negative_val_loss,
                                                               score_name='loss')
            latest_checkpoint_handler = handlers.ModelCheckpoint(dirname=self.run_dir.as_posix(),
                                                                 filename_prefix='latest',
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
        early_stopper = handlers.EarlyStopping(patience=self.patience,
                                               score_function=self._negative_val_loss,
                                               trainer=self.trainer)
        self.evaluator.add_event_handler(engine.Events.COMPLETED, early_stopper)

    def _init_stats_recorder(self) -> Tuple[TextIO, csv.DictWriter]:
        """Open a file and initialize a dict writer to persist training statistics."""
        csv_file, csv_writer = None, None
        if self.main_proc:
            fieldnames = [
                'train_acc@1', 'dev_acc@1',
                'train_acc@5', 'dev_acc@5',
                'train_iou', 'dev_iou',
                'train_ce_loss', 'dev_ce_loss',
            ]
            if self.mode == 'ae':
                fieldnames.extend([
                    'train_mse_loss', 'dev_mse_loss',
                    'train_total_loss', 'dev_total_loss',
                ])
            elif self.mode == 'vae':
                fieldnames.extend([
                    'train_mse_loss', 'dev_mse_loss',
                    'train_kld_loss', 'dev_kld_loss',
                    'train_total_loss', 'dev_total_loss',
                    'train_kld_factor'
                ])
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
        if self.main_proc and not self.notification_sent:
            if self.tb_logger:
                self.tb_logger.close()

            self._log('Training Completed. Sending notification.')
            text = f'Finished training job.'
            fields = [
                {
                    'title': 'Epoch',
                    'value': self.trainer.state.epoch,
                    'short': False
                },
                {
                    'title': 'Accuracy@1 (Train / Dev)',
                    'value': f'{self.metrics["train_acc@1"]:.4f} / {self.metrics["dev_acc@1"]:.4f}',
                    'short': False
                },
                {
                    'title': 'Accuracy@5 (Train / Dev)',
                    'value': f'{self.metrics["train_acc@5"]:.4f} / {self.metrics["dev_acc@5"]:.4f}',
                    'short': False
                },
                {
                    'title': 'IoU (Train / Dev)',
                    'value': f'{self.metrics["train_iou"]:.4f} / {self.metrics["dev_iou"]:.4f}',
                    'short': False
                },
                {
                    'title': 'CE Loss (Train / Dev)',
                    'value': f'{self.metrics["train_ce_loss"]:.4f} / {self.metrics["dev_ce_loss"]:.4f}',
                    'short': False
                },
            ]
            if self.mode in ['ae', 'vae']:
                fields.append({
                    'title': 'MSE Loss (Train / Dev)',
                    'value': f'{self.metrics["train_mse_loss"]:.4f} / {self.metrics["dev_mse_loss"]:.4f}',
                    'short': False
                })
                if self.mode in ['vae']:
                    fields.append({
                        'title': 'KLD Loss (Train / Dev)',
                        'value': f'{self.metrics["train_kld_loss"]:.4f} / {self.metrics["dev_kld_loss"]:.4f}',
                        'short': False
                    })
                fields.append({
                    'title': 'Total Loss (Train / Dev)',
                    'value': f'{self.metrics["train_total_loss"]:.4f} / {self.metrics["dev_total_loss"]:.4f}',
                    'short': False
                })
            hp.notify('good', self.name, text, fields)
            self._log(f'Done.')
            self.notification_sent = True

    def _on_epoch_started(self, _engine: engine.Engine) -> None:
        """Event handler for start of epoch. Sets epoch of distributed train sampler if necessary."""
        self._log('TRAINING.')
        if self.data_bunch.train_sampler is not None:
            self.data_bunch.train_sampler.set_epoch(_engine.state.epoch)

    def _on_epoch_completed(self, _engine: engine.Engine) -> None:
        """Event handler for end of epoch. Evaluates model on training and validation set. Schedules LR."""
        self._aggregate_metrics(_engine, 'train')
        self._evaluate(_engine)
        self._schedule_lr()
        if self.mode == 'vae':
            self._step_kld(_engine)

    def _on_iteration_completed(self, _engine: engine.Engine) -> None:
        """Event handler for end of batch processing. Logs statistics once in a while."""
        iteration = (_engine.state.iteration - 1) % self.iterations + 1
        if iteration % self.log_interval == 0:
            if self.mode == 'class':
                ce_loss = th.tensor([_engine.state.output[0]], requires_grad=False, device=self.device)
                if self.distributed:
                    distributed.all_reduce(ce_loss)
                ce_loss = ce_loss.cpu()
                ce_loss = float(ce_loss[0].item()) / self.world_size

                self._log(f'[Epoch: {_engine.state.epoch:03d}/{_engine.state.max_epochs:03d}]'
                          f'[Batch: {iteration:04d}/{self.iterations:04d}]'
                          f'[Time: {self.iter_timer.value():4.2f}s]'
                          f'[CE Loss: {ce_loss:10.4f}]')
            elif self.mode == 'ae':
                ce_mse_loss = [_engine.state.output[-2], _engine.state.output[-1]]
                ce_mse_loss = th.tensor(ce_mse_loss, requires_grad=False, device=self.device)
                if self.distributed:
                    distributed.all_reduce(ce_mse_loss)
                ce_mse_loss = ce_mse_loss.cpu()
                ce_loss = float(ce_mse_loss[0]) / self.world_size
                mse_loss = float(ce_mse_loss[1]) / self.world_size

                self._log(f'[Epoch: {_engine.state.epoch:03d}/{_engine.state.max_epochs:03d}]'
                          f'[Batch: {iteration:04d}/{self.iterations:04d}]'
                          f'[Time: {self.iter_timer.value():4.2f}s]'
                          f'[CE Loss: {ce_loss:10.4f}]'
                          f'[MSE Loss: {mse_loss:10.4f}]')
            else:
                kld_factor = _engine.state.output[-1]
                ce_mse_kld_loss = [_engine.state.output[-4], _engine.state.output[-3], _engine.state.output[-2]]
                ce_mse_kld_loss = th.tensor(ce_mse_kld_loss, requires_grad=False, device=self.device)
                if self.distributed:
                    distributed.all_reduce(ce_mse_kld_loss)
                ce_mse_kld_loss = ce_mse_kld_loss.cpu()
                ce_loss = float(ce_mse_kld_loss[0]) / self.world_size
                mse_loss = float(ce_mse_kld_loss[1]) / self.world_size
                kld_loss = float(ce_mse_kld_loss[2]) / self.world_size

                self._log(f'[Epoch: {_engine.state.epoch:03d}/{_engine.state.max_epochs:03d}]'
                          f'[Batch: {iteration:04d}/{self.iterations:04d}]'
                          f'[Time: {self.iter_timer.value():2.2f}s]'
                          f'[CE Loss: {ce_loss:10.4f}]'
                          f'[MSE Loss: {mse_loss:10.4f}]'
                          f'[KLD Loss: {kld_factor:10.4f} x {kld_loss:8.4f}]')

    def _on_exception_raised(self, _engine: engine.Engine, exception: Exception) -> None:
        """Event handler for raised exception. Performs cleanup. Sends slack notification."""
        if distributed.is_available() and self.local_rank != -1:
            distributed.barrier()

        if self.main_proc:
            if self.tb_logger:
                self.tb_logger.close()
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
        self.evaluator.run(self.data_bunch.dev_loader)

        msg = (f'[{"TRAIN EVALUATION":<16}]'
               f'[Acc@1: {self.metrics["train_acc@1"]:6.4f}]'
               f'[Acc@5: {self.metrics["train_acc@5"]:6.4f}]'
               f'[IoU: {self.metrics["train_iou"]:6.4f}]'
               f'[CE Loss: {self.metrics["train_ce_loss"]:10.4f}]')
        if self.mode in ['ae', 'vae']:
            msg += f'[MSE Loss: {self.metrics["train_mse_loss"]:10.4f}]'
            if self.mode in ['vae']:
                msg += f'[KLD Loss: {self.metrics["train_kld_loss"]:10.4f}]'
            msg += f'[Total Loss: {self.metrics["train_total_loss"]:10.4f}]'
        self._log(msg)

        msg = (f'[{"DEV EVALUATION":<16}]'
               f'[Acc@1: {self.metrics["dev_acc@1"]:6.4f}]'
               f'[Acc@5: {self.metrics["dev_acc@5"]:6.4f}]'
               f'[IoU: {self.metrics["dev_iou"]:6.4f}]'
               f'[CE Loss: {self.metrics["dev_ce_loss"]:10.4f}]')
        if self.mode in ['ae', 'vae']:
            msg += f'[MSE Loss: {self.metrics["dev_mse_loss"]:10.4f}]'
            if self.mode in ['vae']:
                msg += f'[KLD Loss: {self.metrics["dev_kld_loss"]:10.4f}]'
            msg += f'[Total Loss: {self.metrics["dev_total_loss"]:10.4f}]'
        self._log(msg)

        # only write stats and state to file if on main process.
        if self.main_proc:
            self.csv_writer.writerow(self.metrics)
            self.csv_file.flush()
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'w') as file:
                state = {
                    'iteration': _engine.state.iteration,
                    'epoch': _engine.state.epoch,
                }
                json.dump(state, file, indent=True)

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

    def _schedule_lr(self):
        """Take a scheduler step according to validation loss."""
        self.lr_scheduler.step(self._negative_val_loss(self.evaluator))

    def _step_kld(self, _engine: engine.Engine) -> None:
        """Gradually increase the KLD from null to full."""
        if _engine.state.epoch != 0 and _engine.state.epoch % ct.KLD_STEP_INTERVAL == 0:
            step = (_engine.state.epoch + 1) // ct.KLD_STEP_INTERVAL
            self.criterion.kld_factor = min(1.0, step * ct.KLD_STEP_SIZE)

    def _negative_val_loss(self, _: engine.Engine) -> float:
        """Return negative CE in discriminative setting and negative (CE + MSE) in autoencoder setting."""
        if self.mode == 'class':
            return round(-self.metrics['dev_ce_loss'], 4)
        else:
            return round(-(self.metrics['dev_ce_loss'] + self.metrics['dev_mse_loss']), 4)

    def _acc_1(self, _engine: engine.Engine) -> float:
        """Return accuracy@1 for model evaluation purposes."""
        return round(_engine.state.metrics['acc@1'], 4)

    def _acc_5(self, _engine: engine.Engine) -> float:
        """Return accuracy@2 for model evaluation purposes."""
        return round(_engine.state.metrics['acc@5'], 4)

    def _log(self, msg: Any):
        """Log message. Flush file handler."""
        self.logger.info(msg)
        if self.main_proc:
            self.logger.handlers[0].flush()

    def run(self) -> None:
        """Start a run."""
        if distributed.is_available() and self.local_rank != -1:
            distributed.barrier()

        self.trainer.run(self.data_bunch.train_loader, max_epochs=self.epochs)

        if distributed.is_available() and self.local_rank != -1:
            distributed.barrier()

        if self.main_proc:
            self.csv_file.close()
