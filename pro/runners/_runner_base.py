import abc
import copy
import dataclasses as dc
import glob
import json
import pathlib as pth
import typing as typ

import ignite.engine as ie
import ignite.handlers as ih
import torch as th
import torch.cuda as cuda
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

import constants as ct
import databunch.databunch as db
import options.experiment_options as eo
import pro.logger as pl


class BaseRunner(abc.ABC):
    def __init__(self, opts: eo.ExperimentOptions, local_rank: int):
        self.opts = opts
        self.local_rank = local_rank
        self.rank, self.world_size, self.main_proc, self.distributed = self._init_distributed()
        self.run_dir = self._init_run()
        self.logger = pl.ExperimentLogger(self.opts, self.main_proc, self.run_dir,
                                          self.opts.trainer.metrics, self.opts.evaluator.metrics)

        self.device = self._init_device()
        self.data_bunch, self.opts.model_opts.num_classes = self._init_databunch()
        self.model = self._init_model()
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_lr_scheduler()
        self.logger.init_log(self.data_bunch, self.model, self.criterion, self.optimizer, self.lr_scheduler)

        self.trainer, self.evaluator = self._init_engines()

        self._init_events()

        self.metrics = {}

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

    def _init_databunch(self) -> typ.Tuple[db.VideoDataBunch, int]:
        """Init databunch optionally for distributed setting. Calculate number of classes."""

        # flag use of distributed sampler; set start method to prevent deadlocks.
        self.opts.databunch_opts.distributed = self.distributed
        if self.distributed:
            mp.set_start_method('forkserver')

        # make sure batch and workers are distributed well across worlds. -1 for the main process
        self.opts.databunch_opts.dlo.batch_size //= self.world_size
        self.opts.databunch_opts.dlo.num_workers //= self.world_size
        self.opts.databunch_opts.dlo.num_workers -= 1

        data_bunch = db.VideoDataBunch(db_opts=self.opts.databunch_opts)

        return data_bunch, len(data_bunch.lids)

    def _init_model(self) -> nn.Module:
        """Initialize, resume model."""
        opts = dc.asdict(copy.deepcopy(self.opts.model_opts))
        model = opts.pop('arch')(**opts).to(self.device)
        if self.opts.resume:
            latest_models = list(glob.glob((self.run_dir / 'latest_model_*.pth').as_posix()))
            if len(latest_models) > 1:
                raise ValueError('More than one latest model available. Remove old versions.')
            model_path = latest_models.pop()
            self.logger.log(f'Loading model from {model_path}...')
            model.load_state_dict(th.load(model_path, map_location=self.device))

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank)

        if self.opts.debug:
            for module in model.modules():
                if type(module) in [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]:
                    module.track_running_stats = False
                elif type(module) in [nn.Dropout, nn.Dropout2d, nn.Dropout3d]:
                    module.p = 0.0

        return model

    def _init_criterion(self):
        """Initialize loss function to correct device."""
        criterion = self.opts.trainer.criterion(**dc.asdict(self.opts.trainer.criterion_opts))

        return criterion.to(self.device)

    def _init_optimizer(self) -> optim.Adam:
        opts = dc.asdict(self.opts.trainer.optimizer)
        optimizer = opts.pop('algorithm')(self.model.parameters(), **opts)

        if self.opts.resume:
            optimizer_path = glob.glob((self.run_dir / 'latest_optimizer_*').as_posix()).pop()
            self.logger.log(f'Loading optimizer from {optimizer_path}...')
            optimizer.load_state_dict(th.load(optimizer_path, map_location=self.device))

        return optimizer

    def _init_lr_scheduler(self):
        """Initialize a LR scheduler that reduces the LR when there was no improvement for some epochs."""
        lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                      milestones=ct.LR_MILESTONES,
                                                      gamma=ct.LR_GAMMA)

        if self.opts.resume:
            lr_scheduler_path = glob.glob((self.run_dir / 'latest_lr_scheduler_*').as_posix()).pop()
            self.logger.log(f'Loading LR scheduler from {lr_scheduler_path}...')
            lr_scheduler.load_state_dict(th.load(lr_scheduler_path, map_location=self.device))

        return lr_scheduler

    @abc.abstractmethod
    def _init_engines(self) -> typ.Tuple[ie.Engine, ie.Engine]:
        """Initialize the trainer and evaluator engines."""
        raise NotImplementedError

    @abc.abstractmethod
    def _init_runner_specific_handlers(self):
        """Register handlers specific to runner type."""

    def _init_events(self) -> None:
        """Initialize the handlers of engine events. All file handling in done only in the main process. """
        self.trainer.add_event_handler(ie.Events.STARTED, self._resume_trainer_state)
        self.trainer.add_event_handler(ie.Events.EPOCH_STARTED, self._set_distributed_sampler_seed)
        self.trainer.add_event_handler(ie.Events.ITERATION_COMPLETED, self._aggregate_metrics)
        self.logger.attach_train_pbar(self.trainer)
        self.trainer.add_event_handler(ie.Events.EPOCH_COMPLETED, self._evaluate)
        self.trainer.add_event_handler(ie.Events.COMPLETED, self._close_logger)
        self.trainer.add_event_handler(ie.Events.EXCEPTION_RAISED, self._graceful_shutdown)

        self.evaluator.add_event_handler(ie.Events.ITERATION_COMPLETED, self._aggregate_metrics)
        self.evaluator.add_event_handler(ie.Events.EPOCH_COMPLETED, self.logger.log_dev_metrics)

        if self.main_proc:
            latest_ckpt_handler, best_ckpt_handler, ckpt_args = self._init_checkpoint_handlers()
            self.evaluator.add_event_handler(ie.Events.COMPLETED, latest_ckpt_handler, ckpt_args)
            self.evaluator.add_event_handler(ie.Events.COMPLETED, best_ckpt_handler, ckpt_args)

        self.evaluator.add_event_handler(ie.Events.EXCEPTION_RAISED, self._graceful_shutdown)

        self._init_runner_specific_handlers()
        self.logger.init_handlers(self.trainer, self.evaluator, self.model, self.optimizer)

    def _init_checkpoint_handlers(self) -> typ.Tuple[ih.ModelCheckpoint, ih.ModelCheckpoint, dict]:
        """Initialize a handler that will store the state dict of the model ,optimizer and scheduler for the best
        and latest models."""
        require_empty = not self.opts.resume
        ckpt_args = {
            'model': self.model.module if hasattr(self.model, 'module') else self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }

        best_ckpt = ih.ModelCheckpoint(dirname=self.run_dir.as_posix(), filename_prefix='best',
                                       n_saved=1, require_empty=require_empty,
                                       save_as_state_dict=True,
                                       score_function=self._negative_val_loss,
                                       score_name='loss')
        latest_ckpt = ih.ModelCheckpoint(dirname=self.run_dir.as_posix(),
                                         filename_prefix='latest',
                                         n_saved=1, require_empty=require_empty,
                                         save_as_state_dict=True, save_interval=1)
        if self.opts.resume:
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'r') as file:
                state = json.load(file)
                best_ckpt._iteration = state['epoch']
                latest_ckpt._iteration = state['epoch']
        return latest_ckpt, best_ckpt, ckpt_args

    def _resume_trainer_state(self, _engine: ie.Engine) -> None:
        """Event handler for start of training. Resume trainer state."""
        if self.opts.resume:
            self.logger.log(f'Loading trainer state from {(self.run_dir / "trainer_state.json").as_posix()}')
            with open((self.run_dir / 'trainer_state.json').as_posix(), 'r') as file:
                state = json.load(file)
                _engine.state.iteration = state['iteration']
                _engine.state.epoch = state['epoch']

    def _set_distributed_sampler_seed(self, _engine: ie.Engine) -> None:
        """Event handler for start of epoch. Sets epoch of distributed train sampler for seeding."""
        if self.distributed:
            self.data_bunch.train_sampler.set_epoch(_engine.state.epoch)

    def _graceful_shutdown(self, _engine: ie.Engine, exception: Exception) -> None:
        """Event handler for raised exception. Performs cleanup. Sends slack notification."""
        self._close_logger(_engine)

        raise exception

    def _close_logger(self, _: ie.Engine):
        self.logger.close()

    def _save_trainer_state(self, _engine: ie.Engine):
        with open((self.run_dir / 'trainer_state.json').as_posix(), 'w') as file:
            state = {
                'iteration': _engine.state.iteration,
                'epoch': _engine.state.epoch,
            }
        json.dump(state, file, indent=True)

    def _evaluate(self, _engine: ie.Engine) -> None:
        """Evaluate on dev set."""
        self.evaluator.run(self.data_bunch.dev_loader)

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

    def _negative_val_loss(self, _: ie.Engine) -> float:
        """Return negative CE."""
        return round(-self.evaluator.state.metrics['ce_loss'], 4)

    def run(self) -> None:
        """Start a run."""
        self.trainer.run(self.data_bunch.train_loader, max_epochs=self.opts.trainer.epochs)

    def _reduce_loss(self, loss: th.Tensor):
        """Average loss across processes and move to cpu."""
        if self.distributed:
            dist.all_reduce(loss)
        return loss.cpu()
