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
import torch.nn as nn
import torch.optim as optim

import constants as ct
import databunch.databunch as db
import helpers as ghp
import logger as pl
import options.experiment_options as eo
import specs.maps as sm


class BaseRunner(abc.ABC):
    class _Decorator:
        @staticmethod
        def sync(func):
            def inner(self, *args, **kwargs):
                if self.distributed:
                    dist.barrier()

                res = func(self, *args, **kwargs)

                if self.distributed:
                    dist.barrier()

                return res

            return inner

        @staticmethod
        def main_proc_only(func):
            def inner(self, *args, **kwargs):
                if not self.main_proc:
                    return

                return func(self, *args, **kwargs)

            return inner

    def __init__(self, opts: eo.ExperimentOptions, local_rank: int):
        self.opts = opts
        self.opts.debug = self.opts.overfit or self.opts.dev
        self.local_rank = local_rank
        self.rank, self.world_size, self.main_proc, self.distributed = self._init_distributed()
        self.opts.world_size, self.opts.distributed = self.world_size, self.distributed
        self.opts.run_dir = self._init_run()

        self.device = self._init_device()
        self.data_bunch, self.opts.model.opts.num_classes = self._init_databunch()
        self.model, self.opts.model.size = self._init_model()
        self.criterion = self._init_criterion()
        self.optimizer = self._init_optimizer()
        self.lr_scheduler = self._init_lr_scheduler()

        self.logger = pl.ExperimentLogger(
            self.opts,
            self.main_proc,
            self.opts.trainer.metrics,
            self.opts.evaluator.metrics
        )
        self.logger.init_log(self.data_bunch, self.model, self.criterion, self.optimizer, self.lr_scheduler)
        self.logger.persist_run_opts()

        self.trainer, self.evaluator = self._init_engines()
        self._init_events()

    def _init_distributed(self) -> typ.Tuple[int, int, bool, bool]:
        """Create distributed setup."""
        if dist.is_available() and self.local_rank != -1:
            dist.init_process_group(backend='nccl')
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert world_size == th.cuda.device_count(), 'Invalid distributed init. World size should equal gpus.'
        else:
            rank = 0
            world_size = 1

        return rank, world_size, rank == 0, world_size > 1

    @_Decorator.sync
    def _init_run(self) -> pth.Path:
        """Create the run directory and store the run options."""
        run_dir = ct.WORK_ROOT / ct.RUNS_ROOT / self.opts.name
        if self.main_proc:
            run_dir.mkdir(parents=True, exist_ok=True)
            if self.opts.debug:
                import os
                import shutil
                for path in os.listdir(run_dir.as_posix()):
                    path = (run_dir / path).as_posix()
                    if os.path.isfile(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
        return run_dir.relative_to(ct.WORK_ROOT)

    @_Decorator.sync
    def _init_device(self) -> typ.Any:
        if self.distributed:
            device = th.device(f'cuda:{self.rank}' if cuda.is_available() else f'cpu')
        else:
            device = th.device(f'cuda' if cuda.is_available() else f'cpu')

        return device

    @_Decorator.sync
    def _init_databunch(self) -> typ.Tuple[db.VideoDataBunch, int]:
        """Init databunch optionally for distributed setting. Calculate number of num_classes."""
        # flag use of distributed sampler
        self.opts.databunch.distributed = self.distributed

        # add batch size from model specification.
        self.opts.databunch.dlo.batch_size = self.opts.model.opts.batch_size

        # when running overfit, keep batch size of data taking into account world size and use only train set.
        if self.opts.overfit:
            self.opts.trainer.epochs = 64
            self.opts.databunch.dlo.batch_size = 1
            self.opts.databunch.train_dso.setting = 'eval'
            self.opts.databunch.dev_dso.meta_path = self.opts.databunch.train_dso.meta_path
            self.opts.databunch.test_dso.meta_path = self.opts.databunch.train_dso.meta_path
            self.opts.databunch.train_dso.keep = self.opts.databunch.dlo.batch_size * self.world_size
            self.opts.databunch.dev_dso.keep = self.opts.databunch.dlo.batch_size * self.world_size
            self.opts.databunch.test_dso.keep = self.opts.databunch.dlo.batch_size * self.world_size

        # when running dev, keep 2 batches, limit epochs and load data in main thread.
        if self.opts.dev:
            self.opts.databunch.dlo.batch_size = 2
            self.opts.trainer.epochs = 4
            self.opts.databunch.dlo.num_workers = 0
            self.opts.databunch.dlo.timeout = 0

            self.opts.databunch.train_dso.keep = self.opts.databunch.dlo.batch_size * self.world_size * 2
            self.opts.databunch.dev_dso.keep = self.opts.databunch.dlo.batch_size * self.world_size * 2
            self.opts.databunch.test_dso.keep = self.opts.databunch.dlo.batch_size * self.world_size * 2

        data_bunch = db.VideoDataBunch(db_opts=self.opts.databunch)

        return data_bunch, len(data_bunch.lids)

    @_Decorator.sync
    def _init_model(self) -> typ.Tuple[nn.Module, str]:
        """Initialize, resume model."""
        num_segments = self.opts.databunch.so.num_segments
        self.opts.model.opts.time_steps = num_segments
        opts = dc.asdict(copy.deepcopy(self.opts.model.opts))
        del opts['batch_size']
        model = sm.Models[self.opts.model.arch].value(**opts).to(self.device)

        if self.opts.resume:
            latest_models = list(glob.glob(str(ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'latest_model_*.pth')))
            if len(latest_models) > 1:
                raise ValueError('More than one latest model available. Remove old versions.')
            model_path = latest_models.pop()
            print(f'Loading model from {model_path}...')
            model.load_state_dict(th.load(model_path, map_location=self.device))

        if self.opts.overfit:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                    module.momentum = 1.0

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank)

        return model, f'{ghp.count_parameters(model):,}'

    @_Decorator.sync
    def _init_criterion(self):
        """Initialize loss function to correct device."""
        criterion = sm.Criteria[self.opts.trainer.criterion].value()

        return criterion.to(self.device)

    @_Decorator.sync
    def _init_optimizer(self) -> optim.Adam:
        opts = dc.asdict(self.opts.trainer.optim_opts)
        # noinspection PyUnresolvedReferences
        optimizer = th.optim.AdamW(self.model.parameters(), **opts)

        if self.opts.resume:
            optimizer_path = glob.glob(str(ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'latest_optimizer_*')).pop()
            print(f'Loading optimizer from {optimizer_path}...')
            optimizer.load_state_dict(th.load(optimizer_path, map_location=self.device))

        return optimizer

    @_Decorator.sync
    def _init_lr_scheduler(self):
        """Initialize a LR scheduler that reduces the LR when there was no improvement for some epochs."""
        lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                      milestones=self.opts.trainer.lr_milestones,
                                                      gamma=self.opts.trainer.lr_gamma)

        if self.opts.resume:
            lr_scheduler_path = glob.glob(
                str(ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'latest_lr_scheduler_*')).pop()
            print(f'Loading LR scheduler from {lr_scheduler_path}...')
            lr_scheduler.load_state_dict(th.load(lr_scheduler_path, map_location=self.device))

        return lr_scheduler

    @_Decorator.sync
    @abc.abstractmethod
    def _init_engines(self) -> typ.Tuple[ie.Engine, ie.Engine]:
        """Initialize the trainer and evaluator engines."""
        raise NotImplementedError

    @_Decorator.sync
    @abc.abstractmethod
    def _init_runner_specific_handlers(self):
        """Register handlers specific to runner type."""

    @_Decorator.sync
    def _init_events(self) -> None:
        """Initialize the handlers of engine events. All file handling in done only in the main process. """
        self.trainer.add_event_handler(ie.Events.STARTED, self._resume_trainer_state)
        self.trainer.add_event_handler(ie.Events.EPOCH_STARTED, self._set_distributed_sampler_seed)
        self.trainer.add_event_handler(ie.Events.ITERATION_COMPLETED, self._aggregate_metrics)
        self.logger.attach_pbar(self.trainer)  # ON ITERATION_COMPLETED
        self.logger.init_handlers(self.trainer, self.evaluator, self.model, self.optimizer)  # ON EPOCH_COMPLETED
        self.trainer.add_event_handler(ie.Events.EPOCH_COMPLETED, self._evaluate)
        self.trainer.add_event_handler(ie.Events.COMPLETED, self._end_run)
        self.trainer.add_event_handler(ie.Events.EXCEPTION_RAISED, self._graceful_shutdown)

        self.evaluator.add_event_handler(ie.Events.ITERATION_COMPLETED, self._aggregate_metrics)
        self.evaluator.add_event_handler(ie.Events.EPOCH_COMPLETED, self.logger.log_dev_metrics)

        # only main process returns checkpoint handlers
        handlers = self._init_checkpoint_handlers()
        if handlers is not None:
            latest_ckpt_handler, best_ckpt_handler, ckpt_args = handlers
            self.evaluator.add_event_handler(ie.Events.COMPLETED, latest_ckpt_handler, ckpt_args)
            self.evaluator.add_event_handler(ie.Events.COMPLETED, best_ckpt_handler, ckpt_args)
            self.evaluator.add_event_handler(ie.Events.COMPLETED, self._save_trainer_state)

        self.evaluator.add_event_handler(ie.Events.EXCEPTION_RAISED, self._graceful_shutdown)
        self._init_runner_specific_handlers()

    @_Decorator.sync
    @_Decorator.main_proc_only
    def _init_checkpoint_handlers(self) -> typ.Tuple[ih.ModelCheckpoint, ih.ModelCheckpoint, dict]:
        """Initialize a handler that will store the state dict of the model ,optimizer and scheduler for the best
        and latest models."""
        require_empty = not self.opts.resume
        ckpt_dir = ct.WORK_ROOT / self.opts.run_dir / 'ckpt'
        ckpt_args = {
            'model': self.model.module if hasattr(self.model, 'module') else self.model,  # noqa
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler
        }
        best_ckpt = ih.ModelCheckpoint(dirname=ckpt_dir.as_posix(), filename_prefix='best',
                                       n_saved=1, require_empty=require_empty,
                                       save_as_state_dict=True,
                                       score_function=self._dev_acc_1,
                                       score_name='dev_acc_1')
        latest_ckpt = ih.ModelCheckpoint(dirname=ckpt_dir.as_posix(),
                                         filename_prefix='latest',
                                         n_saved=1, require_empty=require_empty,
                                         save_as_state_dict=True, save_interval=1)
        if self.opts.resume:
            with open((ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'trainer_state.json').as_posix(), 'r') as file:
                state = json.load(file)
                best_ckpt._iteration = state['epoch']
                latest_ckpt._iteration = state['epoch']
        return latest_ckpt, best_ckpt, ckpt_args

    def _save_trainer_state(self, _engine: ie.Engine):
        with open((ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'trainer_state.json').as_posix(), 'w') as file:
            state = {
                'iteration': self.trainer.state.iteration,
                'epoch': self.trainer.state.epoch,
            }
            json.dump(state, file, indent=True)

    def _resume_trainer_state(self, _: ie.Engine) -> None:
        """Event handler for start of training. Resume trainer state."""
        if self.opts.resume:
            print(
                f'Loading trainer state from {str(ct.WORK_ROOT / self.opts.run_dir / "ckpt" / "trainer_state.json")}')
            with open(str(ct.WORK_ROOT / self.opts.run_dir / 'ckpt' / 'trainer_state.json'), 'r') as file:
                state = json.load(file)
                self.trainer.state.iteration = state['iteration']
                self.trainer.state.epoch = state['epoch']

    def _set_distributed_sampler_seed(self, _engine: ie.Engine) -> None:
        """Event handler for start of epoch. Sets epoch of distributed train sampler for seeding."""
        if self.distributed:
            self.data_bunch.train_sampler.set_epoch(_engine.state.epoch)

    def _end_run(self, _: ie.Engine):
        if self.local_rank != -1 and dist.is_initialized():
            dist.barrier()
        self.logger.close()
        if self.local_rank != -1 and dist.is_initialized():
            dist.destroy_process_group()

    def _graceful_shutdown(self, _engine: ie.Engine, exception: Exception) -> None:
        """Event handler for raised exception. Performs cleanup. Sends slack notification."""
        self._end_run(_engine)

        raise exception

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

    def _dev_acc_1(self, _: ie.Engine):
        return self.evaluator.state.metrics['acc_1']

    def _neg_dev_total_loss(self, _: ie.Engine):
        return -self.evaluator.state.metrics['total_loss']

    def run(self) -> None:
        """Start a run."""
        if self.distributed:
            dist.barrier()
        self.trainer.run(self.data_bunch.train_loader, max_epochs=self.opts.trainer.epochs)

    def _reduce_loss(self, loss: th.Tensor):
        """Average loss across processes and move to cpu."""
        if self.distributed:
            dist.all_reduce(loss)
        return loss.cpu()
