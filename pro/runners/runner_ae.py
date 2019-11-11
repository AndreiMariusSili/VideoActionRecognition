import typing as typ

import ignite.engine as ien

import options.experiment_options
import pro.engine as pe
import pro.runners._runner_base as _base


class AutoEncoderRunner(_base.BaseRunner):
    def __init__(self, opts: options.experiment_options.ExperimentOptions, local_rank: int):
        super(AutoEncoderRunner, self).__init__(opts, local_rank)
        assert self.opts.mode == 'ae'

    def _init_engines(self) -> typ.Tuple[ien.Engine, ien.Engine]:
        trainer = pe.create_ae_trainer(self.model,
                                       self.optimizer,
                                       self.criterion,
                                       self.opts.trainer.metrics,
                                       self.device)
        evaluator = pe.create_ae_evaluator(self.model,
                                           self.opts.evaluator.metrics,
                                           self.device)

        return trainer, evaluator

    def _init_runner_specific_handlers(self):
        pass
