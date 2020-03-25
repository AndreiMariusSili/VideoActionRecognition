import typing as typ

import ignite.engine as ien

import options.experiment_options
import pro.engine as pe
import pro.runners._runner_base as _base
import specs.maps as sm


class ClassRunner(_base.BaseRunner):
    def __init__(self, opts: options.experiment_options.ExperimentOptions, local_rank: int):
        super(ClassRunner, self).__init__(opts, local_rank)
        assert self.opts.model.type == 'class'

    def _init_engines(self) -> typ.Tuple[ien.Engine, ien.Engine]:
        trainer_metrics = sm.Metrics[self.opts.trainer.metrics].value
        evaluator_metrics = sm.Metrics[self.opts.evaluator.metrics].value

        trainer = pe.create_cls_trainer(self.model,
                                        self.optimizer,
                                        self.criterion,
                                        trainer_metrics,
                                        self.device)
        evaluator = pe.create_cls_evaluator(self.model,
                                            evaluator_metrics,
                                            self.device)

        return trainer, evaluator

    def _init_runner_specific_handlers(self):
        pass
