import typing as typ

import ignite.engine as ie

import options.experiment_options as eo
import pro.engine as pe
import pro.evaluators._evaluator_base as _base


class ClassEvaluator(_base.BaseEvaluator):
    def __init__(self, opts: eo.ExperimentOptions, local_rank: int):
        super(ClassEvaluator, self).__init__(opts, local_rank)

    def _init_evaluators(self) -> typ.Tuple[ie.Engine, ie.Engine, ie.Engine]:
        train_evaluator = pe.create_ae_evaluator(self.model, self.opts.evaluator.metrics, self.device, True)
        dev_evaluator = pe.create_ae_evaluator(self.model, self.opts.evaluator.metrics, self.device, True)
        test_evaluator = pe.create_ae_evaluator(self.model, self.opts.evaluator.metrics, self.device, True)

        return train_evaluator, dev_evaluator, test_evaluator
