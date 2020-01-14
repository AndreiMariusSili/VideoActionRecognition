import typing as typ

import ignite.engine as ie

import constants as ct
import options.experiment_options
import pro.engine as pe
import pro.runners._runner_base as _base
import specs.maps as sm


class VariationalAutoEncoderRunner(_base.BaseRunner):

    def __init__(self, opts: options.experiment_options.ExperimentOptions, local_rank: int):
        super(VariationalAutoEncoderRunner, self).__init__(opts, local_rank)
        assert self.opts.model.type == 'vae'

    def _init_engines(self) -> typ.Tuple[ie.Engine, ie.Engine]:
        trainer_metrics = sm.Metrics[self.opts.trainer.metrics].value
        evaluator_metrics = sm.Metrics[self.opts.evaluator.metrics].value

        trainer = pe.create_vae_trainer(self.model,
                                        self.optimizer,
                                        self.criterion,
                                        trainer_metrics,
                                        self.device)
        evaluator = pe.create_vae_evaluator(self.model,
                                            evaluator_metrics,
                                            self.device,
                                            ct.VAE_NUM_SAMPLES_DEV)

        return trainer, evaluator

    def _init_runner_specific_handlers(self):
        self.trainer.add_event_handler(ie.Events.EPOCH_COMPLETED, self._step_kld)

    def _step_kld(self, _engine: ie.Engine) -> None:
        """Gradually increase the KLD from null to full."""
        step = 1.0 / self.opts.trainer.kld_warmup_epochs
        self.criterion.kld_factor = min(1.0, _engine.state.epoch * step)
