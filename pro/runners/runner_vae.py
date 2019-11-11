import typing as typ

import ignite.engine as ie

import constants as ct
import options.experiment_options
import pro.engine as pe
import pro.runners._runner_base as _base


class VariationalAutoEncoderRunner(_base.BaseRunner):

    def __init__(self, opts: options.experiment_options.ExperimentOptions, local_rank: int):
        super(VariationalAutoEncoderRunner, self).__init__(opts, local_rank)
        assert self.opts.mode == 'ae'

    def _init_engines(self) -> typ.Tuple[ie.Engine, ie.Engine]:
        trainer = pe.create_vae_trainer(self.model,
                                        self.optimizer,
                                        self.criterion,
                                        self.opts.trainer.metrics,
                                        self.device)
        evaluator = pe.create_vae_evaluator(self.model,
                                            self.opts.evaluator.metrics,
                                            self.device,
                                            ct.VAE_NUM_SAMPLES_DEV)

        return trainer, evaluator

    def _init_runner_specific_handlers(self):
        self.trainer.add_event_handler(ie.Events.EPOCH_COMPLETED, self._step_kld)

    def _step_kld(self, _engine: ie.Engine) -> None:
        """Gradually increase the KLD from null to full."""
        if _engine.state.epoch != 0 and _engine.state.epoch % ct.KLD_STEP_INTERVAL == 0:
            step = (_engine.state.epoch + 1) // ct.KLD_STEP_INTERVAL
            self.criterion.kld_factor = min(1.0, step * ct.KLD_STEP_SIZE)
