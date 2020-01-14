"""Holds specifications for evaluators."""
import options.experiment_options

########################################################################################################################
# EVALUATORS
########################################################################################################################
class_evaluator = options.experiment_options.EvaluatorOptions(
    metrics='eval_class_metrics'
)
class_ae_evaluator = options.experiment_options.EvaluatorOptions(
    metrics='eval_ae_metrics'
)
class_vae_evaluator = options.experiment_options.EvaluatorOptions(
    metrics='eval_vae_metrics'
)
