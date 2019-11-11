"""Holds specifications for evaluators."""
import ignite.metrics as im
import torch.nn as nn

import criterion.custom as cc
import metrics.custom as cm
import options.experiment_options

########################################################################################################################
# AE HELPERS
########################################################################################################################
ae_loss_metric = cm.AELoss(cc.AECriterion())
ae_total_loss = im.MetricsLambda(lambda x: sum(x), ae_loss_metric)
########################################################################################################################
# VAE HELPERS
########################################################################################################################
vae_loss_metric = cm.VAELoss(cc.VAECriterion())
vae_total_loss = im.MetricsLambda(lambda x: sum(x), vae_loss_metric)
########################################################################################################################
# EVALUATORS
########################################################################################################################
class_evaluator = options.experiment_options.EvaluatorOptions(
    metrics={
        'acc_1': im.Accuracy(output_transform=lambda x: x[0:2]),
        'acc_5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: x[0:2]),
        'ce_loss': im.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[0:2])
    }
)
class_ae_evaluator = options.experiment_options.EvaluatorOptions(
    metrics={
        'acc_1': im.Accuracy(output_transform=lambda x: (x[1], x[4])),
        'acc_5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[4])),
        'ce_loss': ae_loss_metric[0],
        'mse_loss': ae_loss_metric[1],
        'total_loss': ae_total_loss,
    }
)
class_vae_evaluator = options.experiment_options.EvaluatorOptions(
    metrics={
        'acc_1': im.Accuracy(output_transform=lambda x: (x[-1], x[-2])),
        'acc_5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[-1], x[-2])),
        'ce_loss': vae_loss_metric[0],
        'mse_loss': vae_loss_metric[1],
        'kld_loss': vae_loss_metric[2],
        'total_loss': vae_total_loss,
    }
)
