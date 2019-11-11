"""Holds specifications for trainers."""
import ignite.metrics as im
import torch.nn as nn

import criterion.custom as cc
import metrics.custom as cm
import options.experiment_options
import specs.optimizers as op

########################################################################################################################
# AE HELPERS
########################################################################################################################
ae_ce_loss = im.RunningAverage(cm.AverageMeter(output_transform=lambda x: (x[-2], x[0].shape[0])))
ae_mse_loss = im.RunningAverage(cm.AverageMeter(output_transform=lambda x: (x[-1], x[0].shape[0])))
ae_total_loss = im.RunningAverage(im.MetricsLambda(lambda x, y: x + y, ae_ce_loss, ae_mse_loss))
########################################################################################################################
# VAE HELPERS
########################################################################################################################
vae_ce_loss = im.RunningAverage(cm.AverageMeter(output_transform=lambda x: (x[-4], x[0].shape[0])))
vae_mse_loss = im.RunningAverage(cm.AverageMeter(output_transform=lambda x: (x[-3], x[0].shape[0])))
vae_kld_loss = im.RunningAverage(cm.AverageMeter(output_transform=lambda x: (x[-2], x[0].shape[0])))
vae_kld_factor = im.RunningAverage(cm.AverageMeter(output_transform=lambda x: (x[-1], 1)))
vae_total_loss = im.RunningAverage(im.MetricsLambda(lambda x, y, z: x + y + z, vae_ce_loss, vae_mse_loss, vae_kld_loss))
########################################################################################################################
# TRAINERS
########################################################################################################################
class_trainer = options.experiment_options.TrainerOptions(
    epochs=100,
    optimizer=op.adam,
    criterion=nn.CrossEntropyLoss,
    metrics={
        'acc_1/train': im.RunningAverage(im.Accuracy(output_transform=lambda x: x[1:3])),
        'acc_5/train': im.RunningAverage(im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: x[1:3])),
        'ce_loss/train': im.RunningAverage(im.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[1:3]))
    }
)
class_ae_trainer = options.experiment_options.TrainerOptions(
    epochs=100,
    optimizer=op.adam,
    criterion=cc.AECriterion,
    metrics={
        'acc_1': im.RunningAverage(im.Accuracy(output_transform=lambda x: (x[1], x[4]))),
        'acc_5': im.RunningAverage(im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[4]))),
        'ce_loss': ae_ce_loss,
        'mse_loss': ae_mse_loss,
        'total_loss': ae_total_loss
    }
)
class_vae_trainer = options.experiment_options.TrainerOptions(
    epochs=60,
    optimizer=op.adam,
    criterion=cc.VAECriterion,
    metrics={
        'acc_1': im.RunningAverage(im.Accuracy(output_transform=lambda x: (x[1].squeeze(dim=1), x[6]))),
        'acc_5': im.RunningAverage(
            im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1].squeeze(dim=1), x[6]))),
        'mse_loss': vae_mse_loss,
        'ce_loss': vae_ce_loss,
        'kld_loss': vae_kld_loss,
        'total_loss': vae_total_loss,
        'kld_factor': vae_kld_factor,
    }
)
