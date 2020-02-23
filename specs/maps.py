import enum
import typing as t

import ignite.metrics as im
import torch.nn as nn

import criterion.custom as cc
import metrics.custom as cm
import models as mo

########################################################################################################################
# AE HELPERS
########################################################################################################################
train_ae_ce_loss = im.RunningAverage(output_transform=lambda x: x[-2])
train_ae_l1_loss = im.RunningAverage(output_transform=lambda x: x[-1])
train_ae_total_loss = im.RunningAverage(output_transform=lambda x: sum([x[-2], x[-1]]))
eval_ae_loss_metric = cm.AELoss(cc.AECriterion())
eval_ae_total_loss = im.MetricsLambda(lambda x: sum(x), eval_ae_loss_metric)

########################################################################################################################
# VAE HELPERS
########################################################################################################################
train_vae_ce_loss = im.RunningAverage(output_transform=lambda x: x[-4])
train_vae_l1_loss = im.RunningAverage(output_transform=lambda x: x[-3])
train_vae_kld_loss = im.RunningAverage(output_transform=lambda x: x[-2])
train_vae_kld_factor = im.RunningAverage(output_transform=lambda x: x[-1])
train_vae_total_loss = im.RunningAverage(output_transform=lambda x: sum([x[-4], x[-3], x[-2]]))
eval_vae_loss_metric = cm.VAELoss(cc.VAECriterion())
eval_vae_total_loss = im.MetricsLambda(lambda x: sum(x), eval_vae_loss_metric)


########################################################################################################################
# MODELS
########################################################################################################################
class Models(enum.Enum):
    tadn: t.Type[mo.tadn.TimeAlignedDenseNet] = mo.tadn.TimeAlignedDenseNet
    tarn: t.Type[mo.tarn.TimeAlignedResNet] = mo.tarn.TimeAlignedResNet
    tarn_ae: t.Type[mo.tarn.AETimeAlignedResNet] = mo.tarn.AETimeAlignedResNet
    tarn_vae: t.Type[mo.tarn.VAETimeAlignedResNet] = mo.tarn.VAETimeAlignedResNet
    i3d: t.Type[mo.i3d.I3D] = mo.i3d.I3D
    i3d_ae: t.Type[mo.i3d.AEI3D] = mo.i3d.AEI3D
    i3d_vae: t.Type[mo.i3d.VAEI3D] = mo.i3d.VAEI3D


########################################################################################################################
# LOSSES
########################################################################################################################
class Criteria(enum.Enum):
    class_criterion: t.Type[nn.CrossEntropyLoss] = nn.CrossEntropyLoss
    ae_criterion: t.Type[cc.AECriterion] = cc.AECriterion
    vae_criterion: t.Type[cc.VAECriterion] = cc.VAECriterion


########################################################################################################################
# METRICS
########################################################################################################################
class Metrics(enum.Enum):
    train_class_metrics: t.Dict[str, im.Metric] = {
        'acc_1': im.RunningAverage(im.Accuracy(output_transform=lambda x: x[1:3])),
        'acc_5': im.RunningAverage(im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: x[1:3])),
        'ce_loss': im.RunningAverage(output_transform=lambda x: x[0]),
        'total_loss': im.RunningAverage(output_transform=lambda x: x[0])
    }
    train_ae_metrics: t.Dict[str, im.Metric] = {
        'acc_1': im.RunningAverage(im.Accuracy(output_transform=lambda x: (x[1], x[5]))),
        'acc_5': im.RunningAverage(im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[5]))),
        'ce_loss': train_ae_ce_loss,
        'l1_loss': train_ae_l1_loss,
        'total_loss': train_ae_total_loss
    }
    train_vae_metrics: t.Dict[str, im.Metric] = {
        'acc_1': im.RunningAverage(im.Accuracy(output_transform=lambda x: (x[1].squeeze(dim=1), x[7]))),
        'acc_5': im.RunningAverage(
            im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1].squeeze(dim=1), x[7]))),
        'ce_loss': train_vae_ce_loss,
        'l1_loss': train_vae_l1_loss,
        'kld_loss': train_vae_kld_loss,
        'total_loss': train_vae_total_loss,
        'kld_factor': train_vae_kld_factor,
    }
    eval_class_metrics = {
        'acc_1': im.Accuracy(output_transform=lambda x: x[0:2]),
        'acc_5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: x[0:2]),
        'ce_loss': im.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[0:2]),
        'total_loss': im.Loss(nn.CrossEntropyLoss(), output_transform=lambda x: x[0:2])
    }
    eval_ae_metrics = {
        'acc_1': im.Accuracy(output_transform=lambda x: (x[1], x[5])),
        'acc_5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[5])),
        'ce_loss': eval_ae_loss_metric[0],
        'l1_loss': eval_ae_loss_metric[1],
        'total_loss': eval_ae_total_loss,
    }
    eval_vae_metrics = {
        'acc_1': im.Accuracy(output_transform=lambda x: (x[-1], x[-2])),
        'acc_5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[-1], x[-2])),
        'ce_loss': eval_vae_loss_metric[0],
        'l1_loss': eval_vae_loss_metric[1],
        'kld_loss': eval_vae_loss_metric[2],
        'total_loss': eval_vae_total_loss,
    }
