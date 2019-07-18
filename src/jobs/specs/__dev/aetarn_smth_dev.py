import ignite.metrics as im
from torch import optim

import constants as ct
import pipeline.smth.databunch as smth
from jobs.specs.__dev._smth_dev import *
from models import ae_tarn, criterion, metrics
from options import model_options

# train_dl_opts = copy.deepcopy(train_dl_opts)
# dev_dl_opts = copy.deepcopy(dev_dl_opts)
# valid_dl_opts = copy.deepcopy(valid_dl_opts)
# train_dl_opts.batch_size = 256
# dev_dl_opts.batch_size = 256
# valid_dl_opts.batch_size = 256

########################################################################################################################
# MODEL AND OPTIMIZER
########################################################################################################################
model_opts = model_options.AETARNOptions(
    num_classes=ct.SMTH_NUM_CLASSES,
    time_steps=4,
    drop_rate=0.0,
    encoder_planes=(16, 32, 64, 128, 256),
    decoder_planes=(256, 128, 64, 32, 16),
)
optimizer_opts = model_options.AdamOptimizerOptions(
    lr=0.001
)
########################################################################################################################
# TRAINER AND EVALUATOR
########################################################################################################################
ce_loss = metrics.AverageMeter(output_transform=lambda x: (x[-2], x[0].shape[0]))
mse_loss = metrics.AverageMeter(output_transform=lambda x: (x[-1], x[0].shape[0]))
total_loss = im.MetricsLambda(lambda x, y: x + y, ce_loss, mse_loss)
trainer_opts = model_options.TrainerOptions(
    epochs=50,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=criterion.AECriterion,
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[1], x[4])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[4])),
        'iou': metrics.MultiLabelIoU(lambda x: (x[1], x[4])),
        'ce_loss': ce_loss,
        'mse_loss': mse_loss,
        'total_loss': total_loss
    }
)
ae_loss_metric = metrics.AELoss(criterion.AECriterion())
total_loss = im.MetricsLambda(lambda x: sum(x), ae_loss_metric)
evaluator_opts = model_options.EvaluatorOptions(
    metrics={
        'acc@1': im.Accuracy(output_transform=lambda x: (x[1], x[4])),
        'acc@5': im.TopKCategoricalAccuracy(k=5, output_transform=lambda x: (x[1], x[4])),
        'iou': metrics.MultiLabelIoU(lambda x: (x[1], x[4])),
        'ce_loss': ae_loss_metric[0],
        'mse_loss': ae_loss_metric[1],
        'total_loss': total_loss,
    }
)
########################################################################################################################
# RUN
########################################################################################################################
aetarn_smth_dev = model_options.RunOptions(
    name='aetarn_smth_dev',
    mode='ae',
    resume=False,
    debug=False,
    log_interval=1,
    patience=50,
    model=ae_tarn.AETimeAlignedResNet,
    model_opts=model_opts,
    data_bunch=smth.SmthDataBunch,
    db_opts=db_opts,
    train_ds_opts=train_ds_opts,
    dev_ds_opts=dev_ds_opts,
    valid_ds_opts=valid_ds_opts,
    train_dl_opts=train_dl_opts,
    dev_dl_opts=dev_dl_opts,
    valid_dl_opts=valid_dl_opts,
    trainer_opts=trainer_opts,
    evaluator_opts=evaluator_opts
)
