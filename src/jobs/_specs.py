from torch import optim, nn
from ignite import metrics

import pipeline as pipe
import models


data_bunch_opts = pipe.DataBunchOptions(
    shape='volume',
    size=120,
    test=False
)
data_set_opts = pipe.DataSetOptions(
    cut=0.5,
    keep=5
)
data_loader_opts = pipe.DataLoaderOptions(
    batch_size=1,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=False
)
model_opts = models.VideoLSTMOptions(
    num_classes=10,
    freeze_conv=True,
    freeze_fc=False
)
optimizer_opts = models.AdamOptimizerOptions(
    lr=0.01
)
trainer_opts = models.TrainerOptions(
    epochs=1,
    optimizer=optim.Adam,
    optimizer_opts=optimizer_opts,
    criterion=nn.CrossEntropyLoss
)
evaluator_opts = models.EvaluatorOptions(
    metrics={
        'acc@1': metrics.Accuracy(),
        'acc@3': metrics.TopKCategoricalAccuracy(k=3),
        'loss': metrics.Loss(nn.CrossEntropyLoss())
    }
)

video_lstm = models.RunOptions(
    name='video_lstm',
    resume=False,
    resume_from=None,
    log_interval=10,
    model=models.VideoLSTM,
    model_opts=model_opts,
    data_bunch=pipe.SmthDataBunch,
    data_bunch_opts=data_bunch_opts,
    data_set_opts=data_set_opts,
    data_loader_opts=data_loader_opts,
    trainer_opts=trainer_opts,
    evaluator_opts=evaluator_opts
)
