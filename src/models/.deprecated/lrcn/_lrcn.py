import os
from typing import Tuple

import torch as th
import torchvision as thv
from torch import nn, optim

import constants as ct

CONV_OUT_C, CONV_OUT_H, CONV_OUT_W = 512, 7, 7
FUSION_IN = CONV_OUT_C * CONV_OUT_H * CONV_OUT_W
FUSION_OUT = LSTM_IN = 4096
LSTM_OUT = CLASSIFIER_IN = 1024


class LRCN(nn.Module):
    num_classes: int
    freeze_features: bool

    features: nn.Sequential
    lstm: nn.Module
    classifier: nn.Module

    def __init__(self, num_classes: int, freeze_features: bool, freeze_fusion: bool):
        super(LRCN, self).__init__()

        self.num_classes = num_classes
        self.freeze_features = freeze_features
        self.freeze_fusion = freeze_fusion

        self.__init_feature_extractor()
        self.__init_fusion()
        self.__init_lstm()
        self.__init_classifier()

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass for the whole LRCN network."""
        bs, sl, c, h, w = _in.shape

        features = self.features(_in.reshape((bs * sl, c, h, w)))
        fusion = self.fusion(features.reshape(bs * sl, FUSION_IN))

        self.lstm.flatten_parameters()

        lstm_out, _ = self.lstm(fusion.reshape(bs, sl, LSTM_IN))
        _out = self.classifier(lstm_out)

        return _out.mean(dim=1), lstm_out.mean(dim=1)

    def __init_feature_extractor(self):
        """Initialize the VGG feature extractor part of the network.
            Use all conv layers. Freeze parameters depending on self.freeze_* properties."""
        self.features = nn.Sequential(*list(thv.models.vgg11_bn(pretrained=False).features.children()))

        if self.freeze_features:
            self.__freeze('features')

    def __init_fusion(self):
        """Initialize the VGG feature fusion part of the network.
            Use all fc layers except the last one. Freeze parameters depending onn self.freeze_* properties."""
        self.fusion = nn.Sequential(*list(thv.models.vgg11_bn(pretrained=False).classifier.children())[:-1])

        if self.freeze_fusion:
            self.__freeze('fusion')

    def __init_lstm(self):
        """Initialize the LSTM layer with Xavier."""
        self.lstm = nn.LSTM(LSTM_IN, LSTM_OUT, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal_(param)

    def __init_classifier(self):
        """Initialize classifier with Xavier."""
        self.classifier = nn.Sequential(
            nn.Linear(CLASSIFIER_IN, self.num_classes),
        )
        for name, param in self.classifier.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal_(param)

    def __freeze(self, module: str) -> None:
        """Freezes the parameters of a module."""
        module: nn.Module = getattr(self, module)
        for child in module.children():
            for param in child.parameters():
                param.requires_grad = False


if __name__ == '__main__':
    from pipeline import smth
    from options import pipe_options

    os.chdir('/Users/Play/Code/AI/master-thesis/src/')
    db_opts = pipe_options.DataBunchOptions(shape='volume', frame_size=224)
    _train_do = pipe_options.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=1.0,
        setting='train',
        transform=None,
        keep=4
    )
    _train_so = pipe_options.SamplingOptions(
        num_segments=4,
        segment_size=4
    )
    _dev_do = pipe_options.DataOptions(
        meta_path=ct.SMTH_META_DEV,
        cut=1.0,
        setting='train',
        transform=None,
        keep=4
    )
    _dev_so = pipe_options.SamplingOptions(
        num_segments=4,
        segment_size=4
    )
    _valid_do = pipe_options.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=1.0,
        setting='valid',
        transform=None,
        keep=2
    )
    _valid_so = pipe_options.SamplingOptions(
        num_segments=4,
        segment_size=4
    )
    dl = pipe_options.DataLoaderOptions(
        batch_size=2,
        pin_memory=False, shuffle=True,
        num_workers=os.cpu_count()
    )
    train_ds = pipe_options.DataSetOptions(_train_do, _train_so)
    dev_ds = pipe_options.DataSetOptions(_train_do, _train_so)
    valid_ds = pipe_options.DataSetOptions(_valid_do, _valid_so)

    bunch = smth.SmthDataBunch(db_opts, train_ds, dev_ds, valid_ds, dl, dl, dl)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    model = LRCN(10, False, False).to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(5):
        for _i, (_x, _y) in enumerate(bunch.train_loader):
            _y_hat = model(_x.to(device=device))
            loss = loss_fn(_y_hat, _y.to(device=device))
            loss.backward()

            grad_mem = 0
            batch_mem = (_x.element_size() * _x.nelement())
            for _param in model.parameters():
                grad_mem += _param.grad.element_size() * _param.grad.nelement()
            print(batch_mem / 1e+6, grad_mem / 1e+6)

            optimizer.step()
            log = f"""[epoch: {epoch}]\
                [prediction: [{th.argmax(_y_hat, dim=1).cpu().numpy()}]\
                [truth: {_y.cpu().numpy()}][loss: {loss.item()}]
                """
            print(log)

            model.zero_grad()
            del loss
