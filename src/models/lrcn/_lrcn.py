import torchvision as thv
from typing import Tuple
from torch import nn
from torch import optim
import torch as th
import os

import pipeline as pipe
import constants as ct

VGG_IN = 224
CONV_OUT_C, CONV_OUT_H, CONV_OUT_W = 512, 7, 7
LSTM_IN = CONV_OUT_C * CONV_OUT_H * CONV_OUT_W
LSTM_OUT = CLASSIFIER_IN = 1024


class LRCN(nn.Module):
    num_classes: int
    freeze_feature_extractor: bool

    features: nn.Sequential
    lstm: nn.Module
    classifier: nn.Module

    def __init__(self, num_classes: int, freeze_feature_extractor: bool):
        super(LRCN, self).__init__()
        self.num_classes = num_classes
        self.freeze_feature_extractor = freeze_feature_extractor
        self.__init_feature_extractor(thv.models.vgg11_bn(pretrained=True))
        self.__init_lstm()
        self.__init_classifier()

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass for the whole LRCN network."""
        features = self._extract_features(_in)
        lstm_out, _ = self.lstm(features)
        _out = self.classifier(lstm_out)

        return _out.mean(dim=1)

    def _extract_features(self, _in: th.Tensor) -> th.Tensor:
        """Forward pass of the VGG feature extractor network for each frame in the input."""
        bs, sl, c, h, w = _in.shape
        _out = self.features(_in.view((bs * sl, c, h, w)))

        return _out.view((bs, sl, CONV_OUT_C * CONV_OUT_H * CONV_OUT_W))

    def __init_feature_extractor(self, vgg: thv.models.VGG):
        """Initialize the VGG feature extractor part of the network.
            Use all conv layers. Freeze parameters depending on self.freeze_* properties."""
        self.features = nn.Sequential(*list(vgg.features.children()))

        if self.freeze_feature_extractor:
            self.__freeze('features')

    def __init_lstm(self):
        """Initialize the LSTM layer with Xavier."""
        self.lstm = nn.LSTM(LSTM_IN, LSTM_OUT, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            else:
                nn.init.xavier_normal_(param)

    def __init_classifier(self):
        """Init classifier with Xavier."""
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


def main():
    db_opts = pipe.DataBunchOptions(shape='volume', frame_size=224)
    _train_do = pipe.DataOptions(
        meta_path=ct.SMTH_META_TRAIN,
        cut=1.0,
        setting='train',
        transform=None,
        keep=None
    )
    _train_so = pipe.SamplingOptions(
        num_segments=4,
        segment_size=4
    )
    _valid_do = pipe.DataOptions(
        meta_path=ct.SMTH_META_VALID,
        cut=1.0,
        setting='valid',
        transform=None,
        keep=None
    )
    _valid_so = pipe.SamplingOptions(
        num_segments=4,
        segment_size=4
    )
    _train_ds_opts = pipe.DataSetOptions(
        do=_train_do,
        so=_train_so
    )
    _valid_ds_opts = pipe.DataSetOptions(
        do=_valid_do,
        so=_valid_so
    )
    train_ds = pipe.DataSetOptions(_train_do, _train_so)
    valid_ds = pipe.DataSetOptions(_valid_do, _valid_so)
    dl = pipe.DataLoaderOptions(batch_size=5, pin_memory=True, shuffle=False, num_workers=os.cpu_count())
    bunch = pipe.SmthDataBunch(db_opts, train_ds, valid_ds, dl, dl)
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    model = LRCN(10, False).to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(5):
        for _i, (_x, _y) in enumerate(bunch.train_loader):
            _y_hat = model(_x.to(device=device))
            loss = loss_fn(_y_hat, _y.to(device=device))
            loss.backward()

            grad_mem = 0
            batch_mem = (_x.element_size() * _x.nelement())
            for param in model.parameters():
                grad_mem += param.grad.element_size() * param.grad.nelement()
            print(batch_mem / 1e+6, grad_mem / 1e+6)

            optimizer.step()
            log = f"""[epoch: {epoch}]\
            [prediction: [{th.argmax(_y_hat, dim=1).cpu().numpy()}]\
            [truth: {_y.cpu().numpy()}][loss: {loss.item()}]
            """
            print(log)

            model.zero_grad()
            del loss


if __name__ == '__main__':
    os.chdir('/Users/Play/Code/AI/master-thesis/src/')
    main()
