import torchvision as thv
from typing import Tuple
from tqdm import tqdm
from torch import nn
import torch as th
import os

import pipeline as pipe

VGG_IN = 224
CONV_OUT_C, CONV_OUT_H, CONV_OUT_W = 512, 7, 7
FC_IN = CONV_OUT_C * CONV_OUT_H * CONV_OUT_W
VGG_OUT = LSTM_IN = 4096
LSTM_OUT = CLASSIFIER_IN = 512


class VideoLSTM(nn.Module):
    num_classes: int
    freeze_conv: bool
    freeze_fc: bool

    conv: nn.Sequential
    fc: nn.Sequential
    lstm: nn.Module
    classifier: nn.Module

    def __init__(self, num_classes: int, freeze_conv: bool, freeze_fc: bool):
        super(VideoLSTM, self).__init__()
        self.num_classes = num_classes
        self.freeze_conv = freeze_conv
        self.freeze_fc = freeze_fc
        self.__init_vgg(thv.models.vgg16_bn(pretrained=True))
        self.__init_lstm()
        self.__init_classifier()

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Forward pass for the whole videoLSTM network."""
        bs, sl, c, h, w = _in.size()
        vgg_out = self._vgg(_in)
        _, (lstm_out, _) = self.lstm(vgg_out)
        _out = self.classifier(lstm_out.view(bs, LSTM_OUT))

        return _out

    def _vgg(self, _in: th.Tensor):
        """Forward pass of the VGG network for each frame in the input."""
        bs, sl, c, h, w = _in.size()
        vgg_out = th.empty((bs, sl, VGG_OUT), dtype=th.float32)
        for i, x in enumerate(_in.split(1, dim=1)):
            x = x.view(bs, c, h, w)
            conv_out = self.conv(x)
            fc_out = self.fc(conv_out.view(bs, FC_IN))
            vgg_out[:, i, :] = fc_out

        return vgg_out

    def __init_vgg(self, vgg16: thv.models.VGG):
        """Initialize the VGG part of the network.
            Use all conv layers and all except last fc layers.
            Freeze parameters depending on self.freeze* properties.
        """
        self.conv = nn.Sequential(*list(vgg16.features.children()))
        self.fc = nn.Sequential(*list(vgg16.classifier.children())[:-1])

        if self.freeze_conv:
            self.__freeze('conv')
        if self.freeze_fc:
            self.__freeze('fc')

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
        self.classifier = nn.Linear(CLASSIFIER_IN, self.num_classes)
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
    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    dl_args = dict(batch_size=2, pin_memory=True, shuffle=False, num_workers=0)
    bunch = pipe.SmthDataBunch(0.3, 'volume', 80, False, dl_args)
    model = VideoLSTM(10, True, False)
    loss_fn = nn.CrossEntropyLoss()
    for _i, (_x, _y) in tqdm(enumerate(bunch.train_loader), total=10):
        _y_hat, _vgg_out = model(_x)
        loss = loss_fn(_y_hat, _y)
        loss.backward()
        break
