# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import torch as th
from torch import nn

import models.helpers as hp
from models.i3d.common import classifier as cls, encoder as en


class I3D(nn.Module):
    num_classes: int
    dropout_prob: float
    name: str

    def __init__(self, num_classes: int, dropout_prob: float = 0.0, name: str = 'inception'):
        super(I3D, self).__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.name = name

        self.encoder = en.I3DEncoder(None)
        self.classifier = cls.I3DClassifier(1024, dropout_prob, num_classes)

        hp.he_init(self)

    def forward(self, _in: th.Tensor) -> th.tensor:
        _out = self.encoder(_in)
        _out, _embed = self.classifier(_out)

        return _out, _embed


if __name__ == '__main__':
    import models.helpers

    i3d = I3D(30, 0.5)
    __in = th.randn((1, 4, 3, 224, 224), dtype=th.float)
    __preds, __embeds = i3d(__in)

    print(i3d)
    print(__preds.shape, __embeds.shape)
    print(f'{models.helpers.count_parameters(i3d):,}')
