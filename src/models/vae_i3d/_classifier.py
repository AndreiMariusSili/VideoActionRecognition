# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import torch as th
from torch import nn

from models import common as cm
from options import model_options as mo


class I3DClassifier(nn.Module):
    num_classes: int
    latent_size: int

    def __init__(self, latent_size: int, num_classes: int):
        super(I3DClassifier, self).__init__()

        self.latent_size = latent_size
        self.num_classes = num_classes
        opts = mo.Unit3DOptions(in_channels=latent_size, out_channels=self.num_classes, kernel_size=(1, 1, 1),
                                stride=(1, 1, 1), activation='none', use_bias=False, use_bn=False, padding='VALID')
        self.classifier = cm.Unit3D(opts)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        return self.classifier(_in)
