from typing import Optional, Tuple

import torch as th
from torch import nn

from models.ae_i3d._classifier import I3DClassifier
from models.ae_i3d._decoder import I3DDecoder
from models.ae_i3d._encoder import I3DEncoder

AE_FORWARD = Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]


class AEI3D(nn.Module):
    num_classes: int
    dropout_prob: float
    embed_planes: int

    def __init__(self, embed_planes: int, dropout_prob: float, num_classes: int):
        super(AEI3D, self).__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embed_planes = embed_planes

        self.encoder = I3DEncoder(self.embed_planes)
        self.decoder = I3DDecoder(self.embed_planes)
        self.classifier = I3DClassifier(self.embed_planes, self.dropout_prob, self.num_classes)

        self._he_init()

    def _he_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                th.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, _in: th.Tensor, inference: bool) -> AE_FORWARD:
        if inference:
            return self._inference(_in)
        else:
            return self._forward(_in)

    def _forward(self, _in: th.Tensor) -> AE_FORWARD:
        _embed = self.encoder(_in)
        _recon = self.decoder(_embed)
        _pred, _embed = self.classifier(_embed)

        return _pred, _embed, _recon

    def _inference(self, _in: th.Tensor) -> AE_FORWARD:
        _embed = self.encoder(_in)
        _recon = self.decoder(_embed)
        _pred, _embed = self.classifier(_embed)

        return _pred, _embed, _recon


if __name__ == '__main__':
    import os
    import models.helpers

    os.chdir('/Users/Play/Code/AI/master-thesis/src')
    ae = AEI3D(1024, 0.0, 10)
    print(ae)

    print("===INPUT===")
    _in = th.randn((2, 4, 3, 224, 224), dtype=th.float)
    print(_in.shape)

    print("===FORWARD===")
    y, z, x = ae(_in, False)
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"recon":20s}:\t{x.shape}')

    print("===INFERENCE===")
    y, z, _ = ae(_in, True)
    print(f'{"latent":20s}:\t{z.shape}')
    print(f'{"pred":20s}:\t{y.shape}')
    print(f'{"recon":20s}:\t{_.shape}')

    print(f'{models.helpers.count_parameters(ae):,}')
