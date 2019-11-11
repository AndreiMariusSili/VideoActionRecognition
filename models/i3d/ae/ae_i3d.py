import typing as tp

import torch as th
from torch import nn

import models.helpers as hp
from models.i3d.common import classifier as cls, decoder as de, encoder as en

AE_FORWARD = tp.Tuple[th.Tensor, th.Tensor, tp.Optional[th.Tensor]]


class AEI3D(nn.Module):
    num_classes: int
    dropout_prob: float
    embed_planes: int

    def __init__(self, embed_planes: int, dropout_prob: float, num_classes: int):
        super(AEI3D, self).__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embed_planes = embed_planes

        self.encoder = en.I3DEncoder(self.embed_planes)
        self.decoder = de.I3DDecoder(self.embed_planes)
        self.classifier = cls.I3DClassifier(self.embed_planes, self.dropout_prob, self.num_classes)

        hp.he_init(self)

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
    ae = AEI3D(1024, 0.5, 30)
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
    print(f'{models.helpers.count_parameters(ae.encoder):,}')
    print(f'{models.helpers.count_parameters(ae.decoder):,}')
    print(f'{models.helpers.count_parameters(ae.classifier):,}')
