# Based on implementation from https://github.com/hassony2/kinetics_i3d_pytorch
import typing as tp

import torch as th
from torch import distributions, nn


class ReparameterizedSample(nn.Module):
    def __init__(self):
        super(ReparameterizedSample, self).__init__()

        self.flatten = Flatten()
        self.unflatten = Unflatten()

    def forward(self, mean: th.Tensor, var: th.Tensor, num_samples: int) -> th.Tensor:
        std = th.sqrt(var)
        dist = distributions.normal.Normal(mean, std)
        z = dist.rsample([num_samples])

        return z.transpose(0, 1).contiguous()


class Upsample(nn.Module):
    shape: tp.Tuple[int, int, int]

    def __init__(self, shape: tp.Tuple[int, int, int]):
        super(Upsample, self).__init__()

        self.shape = shape

    def forward(self, _in: th.Tensor):
        return nn.functional.interpolate(_in, self.shape)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, _in: th.Tensor):
        bs = _in.shape[0]
        return _in.reshape((bs, -1))


class Unflatten(nn.Module):
    shape: tp.Tuple[int, ...]

    def __init__(self):
        super(Unflatten, self).__init__()

    def forward(self, _in: th.Tensor, num_samples: int, shape: tp.Tuple[int, ...]) -> th.Tensor:
        bs = _in.shape[0]

        return _in.reshape((bs, max(1, num_samples), *shape))


class Standardize(nn.Module):
    def __init__(self, means: tp.List[float], stds: tp.List[float]):
        super(Standardize, self).__init__()
        self.means = nn.Parameter(th.tensor(means, dtype=th.float).reshape((1, 3, 1, 1)), requires_grad=False)
        self.stds = nn.Parameter(th.tensor(stds, dtype=th.float).reshape((1, 3, 1, 1)), requires_grad=False)

    def forward(self, _in: th.Tensor) -> th.Tensor:
        return _in.sub(self.means).div(self.stds)
