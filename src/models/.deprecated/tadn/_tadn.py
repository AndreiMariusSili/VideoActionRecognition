from typing import Tuple

import torch as th
from models.tadn._inception3 import InceptionBase
from torch import nn

INC_OUT_C, INC_OUT_H, INC_OUT_W = 2048, 5, 5


class DenseLayer(nn.Module):
    step: int
    growth_rate: int
    n_input_plane: int
    drop_rate: int

    _sequential: nn.Sequential

    def __init__(self, n_channels: int, step: int, growth_rate: int, n_input_plane: int, drop_rate: float):
        super(DenseLayer, self).__init__()

        self.step = step
        self.n_input_plane = n_input_plane
        self.growth_rate = growth_rate

        self._sequential = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Dropout(p=drop_rate)
        )

    def _cat(self, _in: th.Tensor, _prev: th.Tensor) -> th.Tensor:
        _out = _in
        if _prev is not None:
            _out = th.cat((_in, _prev), dim=1)  # cat across channels
        return _out

    def forward(self, _in: th.Tensor, _prev) -> th.Tensor:
        _out = self._cat(_in, _prev)
        _out = self._sequential(_out)

        if self.step > 0:
            # concatenate outputs of other time-steps
            _out = th.cat((_prev, _out), dim=1)
        return _out


class TimeAlignedDenseNet(nn.Module):
    spatial: nn.Module
    temporal: nn.ModuleList
    aggregation: nn.Sequential
    classifier: nn.Sequential

    n_input_plane: int
    n_output_plane: int
    time_steps: int
    growth_rate: int
    drop_rate: float
    num_classes: int

    def __init__(self, time_steps: int, growth_rate: int, drop_rate: float, num_classes: int):
        super(TimeAlignedDenseNet, self).__init__()

        self.n_input_plane = INC_OUT_C
        self.n_output_plane = time_steps * growth_rate
        self.time_steps = time_steps
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        self.spatial = self._init_spatial()
        self.temporal = self._init_temporal()
        self.aggregation = self._init_aggregation()
        self.classifier = self._init_classifier()

        self._he_init()

    def _init_spatial(self) -> nn.Sequential:
        return InceptionBase()

    def _init_temporal(self) -> nn.ModuleList:
        temporal_features = nn.ModuleList()
        n_channels = self.n_input_plane
        for step in range(self.time_steps):
            layer = DenseLayer(n_channels, step, self.growth_rate, self.n_input_plane, self.drop_rate)
            n_channels = self.n_input_plane + (step + 1) * self.growth_rate

            temporal_features.append(layer)

        return temporal_features

    def _init_classifier(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(1024, self.num_classes, kernel_size=1, bias=False),
        )

    def _init_aggregation(self) -> nn.Sequential:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_output_plane, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024, 1e-3),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024, 1e-3),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _he_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                th.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, _in: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        bs, t, c, h, w = _in.shape
        _in = self.spatial(_in.reshape((bs * t, c, h, w))).reshape((bs, t, INC_OUT_C, INC_OUT_H, INC_OUT_W))

        _in_prev = None
        _out = None
        for t in range(self.time_steps):
            _in_t = _in[:, t, :, :, :]
            _out = self.temporal[t](_in_t, _in_prev)
            _in_prev = _out
        _embeds = self.aggregation(_out)
        _out = self.classifier(_embeds)

        return _out.reshape(bs, self.num_classes), _embeds.reshape(bs, -1)


if __name__ == "__main__":
    _batch_size = 2
    _time_steps = 4
    _num_class = 10
    _growth_rate = 64
    _input_var = th.randn(_batch_size, _time_steps, 3, 224, 224)
    model = TimeAlignedDenseNet(_time_steps, _growth_rate, 0, _num_class)
    output, embeds = model(_input_var)
    print(output.shape, embeds.shape)
