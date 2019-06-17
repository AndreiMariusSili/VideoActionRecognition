from typing import Optional, Tuple

import torch as th
from torch import nn

from models.tarn import _helpers as hp, _resnet as rn

SPATIAL_OUT_C, SPATIAL_OUT_H, SPATIAL_OUT_W = 256, 7, 7
th.autograd.set_detect_anomaly(True)


class TemporalResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(TemporalResidualBlock, self).__init__()

        self.conv1 = hp.conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()
        self.conv2 = hp.conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride

    def forward(self, _in: th.Tensor, _prev: Optional[th.Tensor] = None) -> th.Tensor:
        _out = _in
        if _prev is not None:
            _out = _in.add(_prev)

        _out = self.conv1(_out)
        _out = self.bn1(_out)
        _out = self.relu(_out)

        _out = self.conv2(_out)
        _out = self.bn2(_out)

        _out = _out.add(_in)
        _out = self.relu(_out)

        return _out


class TimeAlignedResNet(nn.Module):
    spatial: nn.Module
    temporal: nn.ModuleList
    aggregation: nn.Sequential
    classifier: nn.Sequential

    in_planes: int
    out_planes: int
    time_steps: int
    drop_rate: float
    num_classes: int

    def __init__(self, time_steps: int, drop_rate: float, num_classes: int):
        super(TimeAlignedResNet, self).__init__()

        self.in_planes = SPATIAL_OUT_C
        self.out_planes = SPATIAL_OUT_C

        self.time_steps = time_steps
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        self.spatial = self._init_spatial()
        self.temporal = self._init_temporal()
        self.aggregation = self._init_aggregation()
        self.classifier = self._init_classifier()

        self._he_init()

    def _init_spatial(self) -> nn.Sequential:
        return rn.ResNet()

    def _init_temporal(self) -> nn.ModuleList:
        temporal = nn.ModuleList()
        for step in range(self.time_steps):
            layer = TemporalResidualBlock(self.in_planes, self.out_planes)
            temporal.append(layer)

        return temporal

    def _init_aggregation(self) -> nn.Sequential:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_planes),
            nn.Dropout(0.5),
            nn.Conv2d(self.out_planes, self.out_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_planes),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def _init_classifier(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(self.out_planes, self.num_classes, kernel_size=1, bias=True),
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
        b, t, c, h, w = _in.shape
        _in = self.spatial(_in.view((b * t, c, h, w))).view((b, t, SPATIAL_OUT_C, SPATIAL_OUT_H, SPATIAL_OUT_W))
        _out = None
        for t in range(self.time_steps):
            _out_prev = _out
            _in_t = _in[:, t, :, :, :]
            _out = self.temporal[t](_in_t, _out_prev)
        _embeds = self.aggregation(_out)
        _out = self.classifier(_embeds)
        return _out.view(b, self.num_classes), _embeds.view(b, -1)


if __name__ == "__main__":
    _batch_size = 2
    _time_steps = 4
    _num_class = 10
    _input_var = th.randn(_batch_size, _time_steps, 3, 224, 224)
    model = TimeAlignedResNet(_time_steps, 0.0, _num_class)
    output, embeds = model(_input_var)
