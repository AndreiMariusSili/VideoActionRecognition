from torch import nn

from models.tarn import _helpers as hp


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = hp.conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = hp.conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 2)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 1, stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * ResidualBlock.expansion:
            downsample = nn.Sequential(
                hp.conv1x1(self.in_planes, planes * ResidualBlock.expansion, stride),
                nn.BatchNorm2d(planes * ResidualBlock.expansion),
            )
        layers = [
            ResidualBlock(self.in_planes, planes, stride, downsample)
        ]
        self.in_planes = planes * ResidualBlock.expansion
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_planes, planes, 1))

        return nn.Sequential(*layers)

    def forward(self, _in):
        _in = self.conv1(_in)
        _in = self.bn1(_in)
        _in = self.relu(_in)
        _in = self.max_pool(_in)

        _in = self.layer1(_in)
        _in = self.layer2(_in)
        _in = self.layer3(_in)
        _in = self.layer4(_in)

        return _in


if __name__ == '__main__':
    import models.helpers
    import torch as th

    _batch_size = 2
    _num_class = 10
    _growth_rate = 64
    _input_var = th.randn(_batch_size, 3, 224, 224)
    model = ResNet()
    feature_maps = model(_input_var)
    print(feature_maps.shape, models.helpers.count_parameters(model))
