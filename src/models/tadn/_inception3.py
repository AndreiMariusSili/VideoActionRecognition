import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision as thv
import torch as th

MODEL_URL = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'


class InceptionBase(thv.models.Inception3):

    Bottleneck: th.nn.Conv2d

    def __init__(self):
        super(InceptionBase, self).__init__(1000, True, True)
        self.load_state_dict(model_zoo.load_url(MODEL_URL))

        del self.AuxLogits
        del self.fc

        self.Bottleneck = th.nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

    def forward(self, x):
        if self.transform_input:
            x_ch0: th.Tensor = th.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1: th.Tensor = th.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2: th.Tensor = th.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = th.cat((x_ch0, x_ch1, x_ch2), 1)
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = self.Bottleneck(x)
        # N x 256 x 8 x 8
        return x
