import os
import urllib

import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1024, pretrained=False):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.fc = nn.Linear(1024, num_classes)

        if pretrained:
            # Input fromat: RGB images have to be loaded in to a range of [-1, 1]
            save_path = os.path.join("tmp", "mobilenet_v1_with_relu_69_5.pth")
            if not os.path.exists(save_path):
                os.makedirs("tmp", exist_ok=True)
                url = "https://storage.googleapis.com/models-hao/mobilenet_v1_with_relu_69_5.pth"
                urllib.request.urlretrieve(url, save_path)

            self.model.load_state_dict(torch.load(save_path, map_location=lambda storage, loc: storage))

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
