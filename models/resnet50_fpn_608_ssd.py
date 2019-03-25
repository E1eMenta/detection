import torch
import torch.nn as nn

from torch.nn import Conv2d, Sequential, ReLU

from torchvision.models import resnet50

from pydet.head.ssd import SSDHead, AnchorCellCreator, SSDPostprocess
from pydet.nn.fpn import FPN

from utils import weight_init


aspect_ratios = [
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1],
    [2, 1/2, 1]
]
image_size = (608, 608)
sizes = [
    (76, 76),
    (38, 38),
    (19, 19),
    (10, 10),
    (5, 5),
    (3, 3)
]
class Resnet50_FPN_SSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        base_net = resnet50(pretrained=True)

        self.conv1 = base_net.conv1
        self.bn1 = base_net.bn1
        self.relu = base_net.relu
        self.maxpool = base_net.maxpool
        self.layer1 = base_net.layer1
        self.layer2 = base_net.layer2
        self.layer3 = base_net.layer3
        self.layer4 = base_net.layer4

        self.extras0 = Sequential(
            Conv2d(in_channels=2048, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
        self.extras1 = Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        )
        self.extras2 = Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        )

        out_channels = 256
        self.fpn = FPN([512, 1024, 2048, 512, 256, 256], out_channels)

        out_channels_list = [out_channels] * 6
        self.head = SSDHead(
            num_classes,
            out_channels_list,
            sizes,
            AnchorCellCreator(aspect_ratios, smin=0.1, smax=0.95)
        )


    def forward(self, x):
        _, _, h, w = x.shape
        if h != image_size[0] or w != image_size[1]:
            raise Exception(f"Image should have size ({image_size[0]}, {image_size[1]}) instead of ({h}, {w})."
                            f"Change 'sizes' and 'image_size' vars to fix it")
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        x = self.extras0(x)
        outs.append(x)
        x = self.extras1(x)
        outs.append(x)
        x = self.extras2(x)
        outs.append(x)

        outs = self.fpn(outs)
        conf, loc, anchors = self.head(outs)

        if self.training:
            return conf, loc, anchors
        else:
            train_out = (conf, loc, anchors)
            return (train_out, SSDPostprocess(train_out))

    def finetune_from(self, path):
        weight_init(self)
        weights = torch.load(path, map_location='cpu')
        load_state = weights.state_dict()
        own_state = self.state_dict()
        for name, param in load_state.items():
            if name not in own_state:
                continue
            if 'head' in name:
                continue
            own_state[name].copy_(param)

# model = Resnet50_FPN_SSD(2)
# x = torch.zeros((1, 3, 300, 300))
#
# o = model(x)