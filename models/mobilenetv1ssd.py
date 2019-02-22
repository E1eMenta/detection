import torch
import torch.nn as nn

from torch.nn import Conv2d, Sequential, ModuleList, ReLU

from utils import weight_init

from .backbones import MobileNetV1
from pydet.head.ssd import SSDHead, AnchorCellCreator, SSDPostprocess


aspect_ratios = [
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3],
    [2, 1/2, 1, 3, 1/3]
]
sizes = [
    (19, 19),
    (10, 10),
    (5, 5),
    (3, 3),
    (2, 2),
    (1, 1),
]

class MobileNetV1SSD(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes
        self.base_net = MobileNetV1(pretrained=True).model
        self.source_layer_indexes = [ 12, 14 ]
        self.extras = ModuleList([
            Sequential(
                Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            )
        ])
        weight_init(self.extras)


        self.head = SSDHead(
            num_classes,
            [512, 1024, 512, 256, 256, 256],
            sizes,
            AnchorCellCreator(aspect_ratios, smin=0.2, smax=0.95)
        )

    def forward(self, x: torch.Tensor):
        start_layer_index = 0
        outs = []
        for end_layer_index in self.source_layer_indexes:

            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            y = x
            outs.append(y)

            start_layer_index = end_layer_index

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            outs.append(x)

        conf, loc, anchors = self.head(outs)

        if self.training:
            return conf, loc, anchors
        else:
            train_out = (conf, loc, anchors)
            return (train_out, SSDPostprocess(train_out))

    def finetune_from(self, path):
        weights = torch.load(path, map_location='cpu')
        own_state = self.state_dict()
        for name, param in weights.items():
            if name not in own_state:
                continue
            if 'head' in name:
                continue
            own_state[name].copy_(param)

        # self.load_state_dict(weights, strict=False)

        weight_init(self.head)

        # self.base_net.load_state_dict(weights.base_net.state_dict())
        # self.extras.load_state_dict(weights.extras.state_dict())