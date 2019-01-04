import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from utils.detection_utils import SSDDecode_np, NMS_np

from models.ssd_base import SSDDetectionHead

backbone_dict = {
    18:  resnet18,
    34:  resnet34,
    50:  resnet50,
    101: resnet101,
    152: resnet152,
}
out_channels_dict = {
    18:  [128, 256, 512, 256, 256, 256],
    34:  [128, 256, 512, 256, 256, 256],
    50:  [512, 1024, 2048, 1024, 512, 512],
    101: [512, 1024, 2048, 1024, 512, 512],
    152: [512, 1024, 2048, 1024, 512, 512]
}

class ResnetSSD(nn.Module):
    def __init__(self, n_classes, resnet_idx=34, pretrained=False):
        super().__init__()
        self.n_classes = n_classes
        resnet = backbone_dict[resnet_idx]
        out_channels = out_channels_dict[resnet_idx]
        backbone = resnet(pretrained=pretrained)

        self.conv1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.block1 = backbone.layer1
        self.block2 = backbone.layer2
        self.block3 = backbone.layer3
        self.block4 = backbone.layer4

        self.ext1 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3] // 2, kernel_size=1, stride=1),
            nn.Conv2d(out_channels[3] // 2, out_channels[3], kernel_size=3, stride=2, padding=1)
        )
        self.ext2 = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[4] // 2, kernel_size=1, stride=1),
            nn.Conv2d(out_channels[4] // 2, out_channels[4], kernel_size=3, stride=1)
        )
        self.ext3 = nn.Sequential(
            nn.Conv2d(out_channels[4], out_channels[5] // 2, kernel_size=1, stride=1),
            nn.Conv2d(out_channels[5] // 2, out_channels[5], kernel_size=3, stride=1)
        )

        self.feature_map_sizes = [
            (38, 38),
            (19, 19),
            (10, 10),
            (5, 5),
            (3, 3),
            (1, 1)
        ]
        self.aspect_ratios = [
            [2, 1, 1/2],
            [2, 1/2, 1, 3, 1/3],
            [2, 1/2, 1, 3, 1/3],
            [2, 1/2, 1, 3, 1/3],
            [2,  1, 1/2],
            [2,  1, 1/2]
        ]

        self.detection_head = SSDDetectionHead(self.feature_map_sizes, self.aspect_ratios, n_classes, out_channels)

    def forward(self, x):
        conv1 = self.conv1(x)
        block1 = self.block1(conv1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        ext1 = self.ext1(block4)
        ext2 = self.ext2(ext1)
        ext3 = self.ext3(ext2)

        feature_maps = [block2, block3, block4, ext1, ext2, ext3]

        conf, loc, anchors = self.detection_head(feature_maps)

        return conf, loc, anchors

    def transform_output(self, model_output, conf_thresh=0.5, nms_threshold=0.5):
        conf_batch, loc_batch, anchors = model_output
        anchors = anchors[0]

        softmax = F.softmax(conf_batch, dim=2)
        softmax, loc_batch, anchors = softmax.cpu().numpy(), loc_batch.cpu().numpy(), anchors.cpu().numpy()
        batch_scores = np.max(softmax, axis=2)
        batch_labels = np.argmax(softmax, axis=2)

        batch_selected_boxes, batch_selected_scores, batch_selected_labels = [], [], []
        for loc, scores, labels in zip(loc_batch, batch_scores, batch_labels):
            # Decode bboxes
            boxes, labels, indices = SSDDecode_np(labels, loc, anchors)
            scores = scores[indices]

            # Confidence supression
            boxes = boxes[scores > conf_thresh]
            labels = labels[scores > conf_thresh]
            scores = scores[scores > conf_thresh]

            # NMS
            nms_indices = NMS_np(boxes, labels, iou_threshold=nms_threshold)

            boxes = boxes[nms_indices]
            labels = labels[nms_indices]
            scores = scores[nms_indices]

            batch_selected_boxes.append(boxes)
            batch_selected_scores.append(scores)
            batch_selected_labels.append(labels)

        return batch_selected_boxes, batch_selected_scores, batch_selected_labels