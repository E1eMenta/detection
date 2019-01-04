import torch
import torch.nn.init as init
from torch import nn
import torch.nn.functional as F

import numpy as np

from utils.detection_utils import SSDDecode_np, NMS_np
from models.ssd_base import SSDDetectionHead

def vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class VGG16_SSD(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        channels = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        self.backbone = nn.ModuleList(vgg(channels))

        self.extras0 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.extras1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.extras2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
        )
        self.extras3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3), nn.ReLU(),
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
        out_channels = [512, 1024, 512, 256, 256, 256]

        self.L2Norm = L2Norm(512, 20)
        self.detection_head = SSDDetectionHead(self.feature_map_sizes, self.aspect_ratios, n_classes, out_channels)


    def forward(self, x):
        for k in range(23):
            x = self.backbone[k](x)
        backbone0 = x
        for k in range(23, len(self.backbone)):
            x = self.backbone[k](x)
        backbone1 = x

        extras0 = self.extras0(backbone1)
        extras1 = self.extras1(extras0)
        extras2 = self.extras2(extras1)
        extras3 = self.extras3(extras2)

        feature_maps = [self.L2Norm(backbone0), backbone1, extras0, extras1, extras2, extras3]

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

class VGG16_BN_SSD(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )

        self.backbone1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), nn.BatchNorm2d(1024), nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1), nn.BatchNorm2d(1024), nn.ReLU()
        )
        self.extras0 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.extras1 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.extras2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.extras3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(),
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
            [2, 1, 1 / 2],
            [2, 1 / 2, 1, 3, 1 / 3],
            [2, 1 / 2, 1, 3, 1 / 3],
            [2, 1 / 2, 1, 3, 1 / 3],
            [2, 1, 1 / 2],
            [2, 1, 1 / 2]
        ]
        out_channels = [512, 1024, 512, 256, 256, 256]

        self.detection_head = SSDDetectionHead(self.feature_map_sizes, self.aspect_ratios, n_classes, out_channels)

    def forward(self, x):
        backbone0 = self.backbone0(x)
        backbone1 = self.backbone1(backbone0)

        extras0 = self.extras0(backbone1)
        extras1 = self.extras1(extras0)
        extras2 = self.extras2(extras1)
        extras3 = self.extras3(extras2)

        feature_maps = [self.L2Norm(backbone0), backbone1, extras0, extras1, extras2, extras3]

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