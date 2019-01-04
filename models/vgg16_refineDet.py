import torch
import torch.nn.init as init
from torch import nn
import torch.functional as F

import numpy as np

from utils.detection_utils import SSDDecode_np, NMS_np
from models.refinedet_base import RefineDetHead

def vgg(cfg, i=3, batch_norm=False):
    layers = []
    in_channels = i
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
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
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

class VGG16_RefineDet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        channels = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
        self.backbone = nn.ModuleList(vgg(channels))

        self.extras = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )

        self.feature_map_sizes = [
            (40, 40),
            (20, 20),
            (10, 10),
            (5, 5)
        ]
        self.aspect_ratios = [
            [2, 1, 1/2],
            [2, 1, 1/2],
            [2, 1, 1/2],
            [2, 1, 1/2]
        ]
        out_channels = [512, 512, 1024, 512]

        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)
        self.detection_head = RefineDetHead(self.feature_map_sizes, self.aspect_ratios, n_classes, out_channels)


    def forward(self, x):
        for k in range(23):
            x = self.backbone[k](x)
        backbone0 = x
        for k in range(23, 30):
            x = self.backbone[k](x)
        backbone1 = x
        for k in range(30, len(self.backbone)):
            x = self.backbone[k](x)
        backbone2 = x

        extras = self.extras(backbone2)

        feature_maps = [self.L2Norm_4_3(backbone0), self.L2Norm_5_3(backbone1), backbone2, extras]
        objectness, refine_loc, conf, loc, anchors = self.detection_head(feature_maps)

        return objectness, refine_loc, conf, loc, anchors

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

# model = VGG16_RefineDet(20)
# x = torch.zeros((10, 3, 320, 320))
# out = model(x)
