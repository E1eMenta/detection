import torch
from torch import nn
import torch.nn.functional as F

from math import sqrt
import numpy as np

from utils.detection_utils import SSDEncode_batch_np

def s_(k, smin, smax, m):
    '''
    Area of box at k-th feature map
    Formula 4 from SSD paper
    :param k: Index of feature map [0, m)
    :param smin: Minimum area
    :param smax: Maximum area
    :param m: Number of feature maps
    :return: Area of box at k-th feature map
    '''
    s_k = smin + (smax - smin) / (m - 1) * k
    return s_k
def createAnchors(sizes, aspect_ratios, smin=0.2, smax=0.9):
    '''
    Create anchor boxes from SSD paper
    :param sizes: Sizes of feature maps. list [N] of (H, W). H - height, W - width.
    :param aspect_ratios: list [N] of anchor box aspect ratios for each feature map
    :param smin: size of boxes from first feature map.
    :param smax: size of boxes from last feature map.
    :return: torch.Tensor [anchors_num, 4]. anchors_num - total number of anchor boxes. Each anchor [cx, cy, w, h]
            cx/cy - center of anchor, h/w - height/width of anchor. All coordinates are relativ.
    '''
    anchors = []
    m = len(sizes)
    for k, (H, W) in enumerate(sizes):
        step_y = 1.0 / H
        step_x = 1.0 / W

        s_k = s_(k, smin, smax, m)
        s_prime = sqrt(s_k * s_(k + 1, smin, smax, m))
        for h_i in range(H):
            for w_i in range(W):
                cx = (w_i + 0.5) * step_x
                cy = (h_i + 0.5) * step_y

                for ar in aspect_ratios[k]:
                    w = s_k * sqrt(ar)
                    h = s_k / sqrt(ar)
                    anchors.append(np.array([cx, cy, w, h]))
                # Extra box
                w = s_prime * sqrt(1.0)
                h = s_prime / sqrt(1.0)
                anchors.append(np.array([cx, cy, w, h]))

    anchors = np.stack(anchors).astype(np.float32)

    anchors = torch.from_numpy(anchors)
    return anchors

class SSDDetectionHead(nn.Module):
    def __init__(self,
                 feature_map_sizes,
                 aspect_ratios,
                 n_classes,
                 in_channels
                 ):
        super().__init__()
        self.feature_map_sizes = feature_map_sizes
        self.aspect_ratios     = aspect_ratios
        self.n_classes         = n_classes
        self.c_                = n_classes + 1 # Background class
        self.in_channels       = in_channels

        self.channels_per_anchor = 4 + self.c_

        self.anchors = createAnchors(feature_map_sizes, aspect_ratios)
        self.anchors = self.anchors.unsqueeze(dim=0)

        cell_anchors_num = [len(ar) + 1 for ar in self.aspect_ratios]

        self.detection_layers = []
        for in_channel, anchors_num in zip(self.in_channels, cell_anchors_num):
            layer = nn.Conv2d(in_channel,  anchors_num * self.channels_per_anchor, 3, padding=1)
            self.detection_layers.append(layer)

        self.detection_layers = nn.ModuleList(self.detection_layers)


    def forward(self, backbone_outs):
        device = backbone_outs[0].device
        batch_size = backbone_outs[0].shape[0]

        outs = []
        for backbone_out, layer in zip(backbone_outs, self.detection_layers):
            out = layer(backbone_out).permute((0, 2, 3, 1)).contiguous()
            out = out.view((batch_size, -1, self.channels_per_anchor))
            outs.append(out)

        output = torch.cat(outs, dim=1)

        conf = output[:, :, :self.c_]
        loc = output[:, :, self.c_:]
        self.anchors = self.anchors.to(device)

        return conf, loc, self.anchors

class SSDLoss(nn.Module):
    def __init__(self, match_thresh=0.5, neg_pos=3,  variance=(0.1, 0.2)):
        super().__init__()
        self.match_thresh = match_thresh
        self.neg_pos = neg_pos
        self.variance = variance

    def forward(self, model_output, targets):
        pred_conf, pred_loc, anchors = model_output
        batch_size, anchors_num, n_classes = pred_conf.shape

        anchors = anchors[0].cpu().numpy()
        targets = [(target[0].cpu().numpy(), target[1].cpu().numpy())for target in targets]

        target_conf, target_loc = SSDEncode_batch_np(targets, anchors, threshold=self.match_thresh)

        # from utils.vis import draw_boxes, draw_anchors
        # import cv2
        # for idx in range(batch_size):
        #     image = np.zeros((300, 300, 3))
        #     target_bboxes, target_labels = targets[idx][0], targets[idx][1]
        #     image = draw_boxes(image, target_bboxes, color=(0, 255, 0))
        #     image = draw_anchors(image, anchors, target_conf[idx], loc=target_loc[idx], color=(255, 0, 255))
        #     cv2.imshow("image", image)
        #     cv2.waitKey()

        target_conf = torch.from_numpy(target_conf).to(pred_conf.device).long()
        target_loc = torch.from_numpy(target_loc).to(pred_conf.device)

        positive_position = target_conf > 0
        N = torch.sum(positive_position).float()

        # Localization Loss (Smooth L1)
        loc_p = pred_loc[positive_position]
        loc_t = target_loc[positive_position]
        localization_loss = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Classification loss
        class_loss = 0
        for batch_i in range(batch_size):
            target_conf_i, target_loc_i = target_conf[batch_i], target_loc[batch_i]
            pred_conf_i, pred_loc_i = pred_conf[batch_i], pred_loc[batch_i]

            positive_num = torch.sum(target_conf_i > 0)
            negative_num = max(10, min(positive_num * self.neg_pos, anchors_num - positive_num))
            class_loss_i = F.cross_entropy(pred_conf_i, target_conf_i, reduction='none')

            class_loss_positives = class_loss_i[target_conf_i > 0]
            class_loss_negatives = class_loss_i[target_conf_i == 0]

            _, loss_idx = class_loss_negatives.sort(0, descending=True)
            class_loss_negatives = class_loss_negatives[loss_idx[:negative_num]]

            class_loss += torch.sum(class_loss_positives) + torch.sum(class_loss_negatives)

        class_loss = class_loss / N
        localization_loss = localization_loss / N
        total_loss = class_loss + localization_loss

        return total_loss, class_loss, localization_loss