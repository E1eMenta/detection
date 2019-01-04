import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from math import sqrt

from utils.detection_utils import GuaranteedEncode_batch, decode_batch

def createAnchors(sizes, aspect_ratios, scale=4):
    '''
    Create anchor boxes from SSD paper
    :param sizes: Sizes of feature maps. list [N] of (H, W). H - height, W - width.
    :param aspect_ratios: list [N] of anchor box aspect ratios for each feature map
    :return: torch.Tensor [anchors_num, 4]. anchors_num - total number of anchor boxes. Each anchor [cx, cy, w, h]
            cx/cy - center of anchor, h/w - height/width of anchor. All coordinates are relativ.
    '''
    anchors = []
    anchors_per_cell = [len(ar) for ar in aspect_ratios]
    m = len(sizes)
    for k, (H, W) in enumerate(sizes):
        step_y = 1.0 / H
        step_x = 1.0 / W

        size_h = scale * step_y
        size_w = scale * step_x
        for h_i in range(H):
            for w_i in range(W):
                cx = (w_i + 0.5) * step_x
                cy = (h_i + 0.5) * step_y

                for ar in aspect_ratios[k]:
                    h = size_h / sqrt(ar)
                    w = size_w * sqrt(ar)
                    anchors.append(np.array([cx, cy, w, h]))

    anchors = np.stack(anchors).astype(np.float32)

    anchors = torch.from_numpy(anchors)
    return anchors, anchors_per_cell

class TransferBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU()
        )
    def forward(self, feature_map, topDownConnection=None):
        x = self.block1(feature_map)
        if topDownConnection is not None:
            topDown = self.upsample(topDownConnection)
            x += topDown

        x = self.block2(x)
        return x

class ARM(nn.Module):
    def __init__(self, in_channels, anchors_per_cell):
        super().__init__()
        self.in_channels = in_channels
        self.anchors_per_cell = anchors_per_cell

        self.refine_layers = nn.ModuleList(
            [nn.Conv2d(in_channel, num * (2 + 4), 3, padding=1)
             for num, in_channel in zip(anchors_per_cell, in_channels)]
        )

    def forward(self, inputs):
        batch_size = inputs[0].shape[0]
        outs = [ layer(input) for layer, input in zip(self.refine_layers, inputs)]
        outs = [ out.permute((0, 2, 3, 1)).contiguous() for out in outs]
        outs = [ out.view((batch_size, -1, (2 + 4))) for out in outs]

        output = torch.cat(outs, dim=1)

        objectness = output[:, :, :2]
        refine_loc = output[:, :, 2:]

        return objectness, refine_loc

class ODM(nn.Module):
    def __init__(self, in_channels, anchors_per_cell, n_classes):
        super().__init__()
        self.in_channels = in_channels
        self.anchors_per_cell = anchors_per_cell
        self.n_classes = n_classes
        self.c_ = n_classes + 1 # Background class

        self.layers = nn.ModuleList(
            [nn.Conv2d(in_channel, num * (self.c_ + 4), 3, padding=1)
             for num, in_channel in zip(anchors_per_cell, in_channels)]
        )

    def forward(self, inputs):
        batch_size = inputs[0].shape[0]
        outs = [layer(input) for layer, input in zip(self.layers, inputs)]
        outs = [out.permute((0, 2, 3, 1)).contiguous() for out in outs]
        outs = [out.view((batch_size, -1, (self.c_ + 4))) for out in outs]

        output = torch.cat(outs, dim=1)

        conf = output[:, :, :self.c_]
        loc = output[:, :, self.c_:]

        return conf, loc

class RefineDetHead(nn.Module):
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
        self.in_channels       = in_channels


        self.anchors, anchors_per_cell = createAnchors(feature_map_sizes, aspect_ratios)
        self.anchors = self.anchors.unsqueeze(dim=0)

        self.arm = ARM(in_channels, anchors_per_cell)

        self.transfer_blocks = nn.ModuleList([TransferBlock(in_channel) for in_channel in in_channels])
        out_channels = [transfer_block.out_channels for transfer_block in self.transfer_blocks]

        self.odm = ODM(out_channels, anchors_per_cell, n_classes)


    def forward(self, backbone_outs):
        device = backbone_outs[0].device
        outs_num = len(backbone_outs)

        objectness, refine_loc = self.arm(backbone_outs)

        topDownConnection = None
        transfered_outs = [None] * outs_num
        for i in range(outs_num,0,-1):
            i -= 1

            transfer_block = self.transfer_blocks[i]
            out = backbone_outs[i]

            if i == outs_num - 1:
                transfered_out = transfer_block(out)
                topDownConnection = transfered_out
                transfered_outs[i] = transfered_out
            else:
                transfered_out = transfer_block(out, topDownConnection)
                topDownConnection = transfered_out
                transfered_outs[i] = transfered_out

        conf, loc = self.odm(transfered_outs)

        anchors = self.anchors.to(device)

        return objectness, refine_loc, conf, loc, anchors

class RefineDetLoss(nn.Module):
    def __init__(self, match_thresh=0.5, neg_pos=3,  variance=(0.1, 0.2), theta=0.99):
        super().__init__()
        self.match_thresh = match_thresh
        self.neg_pos = neg_pos
        self.variance = variance
        self.theta = theta

    def forward(self, model_output, targets):
        objectness, refine_loc, pred_conf, pred_loc, anchors = model_output
        # pred_conf, pred_loc, anchors = model_output
        batch_size, anchors_num, n_classes = pred_conf.shape

        target_conf, target_loc = GuaranteedEncode_batch(targets, anchors[0], threshold=self.match_thresh)

        positive_position = target_conf > 0
        arm_N = torch.sum(positive_position).float()

        # ARM Localization Loss
        arm_localization_loss = F.smooth_l1_loss(
            refine_loc[positive_position],
            target_loc[positive_position],
            reduction='sum'
        )
        arm_class_loss = F.cross_entropy(objectness.view(-1, 2), positive_position.view(-1).long(), reduction='sum')
        object_mask = F.softmax(objectness, dim=-1)[:, :, 0] < self.theta

        refined_anchors = decode_batch(refine_loc, anchors[0], self.variance)
        # target_conf, target_loc = GuaranteedEncode_batch(targets, refined_anchors, threshold=self.match_thresh)

        N = 0
        localization_loss = 0
        class_loss = 0
        for batch_i in range(batch_size):
            target_conf_i, target_loc_i = GuaranteedEncode_batch(
                targets[batch_i : batch_i + 1],
                refined_anchors[batch_i],
                threshold=self.match_thresh
            )
            object_mask_i = object_mask[batch_i]
            target_conf_i = target_conf_i[0][object_mask_i]
            target_loc_i = target_loc_i[0][object_mask_i]
            pred_conf_i = pred_conf[batch_i][object_mask_i]
            pred_loc_i = pred_loc[batch_i][object_mask_i]

            positive_position_i = target_conf_i > 0
            N_i = torch.sum(positive_position_i).float()
            N += N_i

            # Localization Loss (Smooth L1)
            localization_loss += F.smooth_l1_loss(
                pred_loc_i[positive_position_i],
                target_loc_i[positive_position_i].detach(),
                reduction='sum'
            )

            # Classification loss
            positive_num = torch.sum(target_conf_i > 0)
            negative_num = max(10, min(positive_num * self.neg_pos, anchors_num - positive_num))
            # print(pred_conf_i.shape, target_conf_i.shape)
            class_loss_i = F.cross_entropy(pred_conf_i, target_conf_i, reduction='none')

            class_loss_positives = class_loss_i[target_conf_i > 0]
            class_loss_negatives = class_loss_i[target_conf_i == 0]

            _, loss_idx = class_loss_negatives.sort(0, descending=True)
            class_loss_negatives = class_loss_negatives[loss_idx[:negative_num]]

            class_loss += torch.sum(class_loss_positives) + torch.sum(class_loss_negatives)

        class_loss = class_loss / N
        localization_loss = localization_loss / N
        arm_class_loss = 0.04 * arm_class_loss / arm_N
        arm_localization_loss = arm_localization_loss / arm_N

        total_loss = class_loss + localization_loss + arm_class_loss + arm_localization_loss

        return total_loss, class_loss, localization_loss, arm_class_loss, arm_localization_loss