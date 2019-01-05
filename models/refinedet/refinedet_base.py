import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from math import sqrt

from utils.detection_utils import GuaranteeEncode_batch, decode_batch, SSDDecode_np, NMS_np
from utils.vis import draw_boxes


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


class Detector(nn.Module):
    def __init__(
            self,
            in_channels,
            anchors_per_cell,
            n_classes
    ):
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

        self.arm = Detector(in_channels, anchors_per_cell, 1)

        self.transfer_blocks = nn.ModuleList([TransferBlock(in_channel) for in_channel in in_channels])
        out_channels = [transfer_block.out_channels for transfer_block in self.transfer_blocks]

        self.odm = Detector(out_channels, anchors_per_cell, n_classes)


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


class SSDGuaranteeLoss(nn.Module):
    def __init__(self, match_thresh=0.5, neg_pos=3,  variance=(0.1, 0.2)):
        super().__init__()
        self.match_thresh = match_thresh
        self.neg_pos = neg_pos
        self.variance = variance

    def forward(self, model_output, targets):
        pred_conf, pred_loc, anchors = model_output
        batch_size, anchors_num, n_classes = pred_conf.shape

        target_conf, target_loc = GuaranteeEncode_batch(targets, anchors[0], threshold=self.match_thresh)

        # from utils.vis import draw_boxes, draw_anchors
        # import cv2
        # for idx in range(batch_size):
        #     image = np.zeros((300, 300, 3))
        #     target_bboxes, target_labels = targets[idx][0], targets[idx][1]
        #     image = draw_boxes(image, target_bboxes, color=(0, 255, 0))
        #     image = draw_anchors(image, anchors, target_conf[idx], loc=target_loc[idx], color=(255, 0, 255))
        #     cv2.imshow("image", image)
        #     cv2.waitKey()

        positive_position = target_conf > 0
        N = torch.sum(positive_position).float()

        # Localization Loss (Smooth L1)
        loc_p = pred_loc[positive_position]
        loc_t = target_loc[positive_position]
        loc_t = loc_t.detach()
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

        return class_loss, localization_loss, N

class RefineDetLoss(nn.Module):
    def __init__(self, match_thresh=0.5, neg_pos=3,  variance=(0.1, 0.2), theta=0.99):
        super().__init__()

        self.loss = SSDGuaranteeLoss(match_thresh=match_thresh, neg_pos=neg_pos,  variance=variance)


        self.match_thresh = match_thresh
        self.neg_pos = neg_pos
        self.variance = variance
        self.theta = theta

    def forward(self, model_output, targets):
        objectness, refine_loc, pred_conf, pred_loc, anchors = model_output
        batch_size, anchors_num, n_classes = pred_conf.shape

        arm_class_loss, arm_localization_loss, _ = self.loss((objectness, refine_loc, anchors[0]), targets)


        object_mask = F.softmax(objectness, dim=-1)[:, :, 0] < self.theta
        refined_anchors = decode_batch(refine_loc, anchors[0], self.variance)

        N = 0
        odm_localization_loss = 0
        odm_class_loss = 0
        for batch_i in range(batch_size):
            object_mask_i = object_mask[batch_i]
            pred_conf_i = pred_conf[batch_i:batch_i+1][object_mask_i]
            pred_loc_i = pred_loc[batch_i:batch_i+1][object_mask_i]
            refined_anchors_i = refined_anchors[batch_i][object_mask_i]

            odm_class_loss_i, odm_localization_loss_i, N_i = \
                self.loss((pred_conf_i, pred_loc_i, refined_anchors_i), targets[batch_i : batch_i + 1])

            N += N_i
            odm_class_loss += N_i * odm_class_loss_i
            odm_localization_loss += N_i * odm_localization_loss_i

        odm_class_loss = odm_class_loss / N
        odm_localization_loss = odm_localization_loss / N

        total_loss = arm_class_loss + arm_localization_loss + odm_class_loss + odm_localization_loss

        return total_loss, odm_class_loss, odm_localization_loss, arm_class_loss, arm_localization_loss


class RefineDetSimpleLoss(nn.Module):
    def __init__(self, match_thresh=0.5, neg_pos=3,  variance=(0.1, 0.2), theta=0.99):
        super().__init__()

        self.loss = SSDGuaranteeLoss(match_thresh=match_thresh, neg_pos=neg_pos,  variance=variance)


        self.match_thresh = match_thresh
        self.neg_pos = neg_pos
        self.variance = variance
        self.theta = theta

    def forward(self, model_output, targets):
        objectness, refine_loc, pred_conf, pred_loc, anchors = model_output
        batch_size, anchors_num, n_classes = pred_conf.shape
        targets_objectness = [[target[0], torch.zeros_like(target[1])]for target in targets]

        arm_class_loss, arm_localization_loss, _ = self.loss((objectness, refine_loc, anchors), targets_objectness)

        refined_anchors = decode_batch(refine_loc, anchors[0], self.variance)

        # #Draw==================================================================
        # is_object = objectness[:, :, 1] > objectness[:, :, 0]
        # second_conf, second_max_idx = torch.max(F.softmax(pred_conf, dim=2), dim=2)
        # second_is_object = second_max_idx > 0
        # print("second", torch.sum(second_is_object))
        # print("first", torch.sum(is_object))
        # import cv2
        # for i in range(len(refined_anchors)):
        #     image = np.zeros((320, 320, 3), dtype=np.uint8)
        #
        #     target_boxes = targets[i][0]
        #     image = draw_boxes(image, target_boxes, color=(0, 255, 0), thickness=3)
        #
        #     boxes = refined_anchors[i][is_object[i]]
        #     image = draw_boxes(image, boxes, thickness=2)
        #
        #     second_boxes = decode_batch(pred_loc[i:i+1], refined_anchors[i], self.variance)[0]
        #     second_boxes = second_boxes[second_is_object[i]]
        #     image = draw_boxes(image, second_boxes, color=(0, 0, 255), thickness=1)
        #
        #
        #     cv2.imshow("image", image)
        #     print(len(second_boxes))
        #     cv2.waitKey(1)
        #
        # #Draw==================================================================
        N = 0
        odm_localization_loss = 0
        odm_class_loss = 0
        for batch_i in range(batch_size):
            pred_conf_i = pred_conf[batch_i:batch_i+1]
            pred_loc_i = pred_loc[batch_i:batch_i+1]
            refined_anchors_i = refined_anchors[batch_i:batch_i+1]

            odm_class_loss_i, odm_localization_loss_i, N_i = \
                self.loss((pred_conf_i, pred_loc_i, refined_anchors_i), targets[batch_i : batch_i + 1])

            N += N_i
            odm_class_loss += N_i * odm_class_loss_i
            odm_localization_loss += N_i * odm_localization_loss_i

        odm_class_loss = odm_class_loss / N
        odm_localization_loss = odm_localization_loss / N

        total_loss = arm_class_loss + arm_localization_loss + odm_class_loss + odm_localization_loss

        return total_loss, odm_class_loss, odm_localization_loss, arm_class_loss, arm_localization_loss



def transform_output_simple(model_output, conf_thresh=0.5, nms_threshold=0.5, variance=(0.1, 0.2)):
    objectness, refine_loc, conf_batch, loc_batch, anchors = model_output
    anchors = anchors[0]
    batch_size = conf_batch.shape[0]

    refined_anchors = decode_batch(refine_loc, anchors, variance)
    # import cv2
    # is_object = objectness[:, :, 1] > objectness[:, :, 0]
    # images = []
    # for i in range(batch_size):
    #     image = np.zeros((320, 320, 3), dtype=np.uint8)
    #     boxes = refined_anchors[i][is_object[i]]
    #     image = draw_boxes(image, boxes, thickness=2)
    #     images.append(image)


    conf_batch_softmax = F.softmax(conf_batch, dim=2)
    conf_batch_softmax = conf_batch_softmax.cpu().numpy()
    loc_batch = loc_batch.cpu().numpy()
    refined_anchors = refined_anchors.cpu().numpy()

    batch_scores = np.max(conf_batch_softmax, axis=2)
    batch_labels = np.argmax(conf_batch_softmax, axis=2)

    batch_selected_boxes, batch_selected_scores, batch_selected_labels = [], [], []
    for batch_i in range(batch_size):
        loc_i = loc_batch[batch_i]
        scores_i = batch_scores[batch_i]
        labels_i = batch_labels[batch_i]
        refined_anchors_i = refined_anchors[batch_i]
        # Decode bboxes
        boxes, labels, indices = SSDDecode_np(labels_i, loc_i, refined_anchors_i)
        scores = scores_i[indices]

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


    # for i in range(batch_size):
    #     image = images[i]
    #     image = draw_boxes(image, batch_selected_boxes[i], thickness=1, color=(0, 0, 255))
    #
    #     cv2.imshow("image", image)
    #     cv2.waitKey(0)

    return batch_selected_boxes, batch_selected_scores, batch_selected_labels