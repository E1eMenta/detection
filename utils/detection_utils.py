import numpy as np
import cv2
import torch

def IoU_b2b_np(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def IoU_b2v_np(box, boxes):
    """
    Find intersection over union
    :param box: (tensor) One box [xmin, ymin, xmax, ymax], shape: [4].
    :param boxes: (tensor) Shape:[N, 4].
    :return: intersection over union. Shape: [N]
    """

    A = np.maximum(box[:2], boxes[:, :2])
    B = np.minimum(box[2:], boxes[:, 2:])
    interArea = np.maximum(B[:, 0] - A[:, 0], 0) * np.maximum(B[:, 1] - A[:, 1], 0)
    boxArea = (box[2] - box[0]) * (box[3] - box[1])

    boxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = boxArea + boxesArea - interArea
    iou = interArea / union
    return iou

def iou_v2v_np(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [xmin, ymin, xmax, ymax].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''

    N = box1.shape[0]
    M = box2.shape[0]
    lt = np.maximum(
        np.expand_dims(box1[:, :2], axis=1),    # [N,2] -> [N,1,2]
        np.expand_dims(box2[:, :2], axis=0)     # [M,2] -> [1,M,2]
    )                                           # -> [N,M,2]
    rb = np.minimum(
        np.expand_dims(box1[:, 2:], axis=1),    # [N,2] -> [N,1,2] -> [N,M,2]
        np.expand_dims(box2[:, 2:], axis=0)     # [M,2] -> [1,M,2] -> [N,M,2]
    )                                           # -> [N,M,2]

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = np.expand_dims(area1, axis=1)   # [N,] -> [N,1]
    area2 = np.expand_dims(area2, axis=0)   # [M,] -> [1,M]

    iou = inter / (area1 + area2 - inter)   # -> [N,M]
    return iou

def iou_v2v(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [xmin, ymin, xmax, ymax].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''

    N = box1.shape[0]
    M = box2.shape[0]
    lt = torch.max(
        torch.unsqueeze(box1[:, :2], dim=1),    # [N,2] -> [N,1,2]
        torch.unsqueeze(box2[:, :2], dim=0)     # [M,2] -> [1,M,2]
    )                                           # -> [N,M,2]
    rb = torch.min(
        torch.unsqueeze(box1[:, 2:], dim=1),    # [N,2] -> [N,1,2] -> [N,M,2]
        torch.unsqueeze(box2[:, 2:], dim=0)     # [M,2] -> [1,M,2] -> [N,M,2]
    )                                           # -> [N,M,2]

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = torch.unsqueeze(area1, dim=1)   # [N,] -> [N,1]
    area2 = torch.unsqueeze(area2, dim=0)   # [M,] -> [1,M]

    iou = inter / (area1 + area2 - inter)   # -> [N,M]
    return iou

def iou_point_np(box, boxes):
    """
    Find intersection over union
    :param box: (tensor) One box [xmin, ymin, xmax, ymax], shape: [4].
    :param boxes: (tensor) Shape:[N, 4].
    :return: intersection over union. Shape: [N]
    """

    A = np.maximum(box[:2], boxes[:, :2])
    B = np.minimum(box[2:], boxes[:, 2:])
    interArea = np.maximum(B[:, 0] - A[:, 0], 0) * np.maximum(B[:, 1] - A[:, 1], 0)
    boxArea = (box[2] - box[0]) * (box[3] - box[1])

    boxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = boxArea + boxesArea - interArea
    iou = interArea / union
    return iou

def SSDDecode_np(labels, loc, anchors, variances=(0.1, 0.2)):
    clipped_loc = np.clip(loc, a_min=-10e5, a_max=10)

    clipped_loc = clipped_loc[labels > 0]
    anchors = anchors[labels > 0]
    label = labels[labels > 0] - 1
    selected_indices = np.where(labels > 0)

    boxes = np.concatenate((
        anchors[:, :2] + clipped_loc[:, :2] * variances[0] * anchors[:, 2:],
        anchors[:, 2:] * np.exp(clipped_loc[:, 2:] * variances[1])), axis=1)

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes, label, selected_indices

def SSDEncode_batch_np(targets, anchors, variances=(0.1, 0.2), threshold=0.5):
    anchors_num = anchors.shape[0]
    conf_batch = []
    loc_batch = []
    anchors_point = point_form_np(anchors)
    for batch_idx, target in enumerate(targets):
        conf = np.zeros((anchors_num), dtype=np.int32)
        loc = np.ones((anchors_num, 4), dtype=np.float32)

        boxes = target[0]
        labels = target[1]

        if len(boxes) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue
        overlaps = iou_v2v_np(boxes, anchors_point)
        overlaps[overlaps < threshold] = 0

        max_iou = np.max(overlaps, axis=0)
        max_idx = np.argmax(overlaps, axis=0)
        max_idx = max_idx[max_iou > 0]

        conf[max_iou > 0] = labels[max_idx] + 1

        matched_boxes = boxes[max_idx]
        matched_anchors = anchors[max_iou > 0]

        g_cxcy = ((matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2) - matched_anchors[:, :2]
        g_cxcy /= (variances[0] * matched_anchors[:, 2:])
        g_wh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / matched_anchors[:, 2:]
        g_wh = np.log(g_wh) / variances[1]
        matched_loc = np.concatenate([g_cxcy, g_wh], axis=1)
        loc[max_iou > 0] = matched_loc

        conf_batch.append(conf)
        loc_batch.append(loc)

    conf_batch = np.stack(conf_batch)#.astype(np.int32)
    loc_batch = np.stack(loc_batch)

    return conf_batch, loc_batch

def SSDEncode_batch(targets, anchors, variances=(0.1, 0.2), threshold=0.5):
    anchors_num = anchors.shape[0]
    conf_batch = []
    loc_batch = []
    anchors_point = point_form(anchors)
    for batch_idx, target in enumerate(targets):
        conf = torch.zeros((anchors_num), dtype=torch.int64).to(anchors.device)
        loc = torch.ones((anchors_num, 4), dtype=torch.float32).to(anchors.device)

        boxes = target[0]
        labels = target[1]

        if len(boxes) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue
        overlaps = iou_v2v(boxes, anchors_point)
        overlaps[overlaps < threshold] = 0

        max_iou, max_idx = torch.max(overlaps, dim=0)
        max_idx = max_idx[max_iou > 0]
        if len(max_idx) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue

        conf[max_iou > 0] = labels[max_idx] + 1

        matched_boxes = boxes[max_idx]
        matched_anchors = anchors[max_iou > 0]

        g_cxcy = ((matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2) - matched_anchors[:, :2]
        g_cxcy /= (variances[0] * matched_anchors[:, 2:])
        g_wh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / matched_anchors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        matched_loc = torch.cat([g_cxcy, g_wh], dim=1)
        loc[max_iou > 0] = matched_loc

        conf_batch.append(conf)
        loc_batch.append(loc)

    conf_batch = torch.stack(conf_batch)
    loc_batch = torch.stack(loc_batch)

    return conf_batch, loc_batch

def GuaranteeEncode_batch_np(targets, anchors, variances=(0.1, 0.2), threshold=0.5):
    anchors_num = anchors.shape[0]
    conf_batch = []
    loc_batch = []
    anchors_point = point_form_np(anchors)
    for batch_idx, target in enumerate(targets):
        conf = np.zeros((anchors_num), dtype=np.int32)
        loc = np.ones((anchors_num, 4), dtype=np.float32)

        boxes = target[0]
        labels = target[1]

        if len(boxes) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue
        overlaps = iou_v2v_np(boxes, anchors_point)
        max_idx = np.argmax(overlaps, axis=1)
        overlaps[np.arange(0, len(max_idx)), max_idx] = 1.0
        overlaps[overlaps < threshold] = 0

        max_iou = np.max(overlaps, axis=0)
        max_idx = np.argmax(overlaps, axis=0)
        max_idx = max_idx[max_iou > 0]

        conf[max_iou > 0] = labels[max_idx] + 1

        matched_boxes = boxes[max_idx]
        matched_anchors = anchors[max_iou > 0]

        g_cxcy = ((matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2) - matched_anchors[:, :2]
        g_cxcy /= (variances[0] * matched_anchors[:, 2:])
        g_wh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / matched_anchors[:, 2:]
        g_wh = np.log(g_wh) / variances[1]
        matched_loc = np.concatenate([g_cxcy, g_wh], axis=1)
        loc[max_iou > 0] = matched_loc

        conf_batch.append(conf)
        loc_batch.append(loc)

    conf_batch = np.stack(conf_batch)#.astype(np.int32)
    loc_batch = np.stack(loc_batch)

    return conf_batch, loc_batch

def GuaranteeEncode_batch(targets, anchors, variances=(0.1, 0.2), threshold=0.5):
    anchors_num = anchors.shape[0]
    conf_batch = []
    loc_batch = []
    anchors_point = point_form(anchors)
    for batch_idx, target in enumerate(targets):
        conf = torch.zeros((anchors_num), dtype=torch.int64).to(anchors.device)
        loc = torch.ones((anchors_num, 4), dtype=torch.float32).to(anchors.device)

        boxes = target[0]
        labels = target[1]

        if len(boxes) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue
        overlaps = iou_v2v(boxes, anchors_point)
        max_idx = torch.argmax(overlaps, dim=1)
        overlaps[torch.arange(0, len(max_idx)), max_idx] = 1.0
        overlaps[overlaps < threshold] = 0

        max_iou, max_idx = torch.max(overlaps, dim=0)
        max_idx = max_idx[max_iou > 0]
        if len(max_idx) == 0:
            conf_batch.append(conf)
            loc_batch.append(loc)
            continue

        conf[max_iou > 0] = labels[max_idx] + 1

        matched_boxes = boxes[max_idx]
        matched_anchors = anchors[max_iou > 0]

        g_cxcy = ((matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2) - matched_anchors[:, :2]
        g_cxcy /= (variances[0] * matched_anchors[:, 2:])
        g_wh = (matched_boxes[:, 2:] - matched_boxes[:, :2]) / matched_anchors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]
        matched_loc = torch.cat([g_cxcy, g_wh], dim=1)
        loc[max_iou > 0] = matched_loc

        conf_batch.append(conf)
        loc_batch.append(loc)

    conf_batch = torch.stack(conf_batch)
    loc_batch = torch.stack(loc_batch)

    return conf_batch, loc_batch

def draw_boxes(image, boxes, labels, color=(255, 0, 0), thickness=1):
    img = image.copy()
    boxes = boxes.copy()
    h, w, _ = img.shape

    boxes[:, 0] *= w
    boxes[:, 1] *= h
    boxes[:, 2] *= w
    boxes[:, 3] *= h

    target = boxes.astype(np.int32)
    for bbox in target:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

    return img



def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def point_form_np(boxes):
    """ Convert boxes from center form (cx, cy, w, h) to point form (xmin, ymin, xmax, ymax)
    Args:
        boxes: (tensor) Boxes in center form.
    Return:
        boxes: (tensor) Converted point form of boxes.
    """
    return np.concatenate((boxes[:, :2] - boxes[:, 2:]/2,      # xmin, ymin
                           boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def decode_np(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    corrected_loc = np.clip(loc, a_min=-10e5, a_max=10)
    boxes = np.concatenate((
        priors[:, :2] + corrected_loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(corrected_loc[:, 2:] * variances[1])), axis=1)

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    corrected_loc = torch.clamp(loc, min=-10e5, max=10)
    boxes = torch.cat(
        [
            priors[:, :2] + corrected_loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(corrected_loc[:, 2:] * variances[1])
        ],
        dim=1)

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes
def decode_batch(loc_batch, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    priors = torch.unsqueeze(priors, dim=0)
    corrected_loc = torch.clamp(loc_batch, min=-10e5, max=10)
    # a = corrected_loc[:, :, :2] * variances[0] * priors[:, :, 2:]
    # a = a +  priors[:, :, :2]
    # b = priors[:, :, 2:] * torch.exp(corrected_loc[:, :, 2:] * variances[1])
    boxes = torch.cat(
        [
            priors[:, :, :2] + corrected_loc[:, :, :2] * variances[0] * priors[:, :, 2:],
            priors[:, :, 2:] * torch.exp(corrected_loc[:, :, 2:] * variances[1])
        ],
        dim=2)

    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes

def NMS_np(
        boxes,
        scores,
        iou_threshold=0.5,
        max_output_size=None
    ):

    if len(scores) == 0:
        return []
    idx_sort = np.argsort(scores)
    boxes = boxes[idx_sort]

    selected_indices = [idx_sort[0]]
    chosen_boxes = np.expand_dims(boxes[0], axis=0)

    for idx, box in enumerate(boxes[1:]):
        real_idx = idx + 1

        IoUs = IoU_b2v_np(box, chosen_boxes)

        if np.sum(IoUs > iou_threshold) == 0:
            selected_indices.append(idx_sort[real_idx])
            chosen_boxes = np.concatenate([chosen_boxes, np.expand_dims(box, axis=0)], axis=0)
            if max_output_size is not None:
                if len(chosen_boxes) >= max_output_size:
                    break
    selected_indices = np.stack(selected_indices)

    return selected_indices

