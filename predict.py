import os
import cv2
import json
import torch
import argparse
import numpy as np

from pydet.utils.numpy import postprocess
from pydet.utils.vis import draw_boxes
from pydet.data.transform import Compose, Normalize, ChannelsFirst, ImageToTensor

parser = argparse.ArgumentParser(description='Run detector on data')
parser.add_argument('--checkpoint', required=True, type=str, help='Path to model')
parser.add_argument('--folder', required=True, type=str, help='Path to model')
parser.add_argument('--label_map', required=True, type=str, help='Path to label map')
parser.add_argument('--preprocess', type=str, default="pad_resizer", help='Image preprocessing function.'
                                                                          'Allowed: pad_resizer, fixed_resizer')
parser.add_argument('--height', default=300, type=int, help='Height of resized image')
parser.add_argument('--width', default=300, type=int, help='Width of resized image')
parser.add_argument('--bs', default=32, type=int, help='Batch size of images')
parser.add_argument('--logdir', type=str, help='Path to folder, where to save logs')
parser.add_argument('--show', default=False, type=bool, help='Show detections')
args = parser.parse_args()

class FixedResizer:
    def __init__(self, height, width, interpolation=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, image):
        image = cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)
        return image

class PadResizer:
    def __init__(self, height, width, border_value=[0, 0, 0], interpolation=cv2.INTER_AREA):
        self.height = height
        self.width = width
        self.border_value = border_value
        self.interpolation = interpolation

    def __call__(self, image):
        src_height, src_width, _ = image.shape

        dst_ar = self.width / self.height
        src_ar = src_width / src_height
        if dst_ar > src_ar:
            bottom = 0
            right = int(src_ar * src_height - src_width)
        else:
            bottom = int(src_width / src_ar - src_height)
            right = 0

        padded_image = cv2.copyMakeBorder(image, top=0, bottom=bottom, left=0, right=right,
                                          borderType=cv2.BORDER_CONSTANT, value=self.border_value)

        resized_image = cv2.resize(padded_image, (self.width, self.height), interpolation=self.interpolation)
        return resized_image


class BaseTransform:
    def __init__(self, mean, std):
        self.transform = Compose([
            Normalize(mean, std),
            ChannelsFirst(),
            ImageToTensor()
        ])

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _, _ = self.transform(image, None, None)
        return image

def isImage(path):
    exts = ['.png', '.jpg', '.jpeg', '.gif', '.png']
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    return ext in exts


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.checkpoint, map_location='cpu')
    model = model.to(DEVICE)
    model = torch.nn.DataParallel(model)
    model.eval()

    with open(args.label_map) as f:
        label_map = f.readlines()
    label_map = [line.rstrip() for line in label_map]


    image_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(args.folder) for f in fn if isImage(f)]
    image_chunks = [image_paths[i:i + args.bs] for i in range(0, len(image_paths), args.bs)]

    mean = [127, 127, 127]
    std = [128.0, 128.0, 128.0]

    base = BaseTransform(mean, std)

    if args.preprocess == "fixed_resizer":
        transform = FixedResizer(args.height, args.width)
    elif args.preprocess == "pad_resizer":
        transform = PadResizer(args.height, args.width, border_value=mean)

    with torch.no_grad():
        for image_path_chunk in image_chunks:
            images_batch = [cv2.imread(image_path) for image_path in image_path_chunk]
            prep_images = [transform(image.astype(np.float32)) for image in images_batch]
            prep_images = [base(image) for image in prep_images]

            prep_images = torch.stack(prep_images)
            prep_images = prep_images.to(DEVICE)

            _, (boxes, labels, scores) = model(prep_images)
            boxes = boxes.cpu().numpy()
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()

            boxes, labels, scores = postprocess(boxes, labels, scores, score_thresh=0.6)

            if args.show:
                for boxes_i, labels_i, scores_i, image in zip(boxes, labels, scores, images_batch):
                    image = draw_boxes(image, boxes_i, labels_i, label_map, scores=scores_i)
                    cv2.imshow("image", image)
                    cv2.waitKey()

            if args.logdir:
                root = args.folder

                for i in range(len(images_batch)):
                    path = image_path_chunk[i]
                    boxes_i = boxes[i]
                    labels_i = labels[i]
                    scores_i = scores[i]
                    names = [label_map[class_id] for class_id in labels_i]

                    height, width, _ = images_batch[i].shape
                    boxes_i[:, 0] *= width
                    boxes_i[:, 1] *= height
                    boxes_i[:, 2] *= width
                    boxes_i[:, 3] *= height

                    path = os.path.relpath(path, root)
                    path = os.path.join(args.logdir, path)
                    path, _ = os.path.splitext(path)
                    path = path + ".json"
                    os.makedirs(os.path.dirname(path), exist_ok=True)

                    def save_log(path, boxes, labels, scores, names):
                        log = []
                        for box, label, score, name in zip(boxes, labels, scores, names):
                           box_log = {
                               "score": float(score),
                               "label": int(label),
                               "name": name,
                               "box": {
                                   "xmin": float(box[0]),
                                   "ymin": float(box[1]),
                                   "xmax": float(box[2]),
                                   "ymax": float(box[3])
                               }
                           }
                           log.append(box_log)
                           with open(path, 'w') as outfile:
                               json.dump(log, outfile, indent=4)


                    save_log(path, boxes_i, labels_i, scores_i, names)