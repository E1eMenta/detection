import os
import cv2
import json
import torch
import argparse
import numpy as np
import torch.nn as nn

from pydet.utils.numpy import postprocess
from pydet.utils.vis import draw_boxes
from pydet.data.transform import  ChannelsFirst, ImageToTensor, PadToAspectRatio
from albumentations import Compose, Normalize, Resize

parser = argparse.ArgumentParser(description='Run detector on data')
parser.add_argument('--checkpoint', required=True, type=str, help='Path to model')
parser.add_argument('--folder', required=True, type=str, help='Path to model')
parser.add_argument('--label_map', required=True, type=str, help='Path to label map')
parser.add_argument('--bs', default=32, type=int, help='Batch size of images')
parser.add_argument('--logdir', type=str, help='Path to folder, where to save logs')
parser.add_argument('--show', default=False, type=bool, help='Show detections')
#Prepocess flags
parser.add_argument('--preprocess', type=str, default="pad", help='Image preprocessing function.')

args = parser.parse_args()

def isImage(path):
    exts = ['.png', '.jpg', '.jpeg', '.gif', '.png']
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    return ext in exts

class PredictionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, outs = self.model(x)
        return outs


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    module = torch.load(args.checkpoint, map_location='cpu')
    model = PredictionWrapper(module)
    model = model.to(DEVICE)
    model = torch.nn.DataParallel(model)
    model.eval()

    height = module.image_size[0]
    width = module.image_size[0]
    mean = module.mean
    std = module.std
    max_pixel_value = module.max_pixel_value


    with open(args.label_map) as f:
        label_map = f.readlines()
    label_map = [line.rstrip() for line in label_map]


    image_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(args.folder) for f in fn if isImage(f)]
    image_chunks = [image_paths[i:i + args.bs] for i in range(0, len(image_paths), args.bs)]


    if args.preprocess == "fixed":
        resizer = Resize(height, width)
    elif args.preprocess == "pad":
        resizer = Compose([
            PadToAspectRatio(width/height),
            Resize(height, width)
        ])
    else:
        raise Exception(f"Resizer {args.preprocess} is not supported")
    transform = Compose([
        resizer,
        Normalize(mean, std, max_pixel_value),
        ChannelsFirst(),
        ImageToTensor()
    ])

    with torch.no_grad():
        for image_path_chunk in image_chunks:
            images_batch = [cv2.imread(image_path, ) for image_path in image_path_chunk]

            prep_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) for image in images_batch]
            prep_images = [transform(image=image)["image"] for image in prep_images]
            prep_images = torch.stack(prep_images)
            prep_images = prep_images.to(DEVICE)

            boxes, labels, scores = model(prep_images)
            boxes = boxes.cpu().numpy()
            labels = labels.cpu().numpy()
            scores = scores.cpu().numpy()

            boxes, labels, scores = postprocess(boxes, labels, scores, score_thresh=0.5)

            if args.show:
                for boxes_i, labels_i, scores_i, image in zip(boxes, labels, scores, images_batch):
                    image = draw_boxes(image, boxes_i, labels_i, label_map, scores=scores_i)
                    cv2.imshow("image", image)
                    cv2.waitKey()

            if args.logdir:
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

                    path = os.path.relpath(path, args.folder)
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