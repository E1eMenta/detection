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
parser.add_argument('--preprocess', type=str, default="pad_resizer", help='Image preprocessing function.'                                                                          'Allowed: pad_resizer, fixed_resizer')
parser.add_argument('--height', default=300, type=int, help='Height of resized image')
parser.add_argument('--width', default=300, type=int, help='Width of resized image')
# Get next parameters from config file in Normalize fn
parser.add_argument('--mean', default="0.485;0.456;0.406", type=str, help='Image normalization: mean of normalization'
                                                                          'Default: 0.229;0.224;0.225')
parser.add_argument('--std', default="0.229;0.224;0.225", type=str, help='Image normalization: std of normalization'
                                                                         'Default: 0.229;0.224;0.225')
parser.add_argument('--max_pixel_value', default=255.0, type=float, help='Image normalization: maximum possible pixel value.'
                                                                         'Default: 255.0')


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

    model = PredictionWrapper(torch.load(args.checkpoint, map_location='cpu'))
    model = model.to(DEVICE)
    model = torch.nn.DataParallel(model)
    model.eval()

    with open(args.label_map) as f:
        label_map = f.readlines()
    label_map = [line.rstrip() for line in label_map]


    image_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(args.folder) for f in fn if isImage(f)]
    image_chunks = [image_paths[i:i + args.bs] for i in range(0, len(image_paths), args.bs)]

    mean = [float(value) for value in args.mean.split(";")]
    std = [float(value) for value in args.std.split(";")]


    if args.preprocess == "fixed":
        resizer = Resize(args.height, args.width)
    elif args.preprocess == "pad":
        resizer = Compose([
            PadToAspectRatio(args.width/args.height),
            Resize(args.height, args.width)
        ])
    else:
        raise Exception(f"Resizer {args.preprocess} is not supported")
    transform = Compose([
        resizer,
        Normalize(mean, std, max_pixel_value=args.max_pixel_value),
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