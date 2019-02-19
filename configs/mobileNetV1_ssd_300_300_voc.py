import os
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR



from vision.datasets.voc_dataset import VOCDataset

from models import MobileNetV1SSD
from pydet.criterion.ssd import MultiboxLoss
from pydet.head.ssd import SSDPostprocess

from pydet.data import CocoDataset, detection_collate
from pydet.data.transform import Compose, PhotometricDistort, Expand, RandomSSDCrop
from pydet.data.transform import RandomMirror, Resize, Normalize, ChannelsFirst, ImageToTensor

from pydet.validation import DetectionValidator
# General
#====================================================================================
batch_size = 24

max_iterations = 140000

debug_steps = 100
val_steps = 5000

logdir = os.path.join("mobilenet_ssd", datetime.datetime.now().strftime("%d-%m-%y_%H-%M"))


# Data
#====================================================================================
image_size = 300
image_mean = np.array([127, 127, 127])
image_std = 128.0

datasets_paths = ["/mnt/dataSSD/renat/VOCdevkit/VOC2007/", "/mnt/dataSSD/renat/VOCdevkit/VOC2012/"]
validation_dataset = "/mnt/dataSSD/renat/VOCdevkit/VOC2007/"

image_size = (300, 300)
mean = [127, 127, 127]
std = [128.0, 128.0, 128.0]

train_transform = Compose([
    PhotometricDistort(),
    Expand(mean),
    RandomSSDCrop(),
    RandomMirror(),
    Resize(image_size),
    Normalize(mean, std),
    ChannelsFirst(),
    ImageToTensor()
])

test_transform = Compose([
    Resize(image_size),
    Normalize(mean, std),
    ChannelsFirst(),
    ImageToTensor()
])
datasets = []
for dataset_path in datasets_paths:
    dataset = VOCDataset(dataset_path, transform=train_transform)
    datasets.append(dataset)
train_dataset = ConcatDataset(datasets)
num_classes = len(datasets[0].class_names)
train_loader = DataLoader(train_dataset, batch_size,
                          num_workers=4,
                          shuffle=True, collate_fn=detection_collate)

val_dataset = VOCDataset(validation_dataset, transform=test_transform, is_test=True)
val_loader = DataLoader(val_dataset, batch_size,
                        num_workers=4,
                        shuffle=False, collate_fn=detection_collate)


# Model
#====================================================================================
model = MobileNetV1SSD(num_classes)

criterion = MultiboxLoss(iou_threshold=0.5, neg_pos_ratio=3, variances=(0.1, 0.2))

# Optimizer
#====================================================================================
lr = 0.01
momentum = 0.9
t_max = 140000

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
scheduler = CosineAnnealingLR(optimizer, t_max)


# Validation
#====================================================================================
validator = DetectionValidator(val_loader, SSDPostprocess, criterion)