import os
import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


from models import MobileNetV1SSD
from pydet.criterion.ssd import MultiboxLoss

from pydet.data import CocoDataset, detection_collate
from pydet.data.transform import Compose, PhotometricDistort, Expand, RandomSSDCrop
from pydet.data.transform import RandomMirror, Resize, Normalize, ChannelsFirst, ImageToTensor

from pydet.validation import DetectionValidator

# General
#====================================================================================
batch_size = 32

max_iterations = 400000

debug_steps = 500
val_steps = 5000

logdir = os.path.join("mobilenet_ssd", datetime.datetime.now().strftime("%d-%m-%y_%H-%M"))


# Data
#====================================================================================
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


root = "/home/renatkhiz/ram/mscoco"
train_dataset = CocoDataset(root, set_name='train2017', transform=train_transform)
val_dataset = CocoDataset(root, set_name='val2017', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size,
                          num_workers=4,
                          shuffle=True, collate_fn=detection_collate)
val_loader = DataLoader(val_dataset, batch_size,
                        num_workers=4,
                        shuffle=False, collate_fn=detection_collate)

num_classes = len(train_dataset.class_names)


# Model
#====================================================================================
model = MobileNetV1SSD(num_classes)
criterion = MultiboxLoss(iou_threshold=0.5, neg_pos_ratio=3, variances=(0.1, 0.2))

# resume_path = ""
# weights = torch.load(resume_path)
# model.load_state_dict(weights.base_net.state_dict())

# finetune_path = ""
# model.finetune_from(finetune_path)


# Optimizer
#====================================================================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=max_iterations)


# Validation
#====================================================================================
validator = DetectionValidator(num_classes, criterion)