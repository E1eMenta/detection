import os
import math
import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


from models import Resnet50_FPN_SSD
from pydet.criterion.ssd import MultiboxLoss

from pydet.data import CocoDataset, detection_collate
from pydet.data.transform import Compose, PhotometricDistort, Expand, RandomSSDCrop
from pydet.data.transform import RandomMirror, Resize, Normalize, ChannelsFirst, ImageToTensor

from pydet.validation import DetectionValidator

class CosineAnnealingWarmup(_LRScheduler):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, T_warmup=0, alfa=1/3):
        self.T_max = T_max
        self.eta_min = eta_min
        self.T_warmup = T_warmup
        self.alfa = alfa
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            min_lrs = [self.alfa * base_lr for base_lr in self.base_lrs]
            lrs = [min_lr + (base_lr - min_lr) * self.last_epoch / self.T_warmup
                for base_lr, min_lr in zip(self.base_lrs, min_lrs)]
            return lrs
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / self.T_max)) / 2
                    for base_lr in self.base_lrs]

# General
#====================================================================================
batch_size = 32

max_iterations = 750000

debug_steps = 500
val_steps = 5000

logdir = os.path.join("resnet_fpn_ssd", datetime.datetime.now().strftime("%d-%m-%y_%H-%M"))

cudnn = False
clip_norm = 10
# Data
#====================================================================================
image_size = (300, 300)
mean = [0, 0, 0]
std = [255.0, 255.0, 255.0]


torchvision_mean = [0.485, 0.456, 0.406]
torchvision_std = [0.229, 0.224, 0.225]

train_transform = Compose([
    PhotometricDistort(),
    Expand(mean),
    RandomSSDCrop(),
    RandomMirror(),
    Resize(image_size),
    Normalize(mean, std),
    Normalize(torchvision_mean, torchvision_std),
    ChannelsFirst(),
    ImageToTensor()
])

test_transform = Compose([
    Resize(image_size),
    Normalize(mean, std),
    Normalize(torchvision_mean, torchvision_std),
    ChannelsFirst(),
    ImageToTensor()
])


root = "/mnt/ram/mscoco"
train_dataset = CocoDataset(root, set_name='train2017', transform=train_transform)
val_dataset = CocoDataset(root, set_name='val2017', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size,
                          num_workers=4,
                          shuffle=True, collate_fn=detection_collate)
val_loader = DataLoader(val_dataset, batch_size,
                        num_workers=4,
                        shuffle=False, collate_fn=detection_collate)

num_classes = len(train_dataset.class_names)
label_map = train_dataset.class_names


# Model
#====================================================================================
print("MobileNetV1_FPN_SSD")
model = Resnet50_FPN_SSD(num_classes)
criterion = MultiboxLoss(iou_threshold=0.5, neg_pos_ratio=3, variances=(0.1, 0.2))

# resume_path = ""
# weights = torch.load(resume_path)
# model.load_state_dict(weights.base_net.state_dict())

# finetune_path = ""
# model.finetune_from(finetune_path)


# Optimizer
#====================================================================================
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
scheduler = CosineAnnealingWarmup(optimizer, T_max=max_iterations)


# Validation
#====================================================================================
validator = DetectionValidator(num_classes, criterion)