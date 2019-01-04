import sys

import torch
import torch.optim as optim
from  torch.utils.data import DataLoader

from pytrainer import Trainer
from pytrainer.callbacks import LearningRateScheduler
from pytrainer.lr_scheduler import Step_decay

from data.augmentations import SSDAugmentation, BaseTransform
from data.coco import CocoDataset
from data.voc import VOCDataset
from data import detection_collate

from models.resnet_ssd import ResnetSSD
from models.ssd_base import SSDLoss
from models.vgg16_ssd import VGG16_SSD
from models.refinedet_base import RefineDetLoss


from validation import DetectionValidator

#============================================================
#COCO
# dataset = CocoDataset("/mnt/dataSSD/renat/mscoco/", set_name="train2017", show=False, transform=SSDAugmentation())
# lr_schedule = Step_decay((280000, 360000, 400000))
# testset = CocoDataset("/mnt/dataSSD/renat/mscoco/", set_name="val2017", show=False, transform=BaseTransform())
# test_loader = DataLoader(
#     testset,
#     batch_size=256,
#     collate_fn=detection_collate,
#     pin_memory=True
# )
# 
# train_loader = DataLoader(
#     dataset,
#     batch_size=64,
#     shuffle=True,
#     collate_fn=detection_collate,
#     pin_memory=True
#     )
#============================================================
#VOC
dataset = VOCDataset("/home/renatkhiz/data/VOCdevkit", transform=SSDAugmentation())
lr_callback = LearningRateScheduler(Step_decay([15000, 25000]), type="batch")
testset = VOCDataset("/home/renatkhiz/data/VOCdevkit", transform=BaseTransform(), image_sets=[('2007', "trainval")])

train_loader = DataLoader(
    dataset,
    batch_size=48,
    shuffle=True,
    collate_fn=detection_collate,
    pin_memory=True,
    num_workers=3
    )

test_loader = DataLoader(
    testset,
    batch_size=48,
    collate_fn=detection_collate,
    pin_memory=True
)

#============================================================
# Resnet
# resnet_idx=18
# model = ResnetSSD(dataset.num_classes(), resnet_idx=resnet_idx, pretrained=True)
#============================================================
# VGG
model = VGG16_SSD(dataset.num_classes())
weights = torch.load("vgg16_reducedfc.pth")
model.backbone.load_state_dict(weights)

#============================================================



criterion = SSDLoss()

optimizer = optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=5e-4)

validator = DetectionValidator(
    test_loader,
    criterion=criterion,
    loss_names=["val_total", "val_conf", "val_loc"],
    save_best=True
)

resume = sys.argv[1] if len(sys.argv) > 1 else None

t = Trainer()

t.compile(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    validation=validator,
    callbacks=[lr_callback]
)
t.fit(
    train_loader,
    report_steps=500,
    val_steps=5000,
    save_steps=5000,
    tag=dataset.name,
)