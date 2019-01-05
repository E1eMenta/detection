import torch
import torch.optim as optim

from models.ssd.ssd_base import SSDLoss
from models.ssd.vgg16_ssd import VGG16SSD
from models.ssd.resnet_ssd import ResnetSSD

from pytrainer.lr_scheduler import Step_decay
from pytrainer.callbacks import LearningRateScheduler

def ModelFactory(backbone, head, **kwargs):
    if backbone == "vgg16" and head == "ssd":
        n_classes = kwargs["n_classes"]
        model = VGG16SSD(n_classes)
        criterion = SSDLoss()
        optimizer = optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=5e-4)

        if "vgg_weights" in kwargs:
            weights = torch.load(kwargs["vgg_weights"])
            model.backbone.load_state_dict(weights)

            lr_schedule = LearningRateScheduler(Step_decay([80000, 100000, 120000]), type="batch")

            return model, criterion, optimizer, lr_schedule

    elif "resnet" in backbone and head == "ssd":
        n_classes = kwargs["n_classes"]
        resnet_idx = int(backbone.replace("resnet", ""))

        model = ResnetSSD(n_classes, resnet_idx, pretrained=True)
        criterion = SSDLoss()
        optimizer = optim.SGD(model.parameters(), 1e-3, momentum=0.9, weight_decay=5e-4)

        lr_schedule = None

        return model, criterion, optimizer, lr_schedule
    else:
        raise Exception("Backbone type {} and detection head {} is unsupported".format(backbone, head))