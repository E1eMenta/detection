import torch
import torch.optim as optim

from models.ssd.ssd_base import SSDLoss
from models.ssd import Resnet18SSD, Resnet34SSD, Resnet50SSD, Resnet101SSD, Resnet152SSD, VGG16SSD

from pytrainer.lr_scheduler import Step_decay
from pytrainer.callbacks import LearningRateScheduler

def ModelFactory(backbone, head, **kwargs):
    if backbone == "vgg16" and head == "ssd":
        n_classes = kwargs["n_classes"]
        model = VGG16SSD(n_classes)
        criterion = SSDLoss()
        optimizer = optim.SGD(model.parameters(), kwargs["lr"], momentum=0.9, weight_decay=5e-4)

        if "vgg_weights" in kwargs:
            weights = torch.load(kwargs["vgg_weights"])
            model.backbone.load_state_dict(weights)

            lr_schedule = LearningRateScheduler(Step_decay([80000, 100000, 120000]), type="batch")

            return model, criterion, optimizer, lr_schedule

    elif "resnet" in backbone and head == "ssd":
        n_classes = kwargs["n_classes"]
        resnet_idx = int(backbone.replace("resnet", ""))
        if resnet_idx == 18:
            model = Resnet18SSD(n_classes, pretrained=True)
            lr_schedule = None
        elif resnet_idx == 34:
            model = Resnet34SSD(n_classes, pretrained=True)
            lr_schedule = None
        elif resnet_idx == 50:
            model = Resnet50SSD(n_classes, pretrained=True)
            lr_schedule = None
        elif resnet_idx == 101:
            model = Resnet101SSD(n_classes, pretrained=True)
            lr_schedule = None
        elif resnet_idx == 152:
            model = Resnet152SSD(n_classes, pretrained=True)
            lr_schedule = None
        else:
            raise Exception("Unknown resnet index {}".format(resnet_idx))
        criterion = SSDLoss()
        optimizer = optim.SGD(model.parameters(), kwargs["lr"], momentum=0.9, weight_decay=5e-4)



        return model, criterion, optimizer, lr_schedule
    else:
        raise Exception("Backbone type {} and detection head {} is unsupported".format(backbone, head))