import sys
import argparse

from pytrainer import Trainer

from data_factory import DataFactory
from model_factory import ModelFactory

from validation import DetectionValidator

parser = argparse.ArgumentParser(description='PyTorch Detection Training')
parser.add_argument('--dataset', default="voc", type=str,
                    help='Dataset name (voc or coco)')
parser.add_argument('--root', default="data", type=str,
                    help='Path to dataset root dir')
parser.add_argument('--batch-size', default=32, type=int,
                    help='mini-batch size (default: 32)')
parser.add_argument('--backbone', default="resnet50", type=str,
                    help='CNN backbone of detector (resnet(18,34,50,101,152), vgg16)')
parser.add_argument('--head', default="ssd", type=str,
                    help='Detection head. Only ssd now')
parser.add_argument('--vgg-weights', default="vgg16_reducedfc.pth", type=str,
                    help='Backbone weights for vgg16')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='learning rate')
args = parser.parse_args()


train_loader, test_loader = DataFactory(args.dataset, root=args.root, batch_size=args.batch_size)

model, criterion, optimizer, lr_schedule = ModelFactory(
    args.backbone,
    args.head,
    n_classes=train_loader.dataset.num_classes(),
    vgg_weights=args.vgg_weights,
    lr=args.lr
)

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
    callbacks=[lr_schedule]
)
t.fit(
    train_loader,
    report_steps=500,
    val_steps=5000,
    save_steps=5000,
    tag=train_loader.dataset.name,
)