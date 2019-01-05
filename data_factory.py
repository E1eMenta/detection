from data.voc import VOCDataset
from data.coco import CocoDataset

from data import detection_collate
from torch.utils.data import DataLoader
from data.augmentations import SSDAugmentation, BaseTransform


def DataFactory(dataset, **kwargs):
    if dataset == "voc":
        root = kwargs["root"]
        batch_size = kwargs["batch_size"]

        dataset = VOCDataset(root, transform=SSDAugmentation())
        valset = VOCDataset(root, transform=BaseTransform(), image_sets=[('2007', "trainval")])

        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=detection_collate,
            pin_memory=True,
            num_workers=3
            )
        val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            collate_fn=detection_collate,
            pin_memory=True
        )

        return train_loader, val_loader

    elif dataset == "coco":
        root = kwargs["root"]
        batch_size = kwargs["batch_size"]

        dataset = CocoDataset(root, set_name="train2017", show=False, transform=SSDAugmentation())
        valset = CocoDataset(root, set_name="val2017", show=False, transform=BaseTransform())

        val_loader = DataLoader(
            valset,
            batch_size=batch_size,
            collate_fn=detection_collate,
            pin_memory=True
        )
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=detection_collate,
            pin_memory=True
            )


        return train_loader, val_loader

    else:
        raise Exception("Dataset type {} is unsupported".format(dataset))
