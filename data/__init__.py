from data.coco import CocoDataset

import numpy as np
import torch

def detection_collate(batch):
    images = np.stack([np.transpose(sample['image'], (2, 0, 1)) for sample in batch])
    images = torch.from_numpy(images)
    targets = [(torch.FloatTensor(sample['bboxes']), torch.LongTensor(sample['labels'])) for sample in batch]

    return images, targets