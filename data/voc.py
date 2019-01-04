import numpy as np
from  torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os.path as osp
import cv2

from albumentations import Compose, RandomSizedCrop, SmallestMaxSize
from data.augmentations import SSDAugmentation, BaseTransform

from utils.vis import draw_boxes

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]



VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class VOCDataset(Dataset):
    """VOC Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, max_ims=None, show=False):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.max_ims = max_ims
        self.show = show
        self.target_transform = VOCAnnotationTransform()
        self.name = 'VOC0712'
        self.class_names = VOC_CLASSES

        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.n_classes = len(VOC_CLASSES)
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __len__(self):
        if self.max_ims != None:
            return self.max_ims
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        target = np.array(target)

        img = img.astype(np.float32)
        bboxes = target[:, :4].astype(np.float32)
        labels = target[:, 4]
        sample = {"image": img, "bboxes": bboxes, "labels": labels}

        if self.transform:
            sample = self.transform(**sample)

        height, width, _ = sample['image'].shape
        if len(sample['bboxes']) == 0:
            bboxes = np.zeros((0, 4))
        else:
            bboxes = np.stack(sample['bboxes'])
        bboxes[:, 0] /= width
        bboxes[:, 1] /= height
        bboxes[:, 2] /= width
        bboxes[:, 3] /= height
        sample['bboxes'] = bboxes

        if self.show:
            image = sample["image"].copy().astype(np.uint8)
            image = draw_boxes(image, sample["bboxes"], sample["labels"], class_idx_to_name=VOC_CLASSES)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        return sample

    def num_classes(self):
        return len(VOC_CLASSES)

# aug = [SmallestMaxSize(max_size=301), RandomSizedCrop([300, 300], 300, 300), ]
# transform = Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': 0, 'min_visibility': 0.5, 'label_fields': ['labels']})

# transform = SSDAugmentation()
# dataset = VOCDataset("/home/renatkhiz/VOCdevkit", transform=transform, show=True)
# for i in range(len(dataset)):
#     dataset[i]