import os
import cv2
import numpy as np
import xmltodict
import xml.etree.ElementTree

from torch.utils.data import Dataset

def get_ann_path(path):
    parts = path.split(os.sep)
    parts[-5] = "Annotations"
    return "/" + os.path.join(*parts)
class ImagenetVID(Dataset):
    def __init__(self, root, set_name, transform=None):
        self.transform = transform

        self.class_names = ["airplane","antelope","bear","bicycle","bird","bus","car","cattle","dog",
                            "domestic_cat", "elephant", "fox", "giant_panda", "hamster", "horse", "lion",
                            "lizard", "monkey", "motorcycle", "rabbit", "red_panda", "sheep", "snake",
                            "squirrel", "tiger", "train", "turtle", "watercraft", "whale", "zebra"]
        self.class_labels = [
            "n02691156","n02419796","n02131653","n02834778","n01503061","n02924116","n02958343","n02402425",
            "n02084071","n02121808","n02503517","n02118333","n02510455","n02342885","n02374451","n02129165",
            "n01674464","n02484322","n03790512","n02324045","n02509815","n02411705","n01726692","n02355227",
            "n02129604","n04468005","n01662784","n04530566","n02062744","n02391049"]

        image_root = os.path.join(root, "Data", "VID", set_name)
        ann_root = os.path.join(root, "Annotations", "VID", set_name)

        data_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(image_root) for f in fn]
        rel_paths = [os.path.relpath(path, image_root) for path in data_paths]
        rel_paths = [os.path.splitext(path)[0] for path in rel_paths]

        self.annotations = []
        self.image_paths = []
        for rel_path in rel_paths:
            image_path = os.path.join(image_root, rel_path + ".JPEG")
            ann_path = os.path.join(ann_root, rel_path + ".xml")

            xml_root = xml.etree.ElementTree.parse(ann_path).getroot()
            if len(xml_root) < 5:
                continue
            class_label = xml_root[4][1].text
            class_idx = self.class_labels.index(class_label) + 1
            xmax = int(xml_root[4][2][0].text)
            xmin = int(xml_root[4][2][1].text)
            ymax = int(xml_root[4][2][2].text)
            ymin = int(xml_root[4][2][3].text)

            self.annotations.append((np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32),
                                     np.array([class_idx], dtype=np.int64)))
            self.image_paths.append(image_path)

        pass

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, item):
        image = cv2.imread(self.image_paths[item])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        return image


    def __getitem__(self, item):
        image = self.load_image(item)
        boxes, labels = self.annotations[item]
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        return image, (boxes, labels)

# dataset = ImagenetVID("/mnt/media2/renat/datasets/Imagenet_VID/ILSVRC2015_VID", "val")
#
# for i in range(len(dataset)):
#     image, (boxes, labels) = dataset[i]
#     image = image.astype(np.uint8)
#     height, width, _ = image.shape
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         x1 = int(x1 * width)
#         y1 = int(y1 * height)
#         x2 = int(x2 * width)
#         y2 = int(y2 * height)
#         image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
#     cv2.imshow("image", image)
#     cv2.waitKey()