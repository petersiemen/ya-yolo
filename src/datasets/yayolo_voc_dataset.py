import os

import torch
from torchvision.datasets import VOCDetection
from PIL import Image

from datasets.yayolo_dataset import YaYoloDataset
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class YaYoloVocDataset(YaYoloDataset, VOCDetection):
    def __init__(self, root_dir, transforms):
        VOCDetection.__init__(
            root=root_dir,
            year='2012',
            image_set='train',
            download=False,
            transforms=transforms)

    def get_ground_truth_boxes(self, annotations):
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        full_image_path = os.path.join(self.root, self.images[index])
        img = Image.open(full_image_path).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, full_image_path

    def __len__(self):
        return VOCDetection.__len__(self)
