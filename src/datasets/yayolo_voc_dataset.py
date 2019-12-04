import os

import torch
from torchvision.datasets import VOCDetection
from PIL import Image

import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class YaYoloVocDataset(VOCDetection):
    def __init__(self, root_dir, batch_size, transforms, image_set='train', download=False):
        VOCDetection.__init__(self,
                              root=root_dir,
                              year='2012',
                              image_set=image_set,
                              download=download,
                              transforms=transforms)
        self.batch_size = batch_size

    def _get_ground_truth_box(self, bndbox, width, height, name):
        xmin = int(bndbox['xmin'])
        ymin = int(bndbox['ymin'])
        xmax = int(bndbox['xmax'])
        ymax = int(bndbox['ymax'])
        x = xmin + (xmax - xmin) / 2
        y = ymin + (ymax - ymin) / 2
        w = xmax - xmin
        h = ymax - ymin
        class_id = self._annotation_to_groundtruth_boxes(name)
        return torch.tensor([x / width, y / height, w / width, h / height, 1, 1, class_id], dtype=torch.float)

    def _annotation_to_groundtruth_boxes(self, annotation):
        width, height = int(annotation['size']['width']), int(annotation['size']['height'])

        ground_truth_boxes = []
        if isinstance(annotation['object'], list):
            for i in range(len(annotation['object'])):
                name = annotation['object'][i]['name']
                ground_truth_boxes.append(
                    self._get_ground_truth_box(annotation['object'][i]['bndbox'], width, height, name))
        else:
            name = annotation['object']['name']
            ground_truth_boxes.append(self._get_ground_truth_box(annotation['object']['bndbox'], width, height, name))
        return ground_truth_boxes

    def collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        target = [self._annotation_to_groundtruth_boxes(item[1]['annotation']) for item in batch]
        image_paths = [item[2] for item in batch]
        return images, target, image_paths

    def _get_class_id(self, name):
        return 2

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
