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
    def __init__(self, root_dir, batch_size, transforms):
        VOCDetection.__init__(self,
                              root=root_dir,
                              year='2012',
                              image_set='train',
                              download=False,
                              transforms=transforms)
        self.batch_size = batch_size

    def get_ground_truth_boxes(self, annotations):
        boxes_for_batch = []
        for b_i in range(self.batch_size):
            boxes_for_image = []

            number_of_objects_in_image = len(annotations['annotation']['object'][b_i]['name'])
            for o_i in range(number_of_objects_in_image):
                bbox_coordinates = annotations[o_i]['bbox']
                xmin = annotations['annotation']['object'][b_i]['bndbox']['xmin'][o_i]
                ymin = annotations['annotation']['object'][b_i]['bndbox']['ymin'][o_i]
                xmax = annotations['annotation']['object'][b_i]['bndbox']['xmax'][o_i]
                ymax = annotations['annotation']['object'][b_i]['bndbox']['ymax'][o_i]

                # TODO ...

                if b_i < len(bbox_coordinates[0]):
                    x = bbox_coordinates[0][b_i].to(dtype=torch.float)
                    y = bbox_coordinates[1][b_i].to(dtype=torch.float)
                    w = bbox_coordinates[2][b_i].to(dtype=torch.float)
                    h = bbox_coordinates[3][b_i].to(dtype=torch.float)
                    annotated_category_id = int(annotations[o_i]['category_id'][b_i].item())
                    category_id = self.annotated_to_detected_class_idx[annotated_category_id]

                    box = [x, y, w, h, 1, 1, category_id]
                    boxes_for_image.append(box)

            boxes_for_batch.append(boxes_for_image)

        ground_truth_boxes = torch.tensor(boxes_for_batch)
        return ground_truth_boxes

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
