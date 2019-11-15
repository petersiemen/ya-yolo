import os

import torch
from torchvision.datasets import CocoDetection

from datasets.yayolo_dataset import YaYoloDataset

HERE = os.path.dirname(os.path.realpath(__file__))


class YaYoloCocoDataset(YaYoloDataset, CocoDetection):

    def __init__(self, images_dir, annotations_file, transforms, batch_size):
        CocoDetection.__init__(self, root=images_dir, annFile=annotations_file,
                               transforms=transforms)
        self.annotated_classnames = {k: v['name'] for k, v in self.coco.cats.items()}
        self.detected_classnames = [v['name'] for k, v in self.coco.cats.items()]

        self.annotated_to_detected_class_idx = {k: i for i, (k, v)
                                                in enumerate(self.coco.cats.items())}
        self.batch_size = batch_size

    def get_classnames(self):
        return self.detected_classnames

    def get_ground_truth_boxes(self, annotations):
        boxes_for_batch = []
        for b_i in range(self.batch_size):
            boxes_for_image = []
            for o_i in range(len(annotations)):
                bbox_coordinates = annotations[o_i]['bbox']
                if b_i < len(bbox_coordinates[0]):
                    x = bbox_coordinates[0][b_i]
                    y = bbox_coordinates[1][b_i]
                    w = bbox_coordinates[2][b_i]
                    h = bbox_coordinates[3][b_i]
                    annotated_category_id = int(annotations[o_i]['category_id'][b_i].item())
                    category_id = self.annotated_to_detected_class_idx[annotated_category_id]

                    box = [x, y, w, h, 1, 1, category_id, 1]
                    boxes_for_image.append(box)

            boxes_for_batch.append(boxes_for_image)

        ground_truth_boxes = torch.tensor(boxes_for_batch)
        return ground_truth_boxes

    def __getitem__(self, index):
        return CocoDetection.__getitem__(self, index)

    def __len__(self):
        return CocoDetection.__len__(self)
