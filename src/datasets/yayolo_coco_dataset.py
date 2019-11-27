import os

import torch
from torchvision.datasets import CocoDetection
from PIL import Image

from datasets.yayolo_dataset import YaYoloDataset


class YaYoloCocoDataset(YaYoloDataset, CocoDetection):

    def __init__(self, images_dir, annotations_file, transforms, batch_size):
        CocoDetection.__init__(self, root=images_dir, annFile=annotations_file,
                               transforms=transforms)
        self.annotated_classnames = {k: v['name'] for k, v in self.coco.cats.items()}
        self.detected_classnames = [v['name'] for k, v in self.coco.cats.items()]

        self.annotated_to_detected_class_idx = {k: i for i, (k, v)
                                                in enumerate(self.coco.cats.items())}
        self.batch_size = batch_size

    def get_ground_truth_boxes(self, annotations):
        boxes_for_batch = []
        for b_i in range(self.batch_size):
            boxes_for_image = []
            for o_i in range(len(annotations)):
                bbox_coordinates = annotations[o_i]['bbox']
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
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        full_image_path = os.path.join(self.root, path)
        img = Image.open(full_image_path).convert('RGB')
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, full_image_path

    def __len__(self):
        return CocoDetection.__len__(self)
