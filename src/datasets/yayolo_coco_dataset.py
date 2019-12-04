import os

import torch
from torchvision.datasets import CocoDetection
from PIL import Image

from datasets.yayolo_dataset import YaYoloDataset


class YaYoloCocoDataset(YaYoloDataset, CocoDetection):

    def __init__(self, images_dir, annotations_file, transforms, batch_size):
        CocoDetection.__init__(self, root=images_dir, annFile=annotations_file,
                               transforms=transforms)
        self._annotated_classnames = {k: v['name'] for k, v in self.coco.cats.items()}
        self._annotated_to_detected_class_idx = {k: i for i, (k, v)
                                                 in enumerate(self.coco.cats.items())}
        self.batch_size = batch_size
        self.class_names = [v['name'] for k, v in self.coco.cats.items()]

    def _get_class_id(self, category_id):
        return self._annotated_to_detected_class_idx[category_id]

    def _get_ground_truth_box(self, bbox, width, height, category_id):
        upper_left_x = bbox[0]
        upper_left_y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        x = w / 2 + upper_left_x
        y = h / 2 + upper_left_y
        class_id = self._get_class_id(category_id)
        return torch.tensor([x / width, y / height, w / width, h / height, 1, 1, class_id], dtype=torch.float)

    def _annotation_to_ground_truth_boxes(self, annotation):
        ground_truth_boxes = []
        for i in range(len(annotation)):
            width, height = annotation[i]['image']['width'], annotation[i]['image']['height']
            bbox = annotation[i]['bbox']
            category_id = annotation[i]['category_id']
            ground_truth_boxes.append(self._get_ground_truth_box(bbox, width, height, category_id))

        return ground_truth_boxes

    def collate_fn(self, batch):
        images = torch.stack([item[0] for item in batch])
        target = [self._annotation_to_ground_truth_boxes(item[1]) for item in batch]
        image_paths = [item[2] for item in batch]
        return images, target, image_paths

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

        width, height = img.size
        self._orig_width_and_height_into_annotation(target, width, height)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, full_image_path

    def __len__(self):
        return CocoDetection.__len__(self)

    def _orig_width_and_height_into_annotation(self, target, width, height):
        for i in range(len(target)):
            target[i]['image'] = {'width': width, 'height': height}
        return target
