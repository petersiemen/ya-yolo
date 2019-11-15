import os
import glob
import torch
import json
from datasets.yayolo_custom_dataset import YaYoloCustomDataset
from exif import load_image_file
from yolo.utils import nms_for_coordinates_and_class_scores_and_confidence
from yolo.utils import plot_boxes
from logging_config import *
from torchvision.transforms import transforms
from torchvision.transforms import ToPILImage

logger = logging.getLogger(__name__)

to_pil_image = transforms.Compose([
    ToPILImage()
])


class DetectedCarDatasetHelper():
    def __init__(self, car_dataset_writer, class_names, iou_thresh, objectness_thresh, batch_size, plot):
        self.car_dataset_writer = car_dataset_writer
        self.class_names = class_names
        self.iou_thresh = iou_thresh
        self.objectness_thresh = objectness_thresh
        self.plot = plot
        self.batch_size = batch_size

    def process_detections(self,
                           coordinates,
                           class_scores,
                           confidence,
                           images,
                           annotations,
                           image_paths,
                           ground_truth_boxes
                           ):
        for b_i in range(self.batch_size):
            boxes = nms_for_coordinates_and_class_scores_and_confidence(
                coordinates[b_i],
                class_scores[b_i],
                confidence[b_i],
                self.iou_thresh,
                self.objectness_thresh)

            if self.plot:
                pil_image = to_pil_image(images[b_i])
                plot_boxes(pil_image, boxes, self.class_names, True)

            if len(boxes) == 1 and boxes[0][6] == 2:
                image_path = image_paths[b_i]
                bb = boxes[0]
                bounding_box = {
                    'x': bb[0], 'y': bb[1], 'w': bb[2], 'h': bb[3]
                }

                self.car_dataset_writer.append(image_path=image_path, make=annotations[0]['make'][b_i],
                                               model=annotations[0]['model'][b_i],
                                               bounding_box=bounding_box)


class DetectedCarDatasetWriter():
    def __init__(self, file_writer):
        self.file_writer = file_writer
        logger.info('Init {}.'.format(self))

    def append(self, image_path, make, model, bounding_box):
        self.file_writer.append(
            json.dumps({'image': image_path,
                        'make': make,
                        'model': model,
                        'bbox': [bounding_box['x'],
                                 bounding_box['y'],
                                 bounding_box['w'],
                                 bounding_box['h']]
                        })
        )

    def __repr__(self):
        return 'DetectedSimpleCarDatasetWriter({})'.format(self.file_writer.fd.name)


class DetectedCarDataset(YaYoloCustomDataset):

    def __init__(self, json_file, transforms, batch_size):
        """
        Args:
            root_dir (string): Directory with all the feeds and images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transforms = transforms
        self.batch_size = batch_size
        self.image_paths = []
        self.annotations = []
        with open(json_file) as f:
            for line in f:
                obj = json.loads(line)
                self.image_paths.append(obj['image'])
                self.annotations.append([obj])

        self.makes = tuple(set([car[0]['make'] for car in self.annotations]))
        self.int_to_make = dict(enumerate(self.makes))
        self.makes_to_int = {make: idx for idx, make in self.int_to_make.items()}

        self.model = tuple(set([car[0]['make'] + "__" + car[0]['model'] for car in self.annotations]))
        self.int_to_model = dict(enumerate(self.model))
        self.models_to_int = {model: idx for idx, model in self.int_to_model.items()}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        target = self.annotations[index]
        image_path = self.image_paths[index]
        image = load_image_file(image_path)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self):
        return len(self.annotations)

    def _get_category_id(self):
        return 1

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
                    # annotated_category_id = int(annotations[o_i]['category_id'][b_i].item())
                    # category_id = self.annotated_to_detected_class_idx[annotated_category_id]
                    category_id = self._get_category_id()
                    box = [x, y, w, h, 1, 1, category_id, 1]
                    boxes_for_image.append(box)

            boxes_for_batch.append(boxes_for_image)

        ground_truth_boxes = torch.tensor(boxes_for_batch)
        return ground_truth_boxes
