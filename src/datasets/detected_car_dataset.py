from shutil import copyfile
import torch
from datasets.yayolo_dataset import YaYoloDataset
from exif import load_image_file
from yolo.plotting import *
from yolo.utils import non_max_suppression
import os
from logging_config import *
import json
from device import DEVICE

logger = logging.getLogger(__name__)
HERE = os.path.dirname(os.path.realpath(__file__))


class DetectedCarDatasetHelper():
    def __init__(self, car_dataset_writer, class_names, conf_thres, nms_thres, batch_size, debug):
        self.car_dataset_writer = car_dataset_writer
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.debug = debug
        self.batch_size = batch_size

    def process_detections(self,
                           coordinates,
                           class_scores,
                           confidence,
                           images,
                           annotations,
                           image_paths
                           ):

        prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)
        detections = non_max_suppression(prediction=prediction,
                                         conf_thres=self.conf_thres,
                                         nms_thres=self.nms_thres
                                         )

        detected_one_car_for_batch = []
        for b_i in range(self.batch_size):
            image_path = image_paths[b_i]
            boxes = detections[b_i]
            detected_car = torch.zeros(len(boxes), dtype=torch.bool, device=DEVICE)
            for box_idx in range(len(boxes)):
                box = boxes[box_idx]
                # if class == 2 (car) and box-width >= 0.5 and box-height >= 0.4 and class_conf >= 0.8
                if box[6] == 2 and box[2] >= 0.5 and box[3] >= 0.3 and box[5] >= 0.8:
                    detected_car[box_idx] = True

            num_detected_cars = torch.sum(detected_car)
            if num_detected_cars == 1:
                bb = boxes[0]
                bounding_box = {
                    'x': bb[0].item(), 'y': bb[1].item(), 'w': bb[2].item(), 'h': bb[3].item()
                }

                copied_image_path = self.car_dataset_writer.copy_image(image_path)
                self.car_dataset_writer.append(image_path=copied_image_path,
                                               make=annotations[0]['make'][b_i],
                                               model=annotations[0]['model'][b_i],
                                               price=annotations[0]['price'][b_i].item(),
                                               date_of_first_registration=annotations[0]['date_of_first_registration'][
                                                   b_i].item(),
                                               bounding_box=bounding_box)
            elif num_detected_cars > 1:
                logger.info("Detected more than 1 car on the image. ({})".format(image_path))
            elif num_detected_cars == 0:
                logger.info("Detected no car on the image ({})".format(image_path))

            detected_one_car_for_batch.append(detected_car)

        if self.debug:
            car_detections = []
            for i in range(len(detections)):
                detected_car = detected_one_car_for_batch[i]
                if len(detected_car) > 0:
                    car_detections.append(detections[i][detected_car])
                else:
                    car_detections.append(detections[i])

            plot_batch(car_detections,
                       None,
                       images, self.class_names)

        return sum([torch.sum(d) for d in detected_one_car_for_batch])


class DetectedCarDatasetWriter():
    def __init__(self, file_writer):
        self.images_dir = os.path.join(os.path.dirname(file_writer.file_path), "images")
        assert len(os.listdir(self.images_dir)) == 0, f"{self.images_dir} is not empty"

        self.file_writer = file_writer
        logger.info('Init {}.'.format(self))

    def copy_image(self, image_path):
        basepath = os.path.basename(image_path)
        to_path = os.path.join(self.images_dir, basepath)
        copyfile(image_path, to_path)
        return "images/" + basepath

    def append(self, image_path, make, model, price, date_of_first_registration, bounding_box):
        self.file_writer.append(
            json.dumps({'image': image_path,
                        'make': make,
                        'model': model,
                        'price': price,
                        'date_of_first_registration': date_of_first_registration,
                        'bbox': [bounding_box['x'],
                                 bounding_box['y'],
                                 bounding_box['w'],
                                 bounding_box['h']]
                        })
        )

    def __repr__(self):
        return 'DetectedSimpleCarDatasetWriter({}, {})'.format(self.images_dir, self.file_writer.fd.name)


class DetectedCarDataset(YaYoloDataset):

    def __init__(self, json_file, transforms, batch_size, allow_unknown_make=False, allow_unknown_model=True):
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
        basedir = os.path.dirname(json_file)
        with open(json_file) as f:
            for line in f:
                obj = json.loads(line)
                if allow_unknown_make == False and obj['make'] == 'UNKNOWN':
                    continue
                if allow_unknown_model == False and obj['model'] == 'UNKNOWN':
                    continue

                image_path = os.path.join(basedir, obj['image'])
                if not os.path.exists(image_path):
                    logger.error('image {} does not exist'.format(image_path))
                    continue

                self.image_paths.append(image_path)
                self.annotations.append([obj])

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

    def _get_class_id(self, annotations, b_i):
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
                    category_id = self._get_class_id(annotations, b_i)
                    box = [x, y, w, h, 1, 1, category_id]
                    boxes_for_image.append(box)

            boxes_for_batch.append(boxes_for_image)

        ground_truth_boxes = torch.tensor(boxes_for_batch)
        return ground_truth_boxes


class DetectedCareMakeDataset(DetectedCarDataset):
    def __init__(self, json_file, transforms, batch_size):
        super(DetectedCareMakeDataset, self).__init__(json_file, transforms, batch_size)
        car_makes_file = os.path.join(os.path.dirname(json_file), "makes.csv")
        assert os.path.exists(car_makes_file), f"{car_makes_file} does not exist"

        with open(car_makes_file) as f:
            self._class_names = [make.strip() for make in f.readlines()]

    def get_num_classes(self):
        return len(self._class_names)

    def get_class_names(self):
        return self._class_names

    def _get_class_id(self, annotations, b_i):
        assert len(
            annotations) == 1, "DetectedCareMakeDataset can only operate on images labels with exactly one annoation. Found {}".format(
            len(annotations))
        make = annotations[0]['make'][b_i]
        return self._class_names.index(make)
