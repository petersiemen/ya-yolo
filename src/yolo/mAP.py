import os
import math
from shutil import copyfile

from torchvision.transforms import ToPILImage
from torchvision.transforms import transforms

from logging_config import *
from yolo.utils import nms_for_coordinates_and_class_scores_and_confidence
from yolo.utils import plot_boxes

logger = logging.getLogger(__name__)

to_pil_image = transforms.Compose([
    ToPILImage()
])


class MeanAveragePrecisionHelper():
    def __init__(self, out_dir, class_names, image_size, get_ground_thruth_boxes, iou_thresh, objectness_thresh, batch_size, keep_images=False,
                 plot=False):
        self.detection_results_dir = os.path.join(out_dir, "detection-results")
        self.ground_truth_dir = os.path.join(out_dir, "ground-truth")
        self.images_optional_dir = os.path.join(out_dir, "images-optional")
        self.keep_images = keep_images

        self.class_names = class_names
        self.image_size = image_size
        self.get_ground_truth_boxes = get_ground_thruth_boxes
        self.iou_thresh = iou_thresh
        self.objectness_thresh = objectness_thresh
        self.batch_size = batch_size
        self.plot = plot


    def process_detections(self,
                           coordinates,
                           class_scores,
                           confidence,
                           images,
                           annotations,
                           image_paths
                           ):

        ground_truth_boxes = self.get_ground_truth_boxes(annotations)

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

            ground_truth = ground_truth_boxes[b_i]
            image_path = image_paths[b_i]
            self.write_detections_to_file(image_path, boxes)
            self.write_ground_truth_to_file(image_path, ground_truth)
            if self.keep_images:
                self.copy_image_file(image_path)

    def _remove_extension(self, filename):
        return filename.rsplit('.', 1)[0]

    def copy_image_file(self, image_path):
        copyfile(image_path, os.path.join(self.images_optional_dir,
                                          os.path.basename(image_path)))

    def write_detections_to_file(self, image_path, detected_boxes):
        with open(os.path.join(self.detection_results_dir,
                               self._remove_extension(os.path.basename(image_path)) + '.txt'), 'w') as f:
            num_boxes = len(detected_boxes)
            writable_boxes = []
            for b_i in range(num_boxes):
                box = detected_boxes[b_i]
                left, top, right, bottom = self.convert_coodinates(
                    x=box[0],
                    y=box[1],
                    w=box[2],
                    h=box[3]
                )
                category = self.get_class(int(box[6]))
                conf = box[4]
                writable_boxes.append([category, conf, left, top, right, bottom])

            writable_boxes.sort(key=lambda x: x[1], reverse=True)

            for box in writable_boxes:
                f.write(" ".join([str(el) for el in box]) + '\n')

    def write_ground_truth_to_file(self, image_path, ground_truth_boxes_for_image):
        with open(os.path.join(self.ground_truth_dir,
                               self._remove_extension(os.path.basename(image_path)) + '.txt'), 'w') as f:
            num_boxes = ground_truth_boxes_for_image.shape[0]

            for b_i in range(num_boxes):
                box = ground_truth_boxes_for_image[b_i]
                left, top, right, bottom = self.convert_coodinates(
                    x=box[0].item(),
                    y=box[1].item(),
                    w=box[2].item(),
                    h=box[3].item()
                )
                category = self.get_class(int(box[6]))
                f.write('{} {} {} {} {}\n'.format(category, left, top, right, bottom))

    def convert_coodinates(self, x, y, w, h):
        left = int(round((x - w / 2) * self.image_size))
        top = int(round((y - h / 2) * self.image_size))
        right = int(round((x + w / 2) * self.image_size))
        bottom = int(round((y + h / 2) * self.image_size))
        return left, top, right, bottom

    def get_class(self, category_id):
        return self.class_names[category_id]
