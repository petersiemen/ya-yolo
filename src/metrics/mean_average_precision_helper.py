from shutil import copyfile

import torch

from yolo.plotting import *
from yolo.utils import non_max_suppression, xyxy2xywh

logger = logging.getLogger(__name__)


class MeanAveragePrecisionHelper():
    def __init__(self, out_dir, class_names, image_size, get_ground_thruth_boxes, conf_thres, nms_thres,
                 batch_size, keep_images=False,
                 plot=False):
        self.detection_results_dir = os.path.join(out_dir, "detection-results")
        self.ground_truth_dir = os.path.join(out_dir, "ground-truth")
        self.images_optional_dir = os.path.join(out_dir, "images-optional")
        self.keep_images = keep_images

        self.class_names = class_names
        self.image_size = image_size
        self.get_ground_truth_boxes = get_ground_thruth_boxes
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

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

        prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)

        detections = non_max_suppression(prediction=prediction,
                                         conf_thres=self.conf_thres,
                                         nms_thres=self.nms_thres
                                         )

        plot_batch(detections, ground_truth_boxes, images, self.class_names)

        detected = 0
        for b_i in range(self.batch_size):
            boxes = detections[b_i]
            num_detected_objects = len(boxes)
            detected += num_detected_objects

            if num_detected_objects > 0:
                boxes[..., :4] = xyxy2xywh(boxes[..., :4])

            ground_truth = ground_truth_boxes[b_i]
            image_path = image_paths[b_i]
            self.write_detections_to_file(image_path, boxes)
            self.write_ground_truth_to_file(image_path, ground_truth)
            if self.keep_images:
                self.copy_image_file(image_path)

        return detected

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
                    x=box[0].item(),
                    y=box[1].item(),
                    w=box[2].item(),
                    h=box[3].item()
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
