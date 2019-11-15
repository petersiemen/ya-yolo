import os

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
    def __init__(self, out_dir, class_names, iou_thresh, objectness_thresh, batch_size, plot):
        self.detection_results_dir = os.path.join(out_dir, "detection-results")
        self.ground_truth_dir = os.path.join(out_dir, "ground-truth")
        self.class_names = class_names
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

            image_path = image_paths[b_i]
            bb = boxes[0]
            bounding_box = {
                'x': bb[0], 'y': bb[1], 'w': bb[2], 'h': bb[3]
            }

            # self.car_dataset_writer.append(image_path=image_path, make=annotations[0]['make'][b_i],
            #                                model=annotations[0]['model'][b_i],
            #                                bounding_box=bounding_box)
