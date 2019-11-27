from collections import defaultdict

import numpy as np

from metrics.classify import Classify
from metrics.precision import Precision
from metrics.recall import Recall


class Metrics:

    def __init__(self):
        self._ground_truth_counter_per_class = defaultdict(lambda: 0)
        self._detections = []
        self._unique_class_ids = set()

    def _update_ground_truth_counter_per_class(self, ground_truths):
        for ground_truth in ground_truths:
            self._ground_truth_counter_per_class[ground_truth.class_id] += 1
            self._unique_class_ids.add(ground_truth.class_id)

    def add_detections_for_batch(self, detections, ground_truths, iou_thres):
        file_ids = set([d.file_id for d in detections])
        for file_id in file_ids:
            detections_for_image = list(filter(lambda x: x.file_id == file_id, detections))
            ground_truths_for_image = list(filter(lambda x: x.file_id == file_id, ground_truths))
            self.add_detections_for_image(detections_for_image, ground_truths_for_image, iou_thres)

    def add_detections_for_image(self, detections, ground_truths, iou_thres):
        """
        :param detections: list of Detection objects
        :param ground_truths: list of GroundTruth objects
        :return:
        """
        self._update_ground_truth_counter_per_class(ground_truths)
        ground_truths = np.array(ground_truths)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        for detection in detections:
            classification, ground_truths = Classify.classify(detection, ground_truths, iou_thres)
            detection.set_classification(classification)

        self._detections += detections

    def compute_accuracy(self):

        return 0

    def compute_average_precision_for_classes(self):
        """
        computes mAP and returns a dictionary containing all classes mAP scores

        :return:
        """
        average_precision_for_classes = {}

        for class_id in self._unique_class_ids:
            detections_for_class = list(filter(lambda d: d.class_id == class_id, self._detections))
            number_of_ground_truth_objects_for_class = self._ground_truth_counter_per_class[class_id]

            detections_for_class.sort(key=lambda d: d.confidence, reverse=True)

            precision = Precision.compute(detections_for_class)
            recall = Recall.compute(detections_for_class, number_of_ground_truth_objects_for_class)
            average_precision_for_class = Metrics.compute_average_precision(recall=recall, precision=precision)

            average_precision_for_classes[f"{class_id}"] = average_precision_for_class

        mAP = np.mean(list(average_precision_for_classes.values()))

        return average_precision_for_classes, mAP

    @staticmethod
    def compute_average_precision(recall, precision):
        """ Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.

        # Arguments
            recall:    The recall curve (list).
            precision: The precision curve (list).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
