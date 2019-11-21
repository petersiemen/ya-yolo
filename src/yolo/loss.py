import torch
import torch.nn as nn
from device import DEVICE
from terminaltables import AsciiTable


#
# class YoloLoss():
#     mse = nn.MSELoss()
#     bce_with_logits_loss = nn.BCEWithLogitsLoss()
#
#     def __init__(self, lambda_coord=5, lambda_no_obj=0.5):
#         self.lambda_coord = lambda_coord
#         self.lambda_no_obj = lambda_no_obj
#
#     def loss(self, detected, confidence, no_object_confidences, class_scores, ground_truth):
#         return self._localization_loss(detected, ground_truth), \
#                self._objectness_loss(confidence), \
#                self._no_objectness_loss(no_object_confidences), \
#                self._classification_loss(class_scores, ground_truth), \
#                self._localization_loss(detected, ground_truth) + \
#                self._objectness_loss(confidence) + \
#                self._no_objectness_loss(no_object_confidences) + \
#                self._classification_loss(class_scores, ground_truth)
#
#     def _localization_loss(self, detected, ground_truth_boxes):
#
#         if ground_truth_boxes.shape[1] != 0:
#             x = ground_truth_boxes[:, :, 0]
#             y = ground_truth_boxes[:, :, 1]
#             w = ground_truth_boxes[:, :, 2]
#             h = ground_truth_boxes[:, :, 3]
#
#             _x = detected[:, :, 0].type_as(x)
#             _y = detected[:, :, 1].type_as(y)
#             _w = detected[:, :, 2].type_as(w)
#             _h = detected[:, :, 3].type_as(h)
#
#             loss = self.lambda_coord * (self.mse(input=_x, target=x)
#                                         + self.mse(input=_y, target=y)) + self.lambda_coord * (
#                            self.mse(input=torch.sqrt(_h), target=torch.sqrt(h))
#                            + self.mse(input=torch.sqrt(_w), target=torch.sqrt(w)))
#             return loss
#         else:
#             return 0.0
#
#     def _objectness_loss(self, confidence):
#         """
#         Confidence score of whether there is an object in the grid_cell or not.
#         When there is an object, we want the score equals to IOU, and when there is no object we want the score to be zero.
#         :return:
#         """
#         if confidence.shape[1] > 0 and confidence.shape[2] > 0:
#             loss = self.mse(input=confidence,
#                             target=torch.ones(confidence.shape).type_as(confidence).to(device=DEVICE))
#             return loss
#         else:
#             return 0.0
#
#     def _no_objectness_loss(self, no_object_confidences):
#         loss = 0.0
#         for no_object_confidence in no_object_confidences:
#             loss += self.lambda_no_obj * self.mse(input=no_object_confidence,
#                                                   target=torch.zeros(no_object_confidence.shape).type_as(
#                                                       no_object_confidence).to(
#                                                       device=DEVICE))
#         return loss
#
#     def _classification_loss(self, class_scores, ground_truth):
#
#         if class_scores.shape[1] != 0:
#             ground_truth_class_scores = torch.zeros(class_scores.shape).type_as(class_scores)
#             batch_size = ground_truth.shape[0]
#             ground_truth_objects_per_image = ground_truth.shape[1]
#             ground_truth_bounding_boxes_per_object_and_image = ground_truth.shape[2]
#             for b_i in range(batch_size):
#                 for o_i in range(ground_truth_objects_per_image):
#                     for g_i in range(ground_truth_bounding_boxes_per_object_and_image):
#                         class_idx = int(ground_truth[b_i, o_i, 6].item())
#                         ground_truth_class_scores[b_i, o_i, g_i, class_idx] = 1
#
#             return self.bce_with_logits_loss(input=class_scores, target=ground_truth_class_scores)
#         else:
#             return 0.0


class YoloLoss():
    mse = nn.MSELoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def print(self):
        return AsciiTable([
            ["Localization", "Objectness", "No Objectness", "Classification"],
            [self.localization_loss.item(), self.objectness_loss.item(), self.no_objectness_loss.item(),
             self.classification_loss.item()]

        ]).table

    def __init__(self,
                 detected,
                 confidence,
                 no_object_confidences,
                 class_scores,
                 ground_truth_boxes,
                 lambda_coord=5, lambda_no_obj=0.5):
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        self.detected = detected
        self.confidence = confidence
        self.no_object_confidences = no_object_confidences
        self.ground_truth_boxes = ground_truth_boxes
        self.class_scores = class_scores

        self.localization_loss = self.compute_localization_loss()
        self.objectness_loss = self.compute_objectness_loss()
        self.no_objectness_loss = self.compute_no_objectness_loss()
        self.classification_loss = self.compute_classification_loss()
        self.total_loss = self.lambda_coord * self.localization_loss + \
                          self.no_objectness_loss + \
                          self.lambda_no_obj * self.no_objectness_loss + \
                          self.classification_loss

    def get(self):
        return self.total_loss

    def compute_localization_loss(self):

        if self.ground_truth_boxes.shape[1] != 0:
            x = self.ground_truth_boxes[:, :, 0]
            y = self.ground_truth_boxes[:, :, 1]
            w = self.ground_truth_boxes[:, :, 2]
            h = self.ground_truth_boxes[:, :, 3]

            _x = self.detected[:, :, 0].type_as(x)
            _y = self.detected[:, :, 1].type_as(y)
            _w = self.detected[:, :, 2].type_as(w)
            _h = self.detected[:, :, 3].type_as(h)

            loss = self.mse(input=_x, target=x) + \
                   self.mse(input=_y, target=y) + \
                   self.mse(input=torch.sqrt(_h), target=torch.sqrt(h)) + \
                   self.mse(input=torch.sqrt(_w), target=torch.sqrt(w))
            return loss
        else:
            return torch.tensor(0.0)

    def compute_objectness_loss(self):
        """
        Confidence score of whether there is an object in the grid_cell or not.
        When there is an object, we want the score equals to IOU, and when there is no object we want the score to be zero.
        :return:
        """
        if self.confidence.shape[1] > 0 and self.confidence.shape[2] > 0:
            loss = self.mse(input=self.confidence,
                            target=torch.ones(self.confidence.shape).type_as(self.confidence).to(device=DEVICE))
            return loss
        else:
            return torch.tensor(0.0)

    def compute_no_objectness_loss(self):
        loss = torch.tensor(0.0)
        for no_object_confidence in self.no_object_confidences:
            loss += self.mse(input=no_object_confidence,
                             target=torch.zeros(no_object_confidence.shape).type_as(
                                 no_object_confidence).to(
                                 device=DEVICE))
        return loss

    def compute_classification_loss(self):

        if self.class_scores.shape[1] != 0:
            ground_truth_class_scores = torch.zeros(self.class_scores.shape).type_as(self.class_scores)
            batch_size = self.ground_truth_boxes.shape[0]
            ground_truth_objects_per_image = self.ground_truth_boxes.shape[1]
            ground_truth_bounding_boxes_per_object_and_image = self.ground_truth_boxes.shape[2]
            for b_i in range(batch_size):
                for o_i in range(ground_truth_objects_per_image):
                    for g_i in range(ground_truth_bounding_boxes_per_object_and_image):
                        class_idx = int(self.ground_truth_boxes[b_i, o_i, 6].item())
                        ground_truth_class_scores[b_i, o_i, g_i, class_idx] = 1

            return self.bce_with_logits_loss(input=self.class_scores, target=ground_truth_class_scores)
        else:
            return torch.tensor(0.0)
