import torch
import torch.nn as nn
from device import DEVICE
from terminaltables import AsciiTable


class YoloLoss():
    mse = nn.MSELoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def print(self):
        return AsciiTable([
            ["Localization", "Objectness", "No Objectness", "Classification", "Total Loss"],
            [self.localization_loss.item(), self.objectness_loss.item(), self.no_objectness_loss.item(),
             self.classification_loss.item(), self.total_loss.item()]

        ]).table

    def __init__(self,
                 coordinates,
                 confidence,
                 class_scores,
                 obj_mask,
                 noobj_mask,
                 cls_mask,
                 target_coordinates,
                 target_confidence,
                 target_class_scores,
                 lambda_coord=5,
                 lambda_no_obj=0.5):
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj

        self.coordinates = coordinates
        self.confidence = confidence
        self.class_scores = class_scores
        self.obj_mask = obj_mask
        self.noobj_mask = noobj_mask
        self.cls_mask = cls_mask
        self.target_coordinates = target_coordinates
        self.target_confidence = target_confidence
        self.target_class_scores = target_class_scores
        self.localization_loss = self.compute_localization_loss().to(DEVICE)
        self.objectness_loss = self.compute_objectness_loss().to(DEVICE)
        self.no_objectness_loss = self.compute_no_objectness_loss().to(DEVICE)
        self.classification_loss = self.compute_classification_loss().to(DEVICE)

        self.total_loss = self.lambda_coord * self.localization_loss + \
                          self.objectness_loss + \
                          self.lambda_no_obj * self.no_objectness_loss + \
                          self.classification_loss

    def get(self):
        return self.total_loss

    def compute_localization_loss(self):
        return self.mse(self.coordinates[self.obj_mask], self.target_coordinates[self.obj_mask])

    def compute_objectness_loss(self):
        return self.mse(self.confidence[self.obj_mask], self.target_confidence[self.obj_mask])

    def compute_no_objectness_loss(self):
        return self.mse(self.confidence[self.noobj_mask], self.target_confidence[self.noobj_mask])

    def compute_classification_loss(self):
        return self.bce_with_logits_loss(self.class_scores[self.cls_mask], self.target_class_scores[self.cls_mask])
