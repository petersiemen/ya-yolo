import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from yolo.utils import plot_boxes

from device import DEVICE
from yolo.utils import boxes_iou

to_pil_image = transforms.Compose([
    ToPILImage()
])


def get_indices_for_center_of_bounding_boxes(num_anchors, grid_widths, x, y):
    for g_i in range(len(grid_widths)):
        grid_offset = sum([gs * gs * num_anchors for gs in grid_widths[0:g_i]])
        grid_width = grid_widths[g_i]
        idx_x = int(x * grid_width)
        idx_y = int(y * grid_width)

        for a_i in range(0, num_anchors):
            idx = grid_offset
            anchor_offset = a_i * grid_width * grid_width
            idx += idx_y * grid_width + idx_x + anchor_offset
            yield idx


def _get_indices_for_center_of_ground_truth_bounding_boxes(ground_truth_boxes, grid_sizes):
    indices_for_batch = []
    for batch_i in range(ground_truth_boxes.shape[0]):
        indices_for_image = []
        for box_j in range(ground_truth_boxes.shape[1]):
            x = ground_truth_boxes[batch_i, box_j, 0]
            y = ground_truth_boxes[batch_i, box_j, 1]
            indices = list(get_indices_for_center_of_bounding_boxes(num_anchors=3,
                                                                    grid_widths=grid_sizes,
                                                                    x=x,
                                                                    y=y))
            indices_for_image.append(indices)
        indices_for_batch.append(indices_for_image)
    return torch.tensor(indices_for_batch)


def _get_indices_for_highest_iou_with_ground_truth_bounding_box(indices, ground_truth_boxes, coordinates):
    indices_for_batch = []
    for batch_i in range(ground_truth_boxes.shape[0]):
        indices_for_image = []
        for box_j in range(ground_truth_boxes.shape[1]):
            candidate_coordinates = coordinates[batch_i, indices[batch_i, box_j]]
            ious = [boxes_iou(ground_truth_boxes[batch_i, box_j], candidate_box) for candidate_box in
                    candidate_coordinates]
            max_iou_idx = np.argmax(ious)
            indices_for_image.append(indices[batch_i, box_j, max_iou_idx])
        indices_for_batch.append(indices_for_image)
    return torch.tensor(indices_for_batch)


def _get_box_and_confidence_with_highest_iou_with_ground_truth_bounding_box(indices,
                                                                            ground_truth_boxes,
                                                                            coordinates,
                                                                            confidence):
    for b_i in range(len(indices)):
        candidate_coordinates = coordinates[b_i][indices[b_i]]
        candidate_confidences = confidence[b_i][indices[b_i]]
        ious = [boxes_iou(ground_truth_boxes[b_i],
                          candidate_box).item() for candidate_box in candidate_coordinates]
        max_iou_idx = np.argmax(ious)
        yield candidate_coordinates[max_iou_idx], candidate_confidences[max_iou_idx]


def _select_boxes(coordinates, indices):
    batch_size = indices.shape[0]
    boxes_per_image_in_batch = indices.shape[1]
    boxes_for_batch = torch.zeros(batch_size, boxes_per_image_in_batch, 4).type_as(coordinates)
    for image_i in range(batch_size):
        for box_j in range(boxes_per_image_in_batch):
            boxes_for_batch[image_i, box_j] = coordinates[image_i, indices[image_i, box_j]]
    return boxes_for_batch


def _select_class_scores(class_scores, indices):
    batch_size = indices.shape[0]
    number_of_groundtruth_objects_per_image = indices.shape[1]
    number_of_groundtruth_bounding_boxes_per_object_and_image = indices.shape[2]
    number_of_classes = class_scores.shape[2]
    classes_scores_for_batch = torch.zeros(batch_size,
                                           number_of_groundtruth_objects_per_image,
                                           number_of_groundtruth_bounding_boxes_per_object_and_image,
                                           number_of_classes).type_as(class_scores)
    for image_i in range(batch_size):
        for object_i in range(number_of_groundtruth_objects_per_image):
            classes_scores_for_batch[image_i, object_i] = class_scores[image_i, indices[image_i, object_i]]
    return classes_scores_for_batch


def _select_confidence(confidence, indices):
    batch_size = indices.shape[0]
    boxes_per_image_in_batch = indices.shape[1]
    confidence_for_batch = torch.zeros(batch_size, boxes_per_image_in_batch, 1).type_as(confidence)
    for image_i in range(batch_size):
        for box_j in range(boxes_per_image_in_batch):
            confidence_for_batch[image_i, box_j] = confidence[image_i, indices[image_i, box_j]]
    return confidence_for_batch


def _negative_select_confidence(confidence, indices):
    batch_size = indices.shape[0]
    boxes_per_image_in_batch = indices.shape[1]
    number_of_gridcells = confidence.shape[1]
    number_of_neg_confidences = number_of_gridcells - boxes_per_image_in_batch
    neg_confidence_for_batch = torch.zeros(batch_size, number_of_neg_confidences).type_as(confidence)
    for image_i in range(batch_size):
        skipped = indices[image_i]
        idx = [i for i in range(len(confidence[image_i])) if i not in skipped]
        neg_confidence_for_batch[image_i] = confidence[image_i, idx]
    return neg_confidence_for_batch


class YoloLoss():
    mse = nn.MSELoss()
    bce_with_logits_loss = nn.BCEWithLogitsLoss()

    def __init__(self, lambda_coord=5, lambda_no_obj=0.5):
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj

    def loss(self, detected, confidence, no_object_confidence, class_scores, ground_truth):
        return self._localization_loss(detected, ground_truth) + \
               self._objectness_loss(confidence) + \
               self._no_objectness_loss(no_object_confidence) + \
               self._classification_loss(class_scores, ground_truth)

    def _localization_loss(self, detected, ground_truth_boxes):
        x = ground_truth_boxes[:, :, 0]
        y = ground_truth_boxes[:, :, 1]
        w = ground_truth_boxes[:, :, 2]
        h = ground_truth_boxes[:, :, 3]

        _x = detected[:, :, 0].type_as(x)
        _y = detected[:, :, 1].type_as(y)
        _w = detected[:, :, 2].type_as(w)
        _h = detected[:, :, 3].type_as(h)

        loss = self.lambda_coord * (self.mse(input=_x, target=x)
                                    + self.mse(input=_y, target=y)) + self.lambda_coord * (
                       self.mse(input=torch.sqrt(_h), target=torch.sqrt(h))
                       + self.mse(input=torch.sqrt(_w), target=torch.sqrt(w)))

        return loss

    def _objectness_loss(self, confidence):
        """
        Confidence score of whether there is an object in the grid_cell or not.
        When there is an object, we want the score equals to IOU, and when there is no object we want the score to be zero.
        :return:
        """

        loss = self.mse(input=confidence,
                        target=torch.ones(confidence.shape).type_as(confidence).to(device=DEVICE))
        return loss

    def _no_objectness_loss(self, no_object_confidence):
        loss = self.lambda_no_obj * self.mse(input=no_object_confidence,
                                             target=torch.zeros(no_object_confidence.shape).type_as(
                                                 no_object_confidence).to(
                                                 device=DEVICE))
        return loss

    def _classification_loss(self, class_scores, ground_truth):

        ground_truth_class_scores = torch.zeros(class_scores.shape).type_as(class_scores)
        batch_size = ground_truth.shape[0]
        ground_truth_objects_per_image = ground_truth.shape[1]
        ground_truth_bounding_boxes_per_object_and_image = ground_truth.shape[2]
        for b_i in range(batch_size):
            for o_i in range(ground_truth_objects_per_image):
                for g_i in range(ground_truth_bounding_boxes_per_object_and_image):
                    class_idx = int(ground_truth[b_i, o_i, 6].item())
                    ground_truth_class_scores[b_i, o_i, g_i, class_idx] = 1

        return self.bce_with_logits_loss(input=class_scores, target=ground_truth_class_scores)


def plot(ground_truth_boxes, images, classnames, plot_labels):
    batch_size = len(ground_truth_boxes)
    for b_i in range(batch_size):
        pil_image = to_pil_image(images[b_i].cpu())
        boxes = ground_truth_boxes[b_i].cpu()

        plot_boxes(pil_image, boxes, classnames, plot_labels)


def _to_plottable_boxes(boxes, batch_indices_with_highest_iou, class_scores):
    batch_size = boxes.shape[0]
    objects_per_image = boxes.shape[1]
    detached_boxes = boxes.detach().cpu().numpy().tolist()
    boxes_for_batch = []
    for b_i in range(batch_size):
        boxes_for_image = []
        for o_i in range(objects_per_image):
            cls_max_conf, cls_max_id = torch.max(class_scores[b_i, batch_indices_with_highest_iou[b_i, o_i]], 0)

            boxes_for_image.append(detached_boxes[b_i][o_i] + [1, cls_max_conf.item(), cls_max_id.item()])
        boxes_for_batch.append(boxes_for_image)
    return boxes_for_batch


def training(model, ya_yolo_dataset, model_dir, num_epochs=1, lr=0.001, limit=None, debug=False, print_every=10):
    print('Number of images: ', len(ya_yolo_dataset.dataset))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grid_sizes = model.grid_sizes
    yolo_loss = YoloLoss()

    data_loader = ya_yolo_dataset.get_dataloader()
    batch_size = ya_yolo_dataset.batch_size
    classnames = ya_yolo_dataset.get_classnames()

    for epoch in range(1, num_epochs + 1):

        running_loss = 0.0

        for batch_i, (images, annotations) in enumerate(data_loader):
            images = images.to(DEVICE)
            if images.shape[0] != batch_size:
                print('Skipping batch {} because batch-size {} is not as expected {}'.format(batch_i, len(images),
                                                                                             batch_size))
                continue

            ground_truth_boxes = ya_yolo_dataset.get_ground_truth_boxes(annotations).to(DEVICE)
            if debug:
                plot(ground_truth_boxes, images, classnames, True)

            model.train()
            coordinates, class_scores, confidence = model(images)

            batch_indices_of_ground_truth_boxes = _get_indices_for_center_of_ground_truth_bounding_boxes(
                ground_truth_boxes, grid_sizes)
            batch_indices_with_highest_iou = _get_indices_for_highest_iou_with_ground_truth_bounding_box(
                batch_indices_of_ground_truth_boxes, ground_truth_boxes, coordinates
            )

            boxes_with_highest_iou = _select_boxes(coordinates, batch_indices_with_highest_iou)
            confidence_with_highest_iou = _select_confidence(confidence, batch_indices_with_highest_iou)

            no_object_confidence = _negative_select_confidence(confidence, batch_indices_with_highest_iou)
            class_scores_for_ground_truth_boxes = _select_class_scores(class_scores,
                                                                       batch_indices_of_ground_truth_boxes)

            if debug:
                plot(_to_plottable_boxes(boxes_with_highest_iou,
                                         batch_indices_with_highest_iou,
                                         class_scores), images, classnames, True)

            loss = yolo_loss.loss(boxes_with_highest_iou,
                                  confidence_with_highest_iou,
                                  no_object_confidence,
                                  class_scores_for_ground_truth_boxes,
                                  ground_truth_boxes)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()

            if batch_i % print_every == 0:  # print every print_every +1  batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch, batch_i + 1, running_loss / 1000))
                running_loss = 0.0

            if limit is not None and batch_i >= limit:
                print('Stop here after training {} batches (limit: {})'.format(batch_i, limit))
                return

        model.save(model_dir,
                   'yolo__num_classes_{}__epoch_{}.pt'.format(model.num_classes, epoch))
