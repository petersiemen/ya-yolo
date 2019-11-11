import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

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


def _get_indices_for_center_of_ground_truth_bounding_boxes(x, y, grid_sizes):
    for b_i in range(len(x)):
        indices = list(get_indices_for_center_of_bounding_boxes(num_anchors=3,
                                                                grid_widths=grid_sizes,
                                                                x=x[b_i].item(),
                                                                y=y[b_i].item()))
        yield indices


def _get_indices_for_highest_iou_with_ground_truth_bounding_box(indices, ground_truth_boxes, coordinates):
    for b_i in range(len(indices)):
        candidate_coordinates = coordinates[b_i][indices[b_i]]
        ious = [boxes_iou(ground_truth_boxes[b_i],
                          candidate_box).item() for candidate_box in candidate_coordinates]
        max_iou_idx = np.argmax(ious)
        yield indices[b_i][max_iou_idx]


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
    for b_i in range(len(indices)):
        yield coordinates[b_i][indices[b_i]]


def _select_confidence(confidence, indices):
    for b_i in range(len(indices)):
        yield confidence[b_i][indices[b_i]]


def _negative_select_confidence(confidence, indices):
    for b_i in range(len(indices)):
        skipped = indices[b_i]
        idx = [i for i in range(len(confidence[b_i])) if i != skipped]
        yield confidence[b_i][idx].unsqueeze(0)


class YoloLoss():
    mse = nn.MSELoss()

    def __init__(self, lambda_coord=5, lambda_no_obj=0.5):
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj

    def loss(self, detected, confidence, no_object_confidence, ground_truth):
        return self._localization_loss(detected, ground_truth) + \
               self._objectness_loss(confidence) + \
               self._no_objectness_loss(no_object_confidence)

    def _localization_loss(self, detected, ground_truth_boxes):
        x = ground_truth_boxes[:, 0]
        y = ground_truth_boxes[:, 1]
        w = ground_truth_boxes[:, 2]
        h = ground_truth_boxes[:, 3]

        _x = detected[:, 0]
        _y = detected[:, 1]
        _w = detected[:, 2]
        _h = detected[:, 3]

        loss = self.lambda_coord * (self.mse(_x, x) + self.mse(_y, y)) + self.lambda_coord * (
                self.mse(torch.sqrt(_h), torch.sqrt(h)) + self.mse(torch.sqrt(_w), torch.sqrt(w)))

        return loss

    def _objectness_loss(self, confidence):
        """
        Confidence score of whether there is an object in the grid_cell or not.
        When there is an object, we want the score equals to IOU, and when there is no object we want the score to be zero.
        :return:
        """

        loss = self.mse(confidence, torch.ones(confidence.shape).type_as(confidence).to(device=DEVICE))
        return loss

    def _no_objectness_loss(self, no_object_confidence):
        loss = self.lambda_no_obj * self.mse(no_object_confidence,
                                             torch.zeros(no_object_confidence.shape).type_as(no_object_confidence).to(
                                                 device=DEVICE))
        return loss


def training(model, dataset, num_epochs=1, batch_size=2):
    print('Number of car images: ', len(dataset))

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grid_sizes = model.grid_sizes
    yolo_loss = YoloLoss()

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, num_epochs + 1):

        running_loss = 0.0

        for batch_i, (images, annotations) in enumerate(data_loader):
            images = images.to(DEVICE)
            if images.shape[0] != batch_size:
                print('Skipping batch {} because batch-size {} is not as expected {}'.format(batch_i, len(images),
                                                                                             batch_size))
                continue

            pil_image = to_pil_image(images[0])
            plt.imshow(pil_image)
            plt.show()

            number_of_annotated_objects = annotations[0]['bbox']

            x = annotations['bbox'][0].to(DEVICE)
            y = annotations['bbox'][1].to(DEVICE)
            w = annotations['bbox'][2].to(DEVICE)
            h = annotations['bbox'][3].to(DEVICE)

            ground_truth_boxes = torch.tensor(
                [[x[i].item(), y[i].item(), w[i].item(), h[i].item()] for i in range(len(x))])

            model.train()
            coordinates, class_scores, confidence = model(images)

            batch_indices = list(_get_indices_for_center_of_ground_truth_bounding_boxes(x, y, grid_sizes))
            batch_indices_with_highest_iou = list(_get_indices_for_highest_iou_with_ground_truth_bounding_box(
                batch_indices, ground_truth_boxes, coordinates
            ))
            boxes_with_highest_iou = torch.cat(
                [box.unsqueeze(0) for box in list(_select_boxes(coordinates, batch_indices_with_highest_iou))], 0)
            confidence_with_highest_iou = torch.cat([conf.unsqueeze(0) for conf in torch.tensor(
                list(_select_confidence(confidence, batch_indices_with_highest_iou)))], 0)

            no_object_confidence = torch.cat(
                list(_negative_select_confidence(confidence, batch_indices_with_highest_iou)), 0)

            loss = yolo_loss.loss(boxes_with_highest_iou,
                                  confidence_with_highest_iou,
                                  no_object_confidence,
                                  ground_truth_boxes,
                                  )

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()

            print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
            if batch_i % 10 == 9:  # print every 10 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0
