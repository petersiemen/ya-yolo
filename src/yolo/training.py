import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToPILImage

from device import DEVICE
from yolo.loss import YoloLoss
from yolo.utils import boxes_iou_for_single_boxes
from yolo.utils import plot_boxes

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


def get_indices_for_center_of_ground_truth_bounding_boxes(ground_truth_boxes, grid_sizes):
    batch_size = ground_truth_boxes.shape[0]
    num_of_boxes_in_image_in_batch = ground_truth_boxes.shape[1]
    indices_for_batch = []
    for image_i in range(batch_size):
        indices_for_image = []
        for box_j in range(num_of_boxes_in_image_in_batch):
            x = ground_truth_boxes[image_i, box_j, 0]
            y = ground_truth_boxes[image_i, box_j, 1]
            indices = list(get_indices_for_center_of_bounding_boxes(num_anchors=3,
                                                                    grid_widths=grid_sizes,
                                                                    x=x,
                                                                    y=y))
            indices_for_image.append(indices)
        indices_for_batch.append(indices_for_image)
    return torch.tensor(indices_for_batch).to(DEVICE)


def get_indices_for_highest_iou_with_ground_truth_bounding_box(indices, ground_truth_boxes, coordinates):
    batch_size = ground_truth_boxes.shape[0]
    num_of_boxes_in_image_in_batch = ground_truth_boxes.shape[1]
    indices_for_batch = []
    for image_i in range(batch_size):
        indices_for_image = []
        for box_j in range(num_of_boxes_in_image_in_batch):
            candidate_coordinates = coordinates[image_i, indices[image_i, box_j]]
            ious = [boxes_iou_for_single_boxes(ground_truth_boxes[image_i, box_j], candidate_box) for candidate_box in
                    candidate_coordinates]
            max_iou_idx = np.argmax(ious)
            indices_for_image.append(indices[image_i, box_j, max_iou_idx])
        indices_for_batch.append(indices_for_image)
    return torch.tensor(indices_for_batch).to(DEVICE)


def plot(ground_truth_boxes, images, classnames, plot_labels):
    batch_size = len(ground_truth_boxes)
    for b_i in range(batch_size):
        pil_image = to_pil_image(images[b_i].cpu())
        boxes = ground_truth_boxes[b_i]

        plot_boxes(pil_image, boxes, classnames, plot_labels)


def to_plottable_boxes(obj_mask, coordinates, class_scores, confidence):
    batch_size = coordinates.size(0)
    filtered_coordinates = coordinates[obj_mask]
    filtered_confidences = confidence[obj_mask]
    filtered_class_scores = class_scores[obj_mask]
    boxes = []
    for i in range(filtered_coordinates.size(0)):
        det_conf = filtered_confidences[i].detach().cpu()
        class_score, class_idx = torch.max(filtered_class_scores[i], 0)

        boxes.append(
            torch.cat((filtered_coordinates[i].detach().cpu(), torch.tensor([det_conf, class_score, class_idx])),
                      0))

    if len(boxes) > 0:
        return torch.cat(boxes).view(batch_size, -1, 7)
    else:
        return torch.tensor([[] for _ in range(batch_size)])


def build_targets(coordinates, class_scores, ground_truth_boxes, grid_sizes):
    batch_size = coordinates.size(0)
    num_anchors = 3
    num_classes = class_scores.size(-1)
    total_num_of_grid_cells = sum([gs * gs * num_anchors for gs in grid_sizes])

    obj_mask = torch.zeros(size=(batch_size, total_num_of_grid_cells), dtype=torch.bool, device=DEVICE)
    noobj_mask = torch.ones(size=(batch_size, total_num_of_grid_cells), dtype=torch.bool, device=DEVICE)
    cls_mask = torch.zeros(size=(batch_size, total_num_of_grid_cells), dtype=torch.bool, device=DEVICE)
    target_coordinates = torch.zeros(size=(batch_size, total_num_of_grid_cells, 4), dtype=torch.float32, device=DEVICE)
    target_class_scores = torch.zeros(size=(batch_size, total_num_of_grid_cells, num_classes), dtype=torch.float32,
                                      device=DEVICE)
    target_confidence = torch.zeros(size=(batch_size, total_num_of_grid_cells), dtype=torch.float32, device=DEVICE)

    batch_indices_of_ground_truth_boxes = get_indices_for_center_of_ground_truth_bounding_boxes(
        ground_truth_boxes, grid_sizes)

    batch_indices_with_highest_iou = get_indices_for_highest_iou_with_ground_truth_bounding_box(
        batch_indices_of_ground_truth_boxes, ground_truth_boxes, coordinates
    )

    for image_i in range(batch_indices_with_highest_iou.size(0)):
        for box_j in range(batch_indices_with_highest_iou.size(1)):
            idx = batch_indices_with_highest_iou[image_i, box_j]
            obj_mask[image_i, idx] = True
            noobj_mask[image_i, idx] = False

            ground_truth_box = ground_truth_boxes[image_i, box_j]
            target_coordinates[image_i, idx] = ground_truth_box[0:4]
            target_confidence[image_i, idx] = 1
            target_class_scores[image_i, idx, int(ground_truth_box[6])] = 1

    for image_i in range(batch_indices_of_ground_truth_boxes.size(0)):
        for box_j in range(batch_indices_of_ground_truth_boxes.size(1)):
            ground_truth_box = ground_truth_boxes[image_i, box_j]
            for idx in batch_indices_of_ground_truth_boxes[image_i, box_j]:
                cls_mask[image_i, idx] = True

    return obj_mask, noobj_mask, cls_mask, target_coordinates, target_confidence, target_class_scores


def training(model,
             ya_yolo_dataset,
             model_dir,
             summary_writer,
             epochs=1,
             lr=0.001,
             lambda_coord=5,
             lambda_no_obj=0.5,
             limit=None,
             debug=False,
             print_every=10):
    print('Number of images: ', len(ya_yolo_dataset))

    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grid_sizes = model.grid_sizes

    data_loader = DataLoader(ya_yolo_dataset, batch_size=ya_yolo_dataset.batch_size, shuffle=False)
    batch_size = ya_yolo_dataset.batch_size
    class_names = model.class_names

    for epoch in range(1, epochs + 1):

        running_loss = 0.0

        for batch_i, (images, annotations, image_paths) in enumerate(data_loader):

            images = images.to(DEVICE)
            if images.shape[0] != batch_size:
                print('Skipping batch {} because batch-size {} is not as expected {}'.format(batch_i + 1, len(images),
                                                                                             batch_size))
                continue

            coordinates, class_scores, confidence = model(images)

            ground_truth_boxes = ya_yolo_dataset.get_ground_truth_boxes(annotations).to(DEVICE)
            num_of_boxes_in_image_in_batch = ground_truth_boxes.shape[1]

            obj_mask, noobj_mask, cls_mask, target_coordinates, target_confidence, target_class_scores = build_targets(
                coordinates, class_scores, ground_truth_boxes, grid_sizes)

            if debug:
                print('processing batch {} with {} annotated objects per image ...'.format(batch_i + 1,
                                                                                           num_of_boxes_in_image_in_batch))
                plot(ground_truth_boxes.cpu(), images, class_names, True)
                plot(
                    to_plottable_boxes(obj_mask, coordinates, class_scores, confidence),
                    images, class_names, True)

            yolo_loss = YoloLoss(coordinates,
                                 confidence,
                                 class_scores,
                                 obj_mask,
                                 noobj_mask,
                                 cls_mask,
                                 target_coordinates,
                                 target_confidence,
                                 target_class_scores)

            loss = yolo_loss.get()

            # zero the parameter (weight) gradients
            optimizer.zero_grad()
            # backward pass to calculate the weight gradients
            loss.backward()
            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item() / batch_size

            # summary_writer.add_scalar('Loss/train', running_loss, batch_i)
            if batch_i % print_every == 0:  # print every print_every +1  batches
                log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(data_loader))
                log_str += yolo_loss.print()

                print(log_str)
                running_loss = 0.0

            if limit is not None and batch_i + 1 >= limit:
                print('Stop here after training {} batches (limit: {})'.format(batch_i, limit))
                return

        model.save(model_dir,
                   'yolo__num_classes_{}__epoch_{}.pt'.format(model.num_classes, epoch))
