import numpy as np
import torch
from torch.utils.data import DataLoader

from device import DEVICE
from logging_config import *
from yolo.loss import YoloLoss
from yolo.plotting import plot_batch
from yolo.utils import boxes_iou_for_single_boxes
from yolo.utils import non_max_suppression
from metrics.utils import *
from metrics.metrics import Metrics
from metrics.ground_truth import GroundTruth
from metrics.detection import Detection
from pprint import pformat
from tqdm import tqdm
import random
from yolo.layers import ConvolutionalLayer
from torch.nn.utils import clip_grad_norm_
import neptune
from dotenv import load_dotenv
from pathlib import Path  # python3 only

random.seed(0)

logger = logging.getLogger(__name__)


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
    batch_size = len(ground_truth_boxes)
    indices_for_batch = []
    for image_i in range(batch_size):
        indices_for_image = []
        num_of_boxes_in_image = len(ground_truth_boxes[image_i])
        for box_j in range(num_of_boxes_in_image):
            x = ground_truth_boxes[image_i][box_j][0]
            y = ground_truth_boxes[image_i][box_j][1]
            indices = list(get_indices_for_center_of_bounding_boxes(num_anchors=3,
                                                                    grid_widths=grid_sizes,
                                                                    x=x,
                                                                    y=y))
            indices_for_image.append(indices)
        indices_for_batch.append(torch.tensor(indices_for_image).to(DEVICE))
    return indices_for_batch


def get_indices_for_highest_iou_with_ground_truth_bounding_box(indices, ground_truth_boxes, coordinates):
    batch_size = len(ground_truth_boxes)
    # num_of_boxes_in_image_in_batch = ground_truth_boxes.shape[1]
    indices_for_batch = []
    for image_i in range(batch_size):
        indices_for_image = []
        num_of_boxes_in_image = len(ground_truth_boxes[image_i])
        for box_j in range(num_of_boxes_in_image):
            candidate_coordinates = coordinates[image_i, indices[image_i][box_j]]
            ious = [boxes_iou_for_single_boxes(ground_truth_boxes[image_i][box_j], candidate_box) for candidate_box in
                    candidate_coordinates]
            max_iou_idx = np.argmax(ious)
            indices_for_image.append(indices[image_i][box_j][max_iou_idx])
        indices_for_batch.append(indices_for_image)
    return indices_for_batch


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

    for image_i in range(batch_size):
        num_of_boxes_in_image = len(ground_truth_boxes[image_i])
        for box_j in range(num_of_boxes_in_image):
            idx = batch_indices_with_highest_iou[image_i][box_j]
            obj_mask[image_i, idx] = True
            noobj_mask[image_i, idx] = False

            ground_truth_box = ground_truth_boxes[image_i][box_j]
            target_coordinates[image_i, idx] = ground_truth_box[0:4]
            target_confidence[image_i, idx] = 1
            target_class_scores[image_i, idx, int(ground_truth_box[6])] = 1

    for image_i in range(batch_size):
        num_of_boxes_in_image = len(ground_truth_boxes[image_i])
        for box_j in range(num_of_boxes_in_image):
            for idx in batch_indices_of_ground_truth_boxes[image_i][box_j]:
                cls_mask[image_i, idx] = True

    return obj_mask, noobj_mask, cls_mask, target_coordinates, target_confidence, target_class_scores


def train(model,
          dataset,
          model_dir,
          summary_writer,
          epochs,
          lr,
          conf_thres,
          nms_thres,
          iou_thres,
          lambda_coord=5,
          lambda_no_obj=0.5,
          gradient_accumulations=2,
          clip_gradients=False,
          limit=None,
          debug=False,
          print_every=10,
          save_every=None,
          log_to_neptune=False):
    if log_to_neptune:
        env_path = Path(os.environ['HOME'], 'workspace/setup-box/neptune.env')
        load_dotenv(dotenv_path=env_path)

        neptune.init('petersiemen/sandbox',
                     api_token=os.getenv("NEPTUNE_API_TOKEN"))

    total = limit if limit is not None else len(dataset)

    logger.info(
        f'Start training on {total} images. Using lr: {lr}, '
        f'lambda_coord: {lambda_coord}, lambda_no_obj: {lambda_no_obj}, '
        f'conf_thres: {conf_thres}, nms_thres:{nms_thres}, iou_thres: {iou_thres}, '
        f'gradient_accumulations: {gradient_accumulations}, '
        f'clip_gradients: {clip_gradients}, lambda_no_obj: {lambda_no_obj}')
    metrics = Metrics()

    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=lr)
    grid_sizes = model.grid_sizes

    data_loader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True,
                             collate_fn=dataset.collate_fn)
    class_names = model.class_names

    for epoch in range(1, epochs + 1):
        for batch_i, (images, ground_truth_boxes, image_paths) in tqdm(enumerate(data_loader), total=total):
            if len(images) != dataset.batch_size:
                logger.warning(
                    f"Skipping batch {batch_i} because it does not have correct size ({dataset.batch_size})")
                continue

            images = images.to(DEVICE)

            coordinates, class_scores, confidence = model(images)

            obj_mask, noobj_mask, cls_mask, target_coordinates, target_confidence, target_class_scores = build_targets(
                coordinates, class_scores, ground_truth_boxes, grid_sizes)
            yolo_loss = YoloLoss(coordinates,
                                 confidence,
                                 class_scores,
                                 obj_mask,
                                 noobj_mask,
                                 cls_mask,
                                 target_coordinates,
                                 target_confidence,
                                 target_class_scores,
                                 lambda_coord=lambda_coord,
                                 lambda_no_obj=lambda_no_obj
                                 )

            class_scores = torch.sigmoid(class_scores)
            prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)

            detections = non_max_suppression(prediction=prediction,
                                             conf_thres=conf_thres,
                                             nms_thres=nms_thres
                                             )

            ground_truth_map_objects = list(GroundTruth.from_ground_truths(image_paths, ground_truth_boxes))
            detection_map_objects = list(Detection.from_detections(image_paths, detections))

            metrics.add_detections_for_batch(detection_map_objects, ground_truth_map_objects, iou_thres=iou_thres)

            if debug:
                plot_batch(detections, ground_truth_boxes, images, class_names)

            loss = yolo_loss.get()
            # backward pass to calculate the weight gradients
            loss.backward()

            if clip_gradients:
                logger.debug("Clipping gradients with max_norm = 1")
                clip_grad_norm_(model.parameters(), max_norm=1)

            if batch_i % print_every == 0:  # print every print_every +1  batches
                yolo_loss.capture(summary_writer, batch_i, during='train')
                plot_weights_and_gradients(model, summary_writer, epoch * batch_i)
                log_performance(epoch, epochs, batch_i, total, yolo_loss, metrics, class_names, summary_writer,log_to_neptune)

            # Accumulates gradient before each step
            if batch_i % gradient_accumulations == 0:
                logger.debug(
                    f"Updating weights for batch {batch_i} (gradient_accumulations :{gradient_accumulations})")
                # update the weights
                optimizer.step()
                # zero the parameter (weight) gradients
                optimizer.zero_grad()

            del images
            del ground_truth_boxes

            if limit is not None and batch_i + 1 >= limit:
                logger.info('Stop here after training {} batches (limit: {})'.format(batch_i, limit))
                log_performance(epoch, epochs, batch_i, total, yolo_loss, metrics, class_names, summary_writer, log_to_neptune)
                save_model(model_dir, model, epoch, batch_i)
                return

            if save_every is not None and batch_i % save_every == 0:
                save_model(model_dir, model, epoch, batch_i)

        # save model after every epoch
        save_model(model_dir, model, epoch, None)


def log_performance(epoch, epochs, batch_i, total, yolo_loss, metrics, class_names, summary_writer, log_to_neptune):
    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, total)
    log_str += yolo_loss.print()
    logger.info(log_str)
    log_average_precision_for_classes(metrics, class_names, summary_writer, epoch * batch_i)

    if log_to_neptune:
        neptune.log_metric('iteration', batch_i)
        neptune.log_metric('loss', yolo_loss.total_loss)


def log_average_precision_for_classes(metrics, class_names, summary_writer, global_step):
    average_precision_for_classes, mAP = metrics.compute_average_precision_for_classes(class_names)
    logger.info(f'mAP: {mAP}\n')

    if len(average_precision_for_classes) > 0:
        logging.info(
            '\n{}'.format(pformat(sorted(average_precision_for_classes.items(), key=lambda kv: kv[1], reverse=True))))

        fig = draw_fig(average_precision_for_classes,
                       window_title="mAP",
                       plot_title="mAP = {0:.2f}%".format(mAP * 100),
                       x_label="Average Precision"
                       )
        data = fig_to_numpy(fig)
        summary_writer.add_image('Average_Precision', data, global_step, dataformats='HWC')


def save_model(model_dir, model, epoch, batch):
    model.save(model_dir,
               'yolo__num_classes_{}__epoch_{}_batch_{}.pt'.format(model.num_classes, epoch, batch))


def plot_weights_and_gradients(model, summary_writer, global_step):
    for layer in model.models:
        if isinstance(layer, ConvolutionalLayer):
            layer_idx = layer.layer_idx
            conv = layer.models[0]
            weight = conv.weight
            summary_writer.add_histogram(f'conv_{layer_idx}/weight', weight.view(-1), global_step)
            if weight.grad is not None:
                weight_grad = weight.grad
                summary_writer.add_histogram(f'conv_{layer_idx}/weight_grad', weight_grad.view(-1), global_step)
            if conv.bias is not None:
                bias = conv.bias
                summary_writer.add_histogram(f'conv_{layer_idx}/bias', bias.view(-1), global_step)
                if conv.bias.grad is not None:
                    bias_grad = conv.bias.grad
                    summary_writer.add_histogram(f'conv_{layer_idx}/bias_grad', bias_grad.view(-1), global_step)
