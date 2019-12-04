from pprint import pformat

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from device import DEVICE
from logging_config import *
from metrics.detection import Detection
from metrics.ground_truth import GroundTruth
from metrics.metrics import Metrics
from metrics.utils import *
from yolo.plotting import plot_batch, save_batch
from yolo.utils import dir_exists_and_is_empty
from yolo.utils import non_max_suppression

logger = logging.getLogger(__name__)


def evaluate(model,
             dataset,
             summary_writer,
             images_results_dir,
             iou_thres, conf_thres, nms_thres,
             log_every=None,
             limit=None,
             plot=False,
             save=False):
    if save:
        assert dir_exists_and_is_empty(images_results_dir), f'{images_results_dir} is not empty or does not exist.'

    logger.info(
        f'Start evaluating model with iou_thres: {iou_thres}, conf_thres: {conf_thres} and nms_thres: {nms_thres}')

    metrics = Metrics()

    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=dataset.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        class_names = model.class_names

        total = limit if limit is not None else len(data_loader)
        for batch_i, (images, ground_truth_boxes, image_paths) in tqdm(enumerate(data_loader), total=total):
            if len(images) != dataset.batch_size:
                logger.warning(f"Skipping batch {batch_i} because it does not have correct size ({dataset.batch_size})")
                continue

            images = images.to(DEVICE)

            coordinates, class_scores, confidence = model(images)

            class_scores = torch.sigmoid(class_scores)

            prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)

            detections = non_max_suppression(prediction=prediction,
                                             conf_thres=conf_thres,
                                             nms_thres=nms_thres)

            if plot:
                plot_batch(
                    detections,
                    ground_truth_boxes, images, class_names)

            if save:
                save_batch(
                    image_paths,
                    images_results_dir,
                    detections,
                    ground_truth_boxes, images, class_names)

            ground_truth_map_objects = list(GroundTruth.from_ground_truths(image_paths, ground_truth_boxes))
            detection_map_objects = list(Detection.from_detections(image_paths, detections))

            metrics.add_detections_for_batch(detection_map_objects, ground_truth_map_objects, iou_thres=iou_thres)

            if limit is not None and batch_i >= limit:
                logger.info(f"Stop evaluation here after {batch_i} batches")
                break

            if batch_i != 0 and log_every is not None and batch_i % log_every == 0:
                log_average_precision_for_classes(metrics, class_names, summary_writer, batch_i)

        log_average_precision_for_classes(metrics, class_names, summary_writer, total)


def log_average_precision_for_classes(metrics, class_names, summary_writer, global_step):
    average_precision_for_classes, mAP = metrics.compute_average_precision_for_classes(class_names)
    logger.info(f'mAP: {mAP}\n')
    logging.info(
        '\n{}'.format(pformat(sorted(average_precision_for_classes.items(), key=lambda kv: kv[1], reverse=True))))

    plot_average_precision_on_tensorboard(average_precision_for_classes, mAP, summary_writer, global_step)
