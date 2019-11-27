from pprint import pformat

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from device import DEVICE
from logging_config import *
from metrics.bounding_box import BoundingBox
from metrics.detection import Detection
from metrics.ground_truth import GroundTruth
from metrics.metrics import Metrics
from metrics.utils import *
from yolo.plotting import plot_batch, save_batch
from yolo.utils import dir_exists_and_is_empty
from yolo.utils import non_max_suppression

logger = logging.getLogger(__name__)


def _image_path_to_image_id(image_path):
    return os.path.basename(image_path).split('.jpg', 1)[0]


def to_mAP_detections(image_paths, detections):
    batch_size = len(detections)
    for image_i in range(batch_size):
        image_id = _image_path_to_image_id(image_paths[image_i])
        for detection in detections[image_i]:
            detection = detection.detach().cpu().numpy()
            confidence = detection[4] * detection[5]
            yield Detection(file_id=image_id, class_id=int(detection[-1]), confidence=confidence,
                            bounding_box=BoundingBox.from_xywh(
                                x=detection[0], y=detection[1], w=detection[2], h=detection[3]
                            ))


def to_mAP_ground_truths(image_paths, ground_truths):
    batch_size = ground_truths.size(0)
    for image_i in range(batch_size):
        image_id = _image_path_to_image_id(image_paths[image_i])
        for ground_truth in ground_truths[image_i]:
            ground_truth = ground_truth.detach().numpy()
            yield GroundTruth(file_id=image_id, class_id=int(ground_truth[-1]), bounding_box=BoundingBox.from_xywh(
                x=ground_truth[0], y=ground_truth[1], w=ground_truth[2], h=ground_truth[3]
            ))


def evaluate(model,
             ya_yolo_dataset,
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
        data_loader = DataLoader(ya_yolo_dataset, batch_size=ya_yolo_dataset.batch_size, shuffle=True)
        class_names = model.class_names

        total = limit if limit is not None else len(data_loader)
        for batch_i, (images, annotations, image_paths) in tqdm(enumerate(data_loader), total=total):
            if len(images) != ya_yolo_dataset.batch_size:
                logger.warning(
                    f'batch {batch_i} does not contain {ya_yolo_dataset.batch_size} training samples. skipping this batch')
                continue

            images = images.to(DEVICE)
            ground_truth_boxes = ya_yolo_dataset.get_ground_truth_boxes(annotations)

            coordinates, class_scores, confidence = model(images)

            # converting the class scores into a probability distribution
            # this only has an effect on how what probability we 'plot' into the resulting images for
            # debugging
            # the nms function below will just pick the maximum class score
            class_scores = torch.nn.Softmax(dim=2)(class_scores)

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

            ground_truth_map_objects = list(to_mAP_ground_truths(image_paths, ground_truth_boxes))
            detection_map_objects = list(to_mAP_detections(image_paths, detections))

            metrics.add_detections_for_batch(detection_map_objects, ground_truth_map_objects, iou_thres=iou_thres)

            if limit is not None and batch_i >= limit:
                logger.info(f"Stop evaluation here after {batch_i} batches")
                break

            if batch_i != 0 and log_every is not None and batch_i % log_every == 0:
                log_average_precision_for_classes(metrics, class_names, summary_writer, batch_i)

        log_average_precision_for_classes(metrics, class_names, summary_writer, total)


def log_average_precision_for_classes(metrics, class_names, summary_writer, global_step):
    average_precision_for_classes, mAP, ground_truth_objects_for_classes = metrics.compute_average_precision_for_classes()
    average_precision_for_classes = dict([('{} (id: {}, #gt:{} )'.format(
        class_names[int(k)], int(k),
        ground_truth_objects_for_classes[int(k)]), v) for k, v in
        average_precision_for_classes.items()])

    logger.info(f'mAP: {mAP}\n')
    logging.info(
        '\n{}'.format(pformat(sorted(average_precision_for_classes.items(), key=lambda kv: kv[1], reverse=True))))

    plot_average_precision_on_tensorboard(average_precision_for_classes, mAP, summary_writer, global_step)
