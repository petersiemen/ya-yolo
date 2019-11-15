import time

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from device import DEVICE
from logging_config import *
from yolo.mAP import MeanAveragePrecisionHelper
from yolo.utils import plot_boxes, nms_for_coordinates_and_class_scores_and_confidence

logger = logging.getLogger(__name__)

to_pil_image = transforms.Compose([
    ToPILImage()
])


def detect_cars(model,
                ya_yolo_dataset,
                car_dataset_writer,
                limit=None,
                batch_size=2,
                skip=None,
                plot=False,
                iou_thresh=0.5,
                objectness_thresh=0.9,
                ):
    logging.info(
        'start object detection run. Using iou_thresh {} and objectness_thresh {}'.format(
            iou_thresh,
            objectness_thresh))

    data_loader = DataLoader(ya_yolo_dataset, batch_size=ya_yolo_dataset.batch_size, shuffle=False)
    class_names = model.class_names

    cnt = 0
    detected = 0
    total = len(ya_yolo_dataset)
    if limit is not None:
        total = limit

    for batch_i, (images, annotations, image_paths) in tqdm(enumerate(data_loader), total=total / batch_size):
        try:
            cnt += batch_size
            if skip is not None and skip > cnt:
                logger.info('Skipping batch of {}. (skip: {}, cnt: {})'.format(batch_size, skip, cnt))
                continue

            images = images.to(DEVICE)
            logger.info('Start detection on batch {} of {} images...'.format(batch_i, len(images)))

            before = time.time()
            coordinates, class_scores, confidence = model(images)
            class_scores = torch.nn.Softmax(dim=2)(class_scores)

            logger.info('Forward pass on {} images took {} s'.format(len(images), time.time() - before))
            for b_i in range(batch_size):
                boxes = nms_for_coordinates_and_class_scores_and_confidence(
                    coordinates[b_i],
                    class_scores[b_i],
                    confidence[b_i],
                    iou_thresh,
                    objectness_thresh)

                if plot:
                    pil_image = to_pil_image(images[b_i])
                    plot_boxes(pil_image, boxes, class_names, True)

                if len(boxes) == 1 and boxes[0][6] == 2:
                    image_path = image_paths[b_i]
                    bb = boxes[0]
                    bounding_box = {
                        'x': bb[0], 'y': bb[1], 'w': bb[2], 'h': bb[3]
                    }

                    car_dataset_writer.append(image_path=image_path, make=annotations[0]['make'][b_i],
                                              model=annotations[0]['model'][b_i],
                                              bounding_box=bounding_box)
                    detected += 1

                elif len(boxes) > 0:
                    logger.warning('Found too many objects () on image. Discarding '.format(len(boxes),
                                                                                            image_paths[b_i]))
                else:
                    logger.warning('Found no object on image. Discarding {}'.format(
                        image_paths[b_i]))

            logger.info('Detected {} in {} images '.format(detected, cnt))
            if limit is not None:
                if cnt > limit:
                    logger.info(
                        'Stopping detection here. Limit {} has been reached after running detection on {} images'.format(
                            limit, cnt))
                    return cnt
            del images
        except Exception as ex:
            logger.error(ex)
            return cnt

    return cnt


def detect_for_map_computation(model,
                               ya_yolo_dataset,
                               out_dir,
                               limit=None,
                               plot=False,
                               batch_size=2,
                               iou_thresh=0.5,
                               objectness_thresh=0.9,
                               ):
    logging.info(
        'start object detection run. Using iou_thresh {} and objectness_thresh {}'.format(
            iou_thresh,
            objectness_thresh))

    mAPhelper = MeanAveragePrecisionHelper(out_dir)

    data_loader = DataLoader(ya_yolo_dataset, batch_size=ya_yolo_dataset.batch_size, shuffle=False)
    class_names = ya_yolo_dataset.get_classnames()

    cnt = 0
    detected = 0
    total = len(ya_yolo_dataset)
    if limit is not None:
        total = limit

    try:
        for batch_i, (images, annotations, image_paths) in tqdm(enumerate(data_loader), total=total / batch_size):
            cnt += batch_size
            images = images.to(DEVICE)
            logger.info('Start detection on batch {} of {} images...'.format(batch_i, len(images)))
            ground_truth_boxes = ya_yolo_dataset.get_ground_truth_boxes(annotations)

            before = time.time()
            coordinates, class_scores, confidence = model(images)
            class_scores = torch.nn.Softmax(dim=2)(class_scores)

            logger.info('Forward pass on {} images took {} s'.format(len(images), time.time() - before))
            for b_i in range(batch_size):
                boxes = nms_for_coordinates_and_class_scores_and_confidence(
                    coordinates[b_i],
                    class_scores[b_i],
                    confidence[b_i],
                    iou_thresh,
                    objectness_thresh)

                if plot:
                    pil_image = to_pil_image(images[b_i])
                    plot_boxes(pil_image, boxes, class_names, True)

                ground_truth = ground_truth_boxes[b_i]
                image_path = image_paths[b_i]

                print(ground_truth)
                print(boxes)
                print(image_path)

            if limit is not None and batch_i + 1 >= limit:
                print('Stop here after training {} batches (limit: {})'.format(batch_i, limit))
                return

    except Exception as ex:
        logger.error(ex)
