import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from device import DEVICE
from logging_config import *

logger = logging.getLogger(__name__)


def detect_and_process(model,
                       ya_yolo_dataset,
                       processor,
                       limit=None,
                       skip=None,
                       iou_thresh=0.5,
                       objectness_thresh=0.9,
                       ):
    logging.info(
        'start object detection run. Using iou_thresh {} and objectness_thresh {}'.format(
            iou_thresh,
            objectness_thresh))

    batch_size = ya_yolo_dataset.batch_size
    data_loader = DataLoader(ya_yolo_dataset, batch_size=ya_yolo_dataset.batch_size, shuffle=False)

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

            processor(
                coordinates,
                class_scores,
                confidence,
                images,
                annotations,
                image_paths
            )

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
