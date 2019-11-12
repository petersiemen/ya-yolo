import time

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from device import DEVICE
from logging_config import *
from yolo.utils import plot_boxes, nms_for_coordinates_and_class_scores_and_confidence

logger = logging.getLogger(__name__)

to_pil_image = transforms.Compose([
    ToPILImage()
])


def detect(model,
           dataset,
           class_names,
           limit=None,
           batch_size=2,
           skip=None,
           plot=False,
           iou_thresh=0.4,
           objectness_thresh=0.9):
    logging.info(
        'start dectecting objects in {} images. Using iou_thresh {} and objectness_thresh {}'.format(len(dataset),
                                                                                                     iou_thresh,
                                                                                                     objectness_thresh))

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    cnt = 0
    detected = 0
    total = len(dataset)
    if limit is not None:
        total = limit

    try:
        for batch_i, batch in tqdm(enumerate(data_loader), total=total / batch_size):
            cnt += batch_size
            if skip is not None and skip > cnt:
                logger.info('Skipping batch of {}. (skip: {}, cnt: {})'.format(batch_size, skip, cnt))
                continue

            images = batch['image'].to(DEVICE)
            logger.info('Start detection on batch {} of {} images...'.format(batch_i, len(images)))

            before = time.time()
            coordinates, class_scores, confidence = model(images)
            logger.info('Forward pass on {} images took {} s'.format(len(images), time.time() - before))
            for b_i in range(batch_size):
                boxes = nms_for_coordinates_and_class_scores_and_confidence(
                    coordinates[b_i],
                    class_scores[b_i],
                    confidence[b_i],
                    iou_thresh,
                    objectness_thresh)
                pil_image = to_pil_image(images[b_i])
                if plot:
                    plot_boxes(pil_image, boxes, class_names, True)

                elif len(boxes) > 0:
                    logger.warning('Found too many objects () on image. Discarding '.format(len(boxes),
                                                                                            batch['image_path'][
                                                                                                b_i]))
                else:
                    logger.warning('Found no car on image. Discarding {}'.format(
                        batch['image_path'][b_i]))

            logger.info('Detected {} in {} images '.format(detected, cnt))
            if limit is not None:
                if cnt > limit:
                    logger.info(
                        'Stopping detection here. Limit {} has been reached after running detection on {} images'.format(
                            limit, cnt))
                    return cnt
            del images
            del batch

    except Exception as ex:
        logger.error(ex)
        return cnt

    return cnt


def print_cuda_stats():
    from torch.cuda import memory_allocated, max_memory_allocated, memory_cached, max_memory_cached

    logger.info(
        '\nmemory_allocated: {:.2f}, \nmax_memory_allocated: {:.2f}, \nmemory_cached: {:.2f}, \nmax_memory_cached: {:.2f}'.format(
            memory_allocated() / 1024. / 1024.,
            max_memory_allocated() / 1024. / 1024.,
            memory_cached() / 1024. / 1024.,
            max_memory_cached() / 1024. / 1024.,
        ))
