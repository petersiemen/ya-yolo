import optparse
import os
import sys

import torch

from datasets.preprocess import *
from datasets.simple_car_dataset import SimpleCarDataset
from device import DEVICE
from logging_config import *
from yolo.detect import detect_and_process
from yolo.mean_average_precision_helper import MeanAveragePrecisionHelper
from yolo.yolo import Yolo

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


def run_detect_for_mAP(in_dir, out_dir, batch_size, limit, iou_thresh, objectness_thresh):
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    with torch.no_grad():
        model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
        model.load_weights(weight_file)
        model.to(DEVICE)
        model.eval()
        image_size = 416

        image_and_target_transform = Compose([
            SquashResize(image_size),
            CocoToTensor()
        ])

        dataset = SimpleCarDataset(
            root_dir=in_dir,
            transforms=image_and_target_transform,
            batch_size=batch_size)

        mAPHelper = MeanAveragePrecisionHelper(out_dir=out_dir,
                                               class_names=model.class_names,
                                               image_size=image_size,
                                               iou_thresh=iou_thresh,
                                               objectness_thresh=objectness_thresh,
                                               batch_size=batch_size,
                                               keep_images=True,
                                               plot=True
                                               )
        cnt = detect_and_process(model=model,
                                 ya_yolo_dataset=dataset,
                                 processor=mAPHelper.process_detections,
                                 limit=limit)

        logger.info("Ran detection of {} images.".format(cnt))


def run():
    logger.info('Start')

    parser = optparse.OptionParser('detect_for_mAP_computation.py')
    parser.add_option("-o", "--out-dir", dest="out_dir",
                      help="out dir where to collect results", metavar="FILE")

    parser.add_option("-i", "--in-dir", dest="in_dir",
                      help="location of raw dataset", metavar="FILE")

    parser.add_option("-b", "--batch-size", dest="batch_size",
                      type=int,
                      default=5,
                      help="batch_size for reading the raw dataset (default: 5)")

    parser.add_option("-l", "--limit", dest="limit",
                      type=int,
                      default=None,
                      help="limit the size of the to be generated dataset (default: None)")

    parser.add_option("-t", "--iou-thresh", dest="iou_thresh",
                      type=float,
                      default=0.5,
                      help="iou threshold for non maximum suppression (default: 0.6)")

    parser.add_option("-p", "--objectness-thresh", dest="objectness_thresh",
                      type=float,
                      default=0.9,
                      help="objectness threshold for non maximum surpresssion (default: 0.9)")

    (options, args) = parser.parse_args()

    if not (options.in_dir or options.out_file):
        parser.print_help()
        sys.exit(1)
    else:
        in_dir = options.in_dir
        out_dir = options.out_dir
        batch_size = options.batch_size
        limit = options.limit
        iou_thresh = options.iou_thresh
        objectness_thresh = options.objectness_thresh

        run_detect_for_mAP(in_dir, out_dir, batch_size, limit, iou_thresh, objectness_thresh)

        sys.exit(0)


if __name__ == '__main__':
    run()
