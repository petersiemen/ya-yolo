import datetime
import argparse
import os
import sys

import torch

from datasets.detected_car_dataset import DetectedCarDatasetWriter, DetectedCarDatasetHelper
from datasets.preprocess import *
from datasets.simple_car_dataset import SimpleCarDataset
from device import DEVICE
from file_writer import FileWriter
from logging_config import *
from yolo.detect import detect_and_process
from yolo.yolo import Yolo
from yolo.utils import load_class_names

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


def run_detect_cars(in_dir, out_dir, batch_size, limit, conf_thres, nms_thres, debug):
    assert os.path.isdir(out_dir), "directory {} does not exist".format(out_dir)

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    detected_dataset_dir = os.path.join(out_dir, now_str)
    os.mkdir(detected_dataset_dir)

    feed_file = os.path.join(detected_dataset_dir, "feed.json")
    detected_dataset_images_dir = os.path.join(detected_dataset_dir, "images")
    os.mkdir(detected_dataset_images_dir)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)
    with torch.no_grad():
        model = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=batch_size)
        model.load_weights(weight_file)
        model.to(DEVICE)
        model.eval()

        image_and_target_transform = Compose([
            SquashResize(416),
            ToTensor()
        ])

        dataset = SimpleCarDataset(
            root_dir=in_dir,
            transforms=image_and_target_transform,
            batch_size=batch_size)

        with FileWriter(file_path=feed_file) as file_writer:
            car_dataset_writer = DetectedCarDatasetWriter(file_writer)

            detected_dataset_helper = DetectedCarDatasetHelper(car_dataset_writer=car_dataset_writer,
                                                               class_names=model.class_names,
                                                               conf_thres=conf_thres,
                                                               nms_thres=nms_thres,
                                                               batch_size=batch_size,
                                                               debug=debug)
            cnt = detect_and_process(model=model,
                                     dataset=dataset,
                                     processor=detected_dataset_helper.process_detections,
                                     limit=limit)

            logger.info("Ran detection of {} images".format(cnt))


def run():
    logger.info('Start')

    parser = argparse.ArgumentParser('create_car_dataset.py')
    parser.add_argument('-o', '--out-dir', metavar='FILE',
                        help="where to write the new dataset to")

    parser.add_argument("-i", "--in-dir", dest="in_dir",
                        help="location of raw dataset", metavar="FILE")

    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        type=int,
                        default=5,
                        help="batch_size for reading the raw dataset (default: 5)")

    parser.add_argument("-l", "--limit", dest="limit",
                        type=int,
                        default=None,
                        help="limit the size of the to be generated dataset (default: None)")

    parser.add_argument("-c", "--conf-thres", dest="conf_thres",
                        type=float,
                        default=0.9,
                        help="detection confidence threshold(default: 0.9)")

    parser.add_argument("-n", "--nms-thres", dest="nms_thres",
                        type=float,
                        default=0.5,
                        help="nms (iou) threshold for non maximum suppression (default: 0.5)")

    parser.add_argument("-d", "--debug", help="plot during detection",
                        action="store_true")

    args = parser.parse_args()
    if args.in_dir is None or args.out_dir is None:
        parser.print_help()
        sys.exit(1)
    else:
        in_dir = args.in_dir
        out_dir = args.out_dir
        batch_size = args.batch_size
        limit = args.limit
        conf_thres = args.conf_thres
        nms_thres = args.nms_thres

        debug = args.debug

        run_detect_cars(in_dir, out_dir, batch_size, limit, conf_thres, nms_thres, debug)

        sys.exit(0)


if __name__ == '__main__':
    run()
