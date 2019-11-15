import optparse
import os
import sys

from datasets.simple_car_dataset import SimpleCarDataset
from datasets.detected_car_dataset import DetectedCarDatasetWriter, DetectedCarDatasetHelper

from datasets.preprocess import *
from yolo.detect import detect_and_process
from logging_config import *
from yolo.utils import load_class_names
from yolo.yolo import Yolo
from device import DEVICE
from file_writer import FileWriter
import torch

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


def run_detect_cars(in_dir, out_file, batch_size, limit, iou_thresh, objectness_thresh, skip):
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    with torch.no_grad():
        model = Yolo(cfg_file=cfg_file, batch_size=batch_size)
        model.load_weights(weight_file)
        model.to(DEVICE)
        model.eval()
        namesfile = os.path.join(HERE, '../cfg/coco.names')
        class_names = load_class_names(namesfile)

        image_and_target_transform = Compose([
            SquashResize(416),
            CocoToTensor()
        ])

        dataset = SimpleCarDataset(
            root_dir=in_dir,
            transforms=image_and_target_transform,
            batch_size=batch_size)

        with FileWriter(file_path=out_file) as file_writer:
            car_dataset_writer = DetectedCarDatasetWriter(file_writer)

            detected_dataset_helper = DetectedCarDatasetHelper(car_dataset_writer=car_dataset_writer,
                                                               class_names=model.class_names,
                                                               iou_thresh=0.5,
                                                               objectness_thresh=0.9,
                                                               batch_size=batch_size,
                                                               plot=True)
            cnt = detect_and_process(model=model,
                                     ya_yolo_dataset=dataset,
                                     processor=detected_dataset_helper.process_detections,
                                     limit=limit)

            logger.info("Ran detection of {} images. Skipped first {} images".format(cnt, skip))


def run():
    logger.info('Start')

    parser = optparse.OptionParser('run_detect.py')
    parser.add_option("-o", "--out-file", dest="out_file",
                      help="out file where to write the new dataset as json to", metavar="FILE")

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
                      default=0.6,
                      help="iou threshold for non maximum suppression (default: 0.6)")

    parser.add_option("-p", "--objectness-thresh", dest="objectness_thresh",
                      type=float,
                      default=0.9,
                      help="objectness threshold for non maximum surpresssion (default: 0.9)")

    parser.add_option("-s", "--skip", dest="skip",
                      type=int,
                      default=None,
                      help="skip first n images")

    (options, args) = parser.parse_args()

    if not (options.in_dir or options.out_file):
        parser.print_help()
        sys.exit(1)
    else:
        in_dir = options.in_dir
        out_file = options.out_file
        batch_size = options.batch_size
        limit = options.limit
        iou_thresh = options.iou_thresh
        objectness_thresh = options.objectness_thresh
        skip = options.skip

        run_detect_cars(in_dir, out_file, batch_size, limit, iou_thresh, objectness_thresh, skip)

        sys.exit(0)


if __name__ == '__main__':
    run()
