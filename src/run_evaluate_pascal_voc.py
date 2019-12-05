import argparse
import os
import sys

from torch.utils.tensorboard import SummaryWriter

from datasets.preprocess import *
from datasets.yayolo_voc_dataset import YaYoloVocDataset
from logging_config import *
from yolo.evaluate import evaluate
from yolo.yolo import Yolo
from yolo.utils import load_class_names

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


def evaluate_pascal_voc(root_dir, download, batch_size,
                        conf_thres, log_every, limit, plot, save,
                        images_results_dir):
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)

    model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
    model.load_weights(weight_file)

    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    dataset = YaYoloVocDataset(root_dir=root_dir,
                               batch_size=batch_size,
                               transforms=image_and_target_transform,
                               image_set='val',
                               download=download,
                               class_names=class_names)

    summary_writer = SummaryWriter(comment=f' evaluate={batch_size}')
    evaluate(model, dataset, summary_writer,
             images_results_dir,
             iou_thres=0.5,
             conf_thres=conf_thres,
             nms_thres=0.5,
             log_every=log_every,
             limit=limit,
             plot=plot,
             save=save)

    summary_writer.close()


def run():
    parser = argparse.ArgumentParser('run_evaluate.py')
    parser.add_argument("--root-dir", dest="root_dir",
                        help="location of pascal voc", metavar="FILE")

    parser.add_argument("-d", "--download", dest="download",
                        action="store_true",
                        help="download dataset ")

    parser.add_argument("-c", "--conf-thres", dest="conf_thres",
                        type=float,
                        default=0.5,
                        help="confidence threshold used in nms")

    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        type=int,
                        default=5,
                        help="batch_size for reading the raw dataset (default: 5)")

    parser.add_argument("-l", "--limit", dest="limit",
                        type=int,
                        default=None,
                        help="limit the size of the to be generated dataset (default: None)")

    parser.add_argument("-n", "--log-every", dest="log_every",
                        type=int,
                        default=None,
                        help="log ever n-th batch")

    parser.add_argument("-r", "--images-results-dir", dest="images_results_dir",
                        default=None,
                        help="location where to store the rendered detections and ground-truth boxes on the images")

    parser.add_argument("-p", "--plot", help="plot results",
                        action="store_true")

    parser.add_argument("-s", "--save", help="save results",
                        action="store_true")

    args = parser.parse_args()
    if args.root_dir is None:
        parser.print_help()
        sys.exit(1)
    else:
        root_dir = args.root_dir
        download = args.download
        batch_size = args.batch_size
        log_every = args.log_every
        limit = args.limit
        conf_thres = args.conf_thres
        plot = args.plot
        save = args.save
        images_results_dir = args.images_results_dir

        evaluate_pascal_voc(root_dir, download, batch_size, conf_thres, log_every, limit, plot, save,
                            images_results_dir)

        sys.exit(0)


if __name__ == '__main__':
    run()
