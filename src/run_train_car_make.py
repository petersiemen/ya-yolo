import argparse
import os
import sys

from torch.utils.tensorboard import SummaryWriter

from datasets.detected_car_dataset import DetectedCareMakeDataset
from datasets.preprocess import *
from logging_config import *
from yolo.yolo import Yolo
from yolo.train import train

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


def train_car_make(car_make_json_file,
                   batch_size,
                   conf_thres,
                   gradient_accumulations,
                   epochs,
                   limit,
                   log_every,
                   model_dir):
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    dataset = DetectedCareMakeDataset(json_file=car_make_json_file,
                                      transforms=image_and_target_transform, batch_size=batch_size)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')

    model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
    model.load_weights(weight_file)
    model.set_num_classes(dataset.get_num_classes())
    model.set_class_names(dataset.get_class_names())

    summary_writer = SummaryWriter(comment=f' evaluate={batch_size}')
    train(model=model,
          ya_yolo_dataset=dataset,
          model_dir=model_dir,
          summary_writer=summary_writer,
          epochs=epochs,
          lr=0.01,
          conf_thres=conf_thres,
          nms_thres=0.5,
          iou_thres=0.5,
          lambda_coord=5,
          lambda_no_obj=0.5,
          gradient_accumulations=gradient_accumulations,
          limit=limit,
          debug=False,
          print_every=log_every)
    summary_writer.close()


def run():
    parser = argparse.ArgumentParser('run_train_car_make.py')
    parser.add_argument("-f", "--car-make-json-file", dest="car_make_json_file",
                        help="location the detected car make file", metavar="FILE")

    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        type=int,
                        default=8,
                        help="batch_size for reading the raw dataset (default: 5)")

    parser.add_argument("-c", "--conf-thres", dest="conf_thres",
                        type=float,
                        default=0.5,
                        help="confidence threshold used in nms")

    parser.add_argument("-g", "--gradient-accumulations", dest="gradient_accumulations",
                        type=int,
                        default=2,
                        help="number of batches to accumulate the losses over before backpropagating the gradients")

    parser.add_argument("-e", "--epochs", dest="epochs",
                        type=int,
                        default=5,
                        help="number of epochs to train")

    parser.add_argument("-l", "--limit", dest="limit",
                        type=int,
                        default=None,
                        help="limit the size of the to be generated dataset (default: None)")

    parser.add_argument("-n", "--log-every", dest="log_every",
                        type=int,
                        default=100,
                        help="log ever n-th batch")

    parser.add_argument("-m", "--model-dir", dest="model_dir",
                        default='models',
                        metavar="FILE",
                        help="location where to store the trained pytorch models")

    args = parser.parse_args()
    if args.car_make_json_file is None:
        parser.print_help()
        sys.exit(1)
    else:
        car_make_json_file = args.car_make_json_file
        batch_size = args.batch_size
        conf_thres = args.conf_thres
        gradient_accumulations = args.gradient_accumulations
        epochs = args.epochs
        log_every = args.log_every
        limit = args.limit
        model_dir = args.model_dir

        train_car_make(car_make_json_file,
                       batch_size,
                       conf_thres,
                       gradient_accumulations,
                       epochs,
                       limit,
                       log_every,
                       model_dir)

        sys.exit(0)


if __name__ == '__main__':
    run()
