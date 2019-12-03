import argparse
import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.yayolo_coco_dataset import YaYoloCocoDataset
from datasets.preprocess import *
from logging_config import *
from yolo.yolo import Yolo
from yolo.train import train
from device import DEVICE

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


def train_coco(coco_images_dir,
               coco_annotations_file,
               batch_size,
               lr,
               conf_thres,
               gradient_accumulations,
               epochs,
               limit,
               log_every,
               save_every,
               model_dir,
               parameters):
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
    model.load_weights(weight_file)

    if parameters is not None:
        logger.info(f"loading model parameters from {parameters}")
        model.load_state_dict(
            torch.load(parameters,
                       map_location=DEVICE))

    # this recreates the last convolutional layer before the yolo layer
    model.set_num_classes(model.num_classes)

    image_and_target_transform = Compose([
        ConvertXandYToCenterOfBoundingBox(),
        AbsoluteToRelativeBoundingBox(),
        SquashResize(416),
        CocoToTensor()
    ])

    dataset = YaYoloCocoDataset(images_dir=coco_images_dir,
                                annotations_file=coco_annotations_file,
                                transforms=image_and_target_transform,
                                batch_size=batch_size)

    summary_writer = SummaryWriter(comment=f' evaluate={batch_size}')
    train(model=model,
          ya_yolo_dataset=dataset,
          model_dir=model_dir,
          summary_writer=summary_writer,
          epochs=epochs,
          lr=lr,
          conf_thres=conf_thres,
          nms_thres=0.5,
          iou_thres=0.5,
          lambda_coord=5,
          lambda_no_obj=0.5,
          gradient_accumulations=gradient_accumulations,
          limit=limit,
          debug=False,
          print_every=log_every,
          save_every=save_every)
    summary_writer.close()


def run():
    parser = argparse.ArgumentParser('run_retrain_coco.py')
    parser.add_argument("-i", "--images-dir", dest="images_dir",
                        help="images dir", metavar="FILE")

    parser.add_argument("-a", "--annotations-file", dest="annotations_file",
                        help="annotations file", metavar="FILE")

    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        type=int,
                        default=8,
                        help="batch_size for reading the raw dataset (default: 5)")

    parser.add_argument("-r", "--learning-rate", dest="lr",
                        type=float,
                        default=0.001,
                        help="learning-rate")

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

    parser.add_argument("-s", "--save-every", dest="save_every",
                        default=None,
                        type=int,
                        help="after how many batches we are saving the models")

    parser.add_argument("-p", "--parameters", dest="parameters",
                        default=None,
                        metavar="FILE",
                        help="if given we initialize the model with these params")

    args = parser.parse_args()
    if args.images_dir is None or args.annotations_file is None:
        parser.print_help()
        sys.exit(1)
    else:
        images_dir = args.images_dir
        annotations_file = args.annotations_file
        batch_size = args.batch_size
        lr = args.lr
        conf_thres = args.conf_thres
        gradient_accumulations = args.gradient_accumulations
        epochs = args.epochs
        log_every = args.log_every
        limit = args.limit
        model_dir = args.model_dir
        save_every = args.save_every
        parameters = args.parameters

        train_coco(images_dir,
                   annotations_file,
                   batch_size,
                   lr,
                   conf_thres,
                   gradient_accumulations,
                   epochs,
                   limit,
                   log_every,
                   save_every,
                   model_dir,
                   parameters)

        sys.exit(0)


if __name__ == '__main__':
    run()
