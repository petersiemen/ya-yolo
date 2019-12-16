import argparse
import os
import sys


import torch
from torch.utils.tensorboard import SummaryWriter

from datasets.detected_car_dataset import DetectedCareMakeDataset
from datasets.preprocess import *
from logging_config import *
from yolo.yolo import Yolo
from yolo.train import train
from device import DEVICE

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


def train_car_make(car_make_json_file,
                   batch_size,
                   lr,
                   conf_thres,
                   gradient_accumulations,
                   clip_gradients,
                   epochs,
                   limit,
                   log_every,
                   save_every,
                   model_dir,
                   parameters,
                   lambda_no_obj):
    image_and_target_transform = Compose([
        SquashResize(416),
        ToTensor()
    ])

    dataset = DetectedCareMakeDataset(json_file=car_make_json_file,
                                      transforms=image_and_target_transform, batch_size=batch_size)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')

    model = Yolo(cfg_file=cfg_file, class_names=dataset.class_names, batch_size=batch_size, coreml_mode=False)
    model.load_weights(weight_file)

    if parameters is not None:
        logger.info(f"loading model parameters from {parameters}")
        model.load_state_dict(
            torch.load(parameters,
                       map_location=DEVICE))

    summary_writer = SummaryWriter(comment=f' evaluate={batch_size}')

    # with neptune.create_experiment(name='start-with-neptune',
    #                                params=PARAMS):
    #     neptune.append_tag('first-example')

    train(model=model,
          dataset=dataset,
          model_dir=model_dir,
          summary_writer=summary_writer,
          epochs=epochs,
          lr=lr,
          conf_thres=conf_thres,
          nms_thres=0.5,
          iou_thres=0.5,
          lambda_coord=5,
          lambda_no_obj=lambda_no_obj,
          gradient_accumulations=gradient_accumulations,
          clip_gradients=clip_gradients,
          limit=limit,
          debug=False,
          print_every=log_every,
          save_every=save_every)
    summary_writer.close()


def run():
    parser = argparse.ArgumentParser('run_train_car_make.py')
    parser.add_argument("-f", "--car-make-json-file", dest="car_make_json_file",
                        help="location the detected car make file", metavar="FILE")

    parser.add_argument("-b", "--batch-size", dest="batch_size",
                        type=int,
                        default=8,
                        help="batch_size for reading the raw dataset (default: 8)")

    parser.add_argument("-r", "--learning-rate", dest="lr",
                        type=float,
                        default=0.001,
                        help="learning-rate (default 0.001)")

    parser.add_argument("-c", "--conf-thres", dest="conf_thres",
                        type=float,
                        default=0.5,
                        help="confidence threshold used in nms (default 0.5)")

    parser.add_argument("-g", "--gradient-accumulations", dest="gradient_accumulations",
                        type=int,
                        default=2,
                        help="number of batches to accumulate the losses over before backpropagating the gradients (default 2)")

    parser.add_argument("--clip-gradients", dest="clip_gradients",
                        action="store_true",
                        help="clip gradients")

    parser.add_argument("-e", "--epochs", dest="epochs",
                        type=int,
                        default=5,
                        help="number of epochs to train (default 5)")

    parser.add_argument("-l", "--limit", dest="limit",
                        type=int,
                        default=None,
                        help="limit the size of the to be generated dataset (default: None)")

    parser.add_argument("-n", "--log-every", dest="log_every",
                        type=int,
                        default=100,
                        help="log ever n-th batch (default 100)")

    parser.add_argument("-m", "--model-dir", dest="model_dir",
                        default='models',
                        metavar="FILE",
                        help="location where to store the trained pytorch models (default models)")

    parser.add_argument("-s", "--save-every", dest="save_every",
                        default=500,
                        type=int,
                        help="after how many batches we are saving the models (default 500)")

    parser.add_argument("-p", "--parameters", dest="parameters",
                        default=None,
                        metavar="FILE",
                        help="if given we initialize the model with these params")

    parser.add_argument( "--lambda-no-obj", dest="lambda_no_obj",
                        default=0.5,
                        type=float,
                        help="weight for the no object loss")


    args = parser.parse_args()
    if args.car_make_json_file is None:
        parser.print_help()
        sys.exit(1)
    else:
        car_make_json_file = args.car_make_json_file
        batch_size = args.batch_size
        lr = args.lr
        conf_thres = args.conf_thres
        gradient_accumulations = args.gradient_accumulations
        clip_gradients = args.clip_gradients
        epochs = args.epochs
        log_every = args.log_every
        limit = args.limit
        model_dir = args.model_dir
        save_every = args.save_every
        parameters = args.parameters
        lambda_no_obj = args.lambda_no_obj

        train_car_make(car_make_json_file,
                       batch_size,
                       lr,
                       conf_thres,
                       gradient_accumulations,
                       clip_gradients,
                       epochs,
                       limit,
                       log_every,
                       save_every,
                       model_dir,
                       parameters,
                       lambda_no_obj)

        sys.exit(0)


if __name__ == '__main__':
    run()
