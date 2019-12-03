import argparse
import os
import shutil
import sys
import json
from tqdm import tqdm

from datasets.detected_car_dataset import DetectedCarDataset
from logging_config import *

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))

import random

random.seed(0)


def split_into_train_and_test(in_file, split):
    assert os.path.exists(in_file), f"file {in_file} does not exist"
    train_file_path = os.path.join(os.path.dirname(in_file), "train.json")
    test_file_path = os.path.join(os.path.dirname(in_file), "test.json")
    assert os.path.exists(train_file_path) == False, f"{train_file_path} already exists."
    assert os.path.exists(test_file_path) == False, f"{test_file_path} already exists."

    with open(train_file_path, 'w') as train_file:
        with open(test_file_path, 'w') as test_file:
            with open(in_file) as f:
                lines = f.readlines()
                random.shuffle(lines)
                for line in lines:
                    if random.random() < split:
                        test_file.write(line)
                    else:
                        train_file.write(line)


def run():
    parser = argparse.ArgumentParser('run_train_test_split.py')
    parser.add_argument("-i", "--in-file", dest="in_file",
                        help="location of raw dataset", metavar="FILE")

    parser.add_argument('-s', '--split',
                        dest='split',
                        type=float,
                        default=0.1,
                        help="train/test split. default 0.1")

    args = parser.parse_args()
    if args.in_file is None:
        parser.print_help()
        sys.exit(1)
    else:
        in_file = args.in_file
        split = args.split

        split_into_train_and_test(in_file, split)

        sys.exit(0)


if __name__ == '__main__':
    run()
