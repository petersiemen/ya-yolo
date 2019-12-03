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


def _read_makes(makes_file):
    with open(makes_file) as f:
        makes = [m.strip() for m in f.readlines()]
    return makes


def run_create_filtered_detected_cars_datast(in_file, out_dir, makes_to_keep, limit):
    assert os.path.isdir(out_dir), "directory {} does not exist".format(out_dir)
    assert len(os.listdir(out_dir)) == 0, "directory {} is not empty".format(out_dir)

    out_images_dir = os.path.join(out_dir, "images")
    os.mkdir(out_images_dir)

    dataset = DetectedCarDataset(
        json_file=in_file,
        transforms=None,
        batch_size=1)

    out_feed_file = os.path.join(out_dir, "feed.json")

    existing_makes = set()

    with open(out_feed_file, 'w') as f:
        total = limit if limit is not None else len(dataset)
        for i, (image, annotations, image_path) in tqdm(enumerate(dataset), total=total):
            annotation = annotations[0]
            make = annotation['make']
            if make in makes_to_keep:
                shutil.copy(image_path, os.path.join(out_images_dir, os.path.basename(image_path)))
                f.write(json.dumps(annotation) + '\n')
                existing_makes.add(make)
                if limit is not None and i > limit:
                    logger.info(f"Stopping here because limit {limit} was set")
                    break

    out_makes_file = os.path.join(out_dir, "makes.csv")
    with open(out_makes_file, 'w') as f:
        f.write('\n'.join(existing_makes))


def run():
    logger.info('Start')

    parser = argparse.ArgumentParser('create_car_dataset.py')
    parser.add_argument("-i", "--in-file", dest="in_file",
                        help="location of raw dataset", metavar="FILE")

    parser.add_argument('-o', '--out-dir', metavar='FILE',
                        help="where to write the new dataset to")

    parser.add_argument("-m", "--makes", dest="makes",
                        type=str,
                        metavar='FILE',
                        help="csv file with makes to keep")

    parser.add_argument("-l", "--limit", dest="limit",
                        type=int,
                        default=None,
                        help="limit the size of the to be generated dataset (default: None)")

    args = parser.parse_args()
    if args.in_file is None or args.out_dir is None or args.makes is None:
        parser.print_help()
        sys.exit(1)
    else:
        in_file = args.in_file
        out_dir = args.out_dir
        limit = args.limit

        makes = _read_makes(args.makes)

        run_create_filtered_detected_cars_datast(in_file, out_dir, makes, limit)

        sys.exit(0)


if __name__ == '__main__':
    run()
