from torchvision.datasets import CocoDetection

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
# replace with path to your coco-dataset
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/train2014')
COCO_ANNOTATIONS_FILE = os.path.join(HERE, '../../../datasets/coco-small/annotations/instances_train2014_10_per_category.json')
COCO_NAMES_FILE = os.path.join(HERE, '../cfg/coco.names')


def test_compare_classnames_from_cocodataset_and_from_darknet_yolo():
    dataset = CocoDetection(root=COCO_IMAGES_DIR, annFile=COCO_ANNOTATIONS_FILE)
    for k, v in dataset.coco.cats.items():
        print(k, v)

    classnames = {k: v['name'] for k, v in dataset.coco.cats.items()}
    print(classnames)
    assert classnames[22] == 'elephant'
    assert classnames[3] == 'car'
    assert len(classnames) == 80


def test_classnames_from_coco():
    dataset = CocoDetection(root=COCO_IMAGES_DIR, annFile=COCO_ANNOTATIONS_FILE)
    for k, v in dataset.coco.cats.items():
        print(k, v)

    classnames = {k: v['name'] for k, v in dataset.coco.cats.items()}
    print(classnames)
    assert classnames[22] == 'elephant'
    assert classnames[3] == 'car'
