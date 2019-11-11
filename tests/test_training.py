from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ToPILImage, RandomCrop, RandomHorizontalFlip, Resize, ToTensor
from torchvision.transforms import transforms

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/train2014')
COCO_ANNOTATIONS_FILE = os.path.join(COCO_IMAGES_DIR, '../../annotations/instances_train2014_10_per_category.json')


def test_training():
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')

    batch_size = 2
    # Load the COCO object classes
    # class_names = load_class_names(namesfile)
    model = Yolo(cfg_file=cfg_file, batch_size=batch_size)
    model.load_weights(weight_file)

    transform = transforms.Compose([
        Resize((416, 416)),
        ToTensor()
    ])

    dataset = CocoDetection(root=COCO_IMAGES_DIR, annFile=COCO_ANNOTATIONS_FILE, transform=transform)

    training(model=model, dataset=dataset, num_epochs=1, batch_size=batch_size)
