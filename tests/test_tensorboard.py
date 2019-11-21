from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from collections import OrderedDict

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_tensorboard():
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')

    batch_size = 2
    COCO_IMAGES_DIR = '/home/peter/datasets/coco-small/cocoapi/images/train2014'
    # COCO_IMAGES_DIR = '/home/ubuntu/datasets/coco/train2014'
    COCO_ANNOTATIONS_FILE = '/home/peter/datasets/coco-small/cocoapi/annotations/instances_train2014_10_per_category.json'
    # COCO_ANNOTATIONS_FILE = '/home/ubuntu/datasets/coco/annotations/instances_train2014.json'

    image_and_target_transform = Compose([
        ConvertXandYToCenterOfBoundingBox(),
        AbsoluteToRelativeBoundingBox(),
        SquashResize(416),
        CocoToTensor()
    ])

    ya_yolo_dataset = YaYoloCocoDataset(images_dir=COCO_IMAGES_DIR,
                                        annotations_file=COCO_ANNOTATIONS_FILE,
                                        transforms=image_and_target_transform,
                                        batch_size=batch_size)
    trainloader = torch.utils.data.DataLoader(ya_yolo_dataset, batch_size=batch_size, shuffle=False)

    images, targets, image_paths = next(iter(trainloader))
    print(images.shape)

    model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
    print(model)
    # model.load_weights(weight_file)

    writer = SummaryWriter()
    writer.add_graph(model, images, verbose=True)
    writer.close()


def test_tensorboard_names_and_scopes():
    class Net(nn.Module):
        def __init__(self, ):
            super(Net, self).__init__()

            self.model = nn.ModuleList([
                nn.Conv2d(1, 20, 5),
                nn.ReLU(),
                nn.Conv2d(20, 64, 5),
                nn.ReLU()
            ])

        def forward(self, x):
            for i, layer in enumerate(self.model):
                x = layer(x)
            return x
            # return self.model(x)

    net = Net()
    # print(net)
    # print(w)
    #
    epoch = 1
    writer = SummaryWriter()
    #
    for model in net.model:
        if isinstance(model, nn.Conv2d):
            writer.add_histogram(str(model) + '.weight', model.weight, epoch)

    #images = torch.rand(2, 1, 416, 416)
    images = torch.rand(1, 416, 416)
    writer.add_graph(net, images.unsqueeze(0), verbose=True)
    writer.close()
