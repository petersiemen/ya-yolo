import os
from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_freeze_params():
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)

    batch_size = 2
    lr = 0.001
    model = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=batch_size)
    model.freeze_parameters()
