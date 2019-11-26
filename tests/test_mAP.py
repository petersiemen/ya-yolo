from .context import *

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
namesfile = os.path.join(HERE, '../cfg/coco.names')


def test_mAP():
    detected = [
        # x y w h conf class image_id
        [],

    ]

    ground_truth = [
        # x y w h class image_id
        [176, 206, 225, 266, 0, 0],
        [170, 156, 350, 240, 1, 0]
    ]

    gt = GroundTruth(file_id='sdf', class_id=1, bounding_box=None)
    print(gt)
