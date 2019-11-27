from torch.utils.tensorboard import SummaryWriter

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/val2014')
COCO_ANNOTATIONS_FILE = os.path.join(COCO_IMAGES_DIR, '../../annotations/instances_val2014_10_per_category.json')


def test_evaluate():
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')

    batch_size = 3
    model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
    model.load_weights(weight_file)

    image_and_target_transform = Compose([
        ConvertXandYToCenterOfBoundingBox(),
        AbsoluteToRelativeBoundingBox(),
        SquashResize(416),
        CocoToTensor()
    ])

    ya_yolo_dataset = YaYoloCocoDataset(images_dir=COCO_IMAGES_DIR, annotations_file=COCO_ANNOTATIONS_FILE,
                                        transforms=image_and_target_transform,
                                        batch_size=batch_size)

    summary_writer = SummaryWriter(comment=f' evaluate={batch_size}')
    images_result_dir = os.path.join(HERE, 'output/evaluated')
    evaluate(model, ya_yolo_dataset, summary_writer, images_result_dir,
             conf_thresh=0.7,
             log_every=1,
             limit=2,
             plot=True,
             save=True)
