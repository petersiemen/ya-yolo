from torchvision.datasets import CocoDetection

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/train2014')
COCO_ANNOTATIONS_FILE = os.path.join(COCO_IMAGES_DIR, '../../annotations/instances_train2014_10_per_category.json')


def test_get_indices_for_center_of_ground_truth_bounding_boxes__for_no_annotations():
    ground_truth_boxes = torch.tensor([[],[]])
    grid_sizes = [13, 26, 52]
    indices = get_indices_for_center_of_ground_truth_bounding_boxes(ground_truth_boxes, grid_sizes)
    assert indices.shape == (2, 0)


def test_training():
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    model_dir = os.path.join(HERE, 'models')

    batch_size = 2
    model = Yolo(cfg_file=cfg_file, batch_size=batch_size)
    model.load_weights(weight_file)

    image_and_target_transform = Compose([
        SquashResize(416),
        ConvertXandYToCenterOfBoundingBox(),
        ScaleBboxRelativeToSize(416),
        # PadToFit(255),
        # RandomCrop(200),
        # RandomHorizontalFlip(),,
        CocoToTensor()
    ])

    dataset = CocoDetection(root=COCO_IMAGES_DIR, annFile=COCO_ANNOTATIONS_FILE, transforms=image_and_target_transform)
    ya_yolo_dataset = YoloCocoDataset(dataset=dataset, batch_size=batch_size)

    training(model=model, ya_yolo_dataset=ya_yolo_dataset, model_dir=model_dir, num_epochs=1, lr=0.001, limit=3,
             debug=True)
