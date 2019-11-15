from torchvision.datasets import CocoDetection

from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
# replace with path to your coco-dataset
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/train2014')
COCO_ANNOTATIONS_FILE = os.path.join(COCO_IMAGES_DIR, '../../annotations/instances_train2014_10_per_category.json')
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


def test_preprocess_coco():
    image_and_target_transform = Compose([
        ConvertXandYToCenterOfBoundingBox(),
        AbsoluteToRelativeBoundingBox(),
        SquashResize(416),

        # PadToFit(255),
        # RandomCrop(200),
        # RandomHorizontalFlip(),,
        CocoToTensor()
    ])

    to_pil_image = transforms.Compose([
        ToPILImage()
    ])

    dataset = CocoDetection(root=COCO_IMAGES_DIR, annFile=COCO_ANNOTATIONS_FILE,
                            transforms=image_and_target_transform)
    classnames = {k: v['name'] for k, v in dataset.coco.cats.items()}

    batch_size = 2
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch_i, (images, annotations) in enumerate(data_loader):

        for b_i in range(batch_size):
            pil_image = to_pil_image(images[b_i])
            boxes = []
            for o_i in range(len(annotations)):
                bbox_coordinates = annotations[o_i]['bbox']

                x = bbox_coordinates[0][b_i]
                y = bbox_coordinates[1][b_i]
                w = bbox_coordinates[2][b_i]
                h = bbox_coordinates[3][b_i]
                category_id = annotations[o_i]['category_id'][b_i]
                box = [x, y, w, h, 1, 1, category_id, 1]
                boxes.append(box)

            boxes = torch.tensor(boxes)
            plot_boxes(pil_image, boxes, classnames, True)

        if batch_i > 3:
            break
