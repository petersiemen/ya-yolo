from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/val2014')
COCO_ANNOTATIONS_FILE = os.path.join(COCO_IMAGES_DIR, '../../annotations/instances_val2014_10_per_category.json')


def test_plot_boxes():
    batch_size = 1
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    dataset = YaYoloCocoDataset(images_dir=COCO_IMAGES_DIR, annotations_file=COCO_ANNOTATIONS_FILE,
                                transforms=image_and_target_transform,
                                batch_size=batch_size)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    yolo = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
    class_names = yolo.class_names
    yolo.load_weights(weight_file)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=dataset.collate_fn)
    batch_i, (images, ground_truth_boxes, image_paths) = next(enumerate(data_loader))

    coordinates, class_scores, confidence = yolo(images)
    prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)
    detections = non_max_suppression(prediction=prediction,
                                     conf_thres=0.9,
                                     nms_thres=0.5)

    plot_batch(detections,
               ground_truth_boxes,
               images,
               class_names)
