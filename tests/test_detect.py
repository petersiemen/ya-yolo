from .context import *
import shutil

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/train2014')
COCO_ANNOTATIONS_FILE = os.path.join(COCO_IMAGES_DIR, '../../annotations/instances_train2014_10_per_category.json')


def test_detect_and_process():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    dataset = SimpleCarDataset(root_dir='/home/peter/datasets/simple_cars/2019-08-23T10-22-54',
                               transforms=image_and_target_transform, batch_size=2)

    to_file = os.path.join(HERE, 'output/cars.json')
    if os.path.exists(to_file):
        os.remove(to_file)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    batch_size = 2
    with torch.no_grad():
        model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
        model.load_weights(weight_file)

        with FileWriter(file_path=to_file) as file_writer:
            car_dataset_writer = DetectedCarDatasetWriter(file_writer)
            detected_dataset_helper = DetectedCarDatasetHelper(car_dataset_writer=car_dataset_writer,
                                                               class_names=model.class_names,
                                                               iou_thresh=0.5,
                                                               objectness_thresh=0.9,
                                                               batch_size=batch_size,
                                                               plot=True)
            detect_and_process(model=model,
                               ya_yolo_dataset=dataset,
                               processor=detected_dataset_helper.process_detections,
                               limit=10)


def test_detect_for_map_computation():
    detection_results_dir = os.path.join(HERE, 'output/detection-results')
    ground_truth_dir = os.path.join(HERE, 'output/ground-truth')
    if os.path.exists(detection_results_dir):
        shutil.rmtree(detection_results_dir)
    os.mkdir(detection_results_dir)
    if os.path.exists(ground_truth_dir):
        shutil.rmtree(ground_truth_dir)
    os.mkdir(ground_truth_dir)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    out_dir = os.path.join(HERE, 'output')

    image_and_target_transform = Compose([
        ConvertXandYToCenterOfBoundingBox(),
        AbsoluteToRelativeBoundingBox(),
        SquashResize(416),
        CocoToTensor()
    ])
    batch_size = 2
    dataset = YaYoloCocoDataset(images_dir=COCO_IMAGES_DIR, annotations_file=COCO_ANNOTATIONS_FILE,
                                transforms=image_and_target_transform,
                                batch_size=batch_size)

    iou_thresh = 0.5
    objectness_thresh = 0.9
    with torch.no_grad():

        model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
        model.load_weights(weight_file)

        mAPHelper = MeanAveragePrecisionHelper(out_dir=out_dir,
                                               class_names=model.class_names,
                                               iou_thresh=iou_thresh,
                                               objectness_thresh=objectness_thresh,
                                               batch_size=batch_size,
                                               plot=True
                                               )
        detect_and_process(model=model,
                           ya_yolo_dataset=dataset,
                           processor=mAPHelper.process_detections,
                           limit=10)
