from .context import *
import shutil
import datetime

HERE = os.path.dirname(os.path.realpath(__file__))
COCO_IMAGES_DIR = os.path.join(HERE, '../../../datasets/coco-small/cocoapi/images/train2014')
COCO_ANNOTATIONS_FILE = os.path.join(COCO_IMAGES_DIR, '../../annotations/instances_train2014_10_per_category.json')


def test_detect_and_process_for_detected_car_dataset():
    image_and_target_transform = Compose([
        SquashResize(416),
        ToTensor()
    ])
    batch_size = 3

    dataset = SimpleCarDataset(root_dir=os.path.join(os.environ['HOME'], 'datasets/simple-cars-small'),
                               transforms=image_and_target_transform, batch_size=batch_size)

    detected_cars = os.path.join(HERE, 'output/detected-cars')
    if os.path.exists(detected_cars):
        shutil.rmtree(detected_cars)
    os.mkdir(detected_cars)

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    detected_dataset_dir = os.path.join(detected_cars, now_str)
    os.mkdir(detected_dataset_dir)
    detected_dataset_images_dir = os.path.join(detected_dataset_dir, "images")
    os.mkdir(detected_dataset_images_dir)

    feed_file = os.path.join(detected_dataset_dir, "feed.json")

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)
    with torch.no_grad():
        model = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=batch_size)
        model.load_weights(weight_file)

        with FileWriter(file_path=feed_file) as file_writer:
            car_dataset_writer = DetectedCarDatasetWriter(file_writer)
            detected_dataset_helper = DetectedCarDatasetHelper(car_dataset_writer=car_dataset_writer,
                                                               class_names=model.class_names,
                                                               conf_thres=0.9,
                                                               nms_thres=0.5,
                                                               batch_size=batch_size,
                                                               debug=True)
            detect_and_process(model=model,
                               dataset=dataset,
                               processor=detected_dataset_helper.process_detections,
                               limit=10)


def test_detect_and_process_for_detected_car_dataset_with_missing_images():
    image_and_target_transform = Compose([
        SquashResize(416),
        ToTensor()
    ])

    dataset = SimpleCarDataset(root_dir='/home/peter/datasets/missing-images',
                               transforms=image_and_target_transform, batch_size=2)

    detected_cars = os.path.join(HERE, 'output/detected-cars')
    if os.path.exists(detected_cars):
        shutil.rmtree(detected_cars)
    os.mkdir(detected_cars)

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%dT%H-%M-%S")
    detected_dataset_dir = os.path.join(detected_cars, now_str)
    os.mkdir(detected_dataset_dir)

    feed_file = os.path.join(detected_dataset_dir, "feed.json")
    detected_dataset_images_dir = os.path.join(detected_dataset_dir, "images")
    os.mkdir(detected_dataset_images_dir)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    batch_size = 2
    with torch.no_grad():
        model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size)
        model.load_weights(weight_file)

        with FileWriter(file_path=feed_file) as file_writer:
            car_dataset_writer = DetectedCarDatasetWriter(detected_dataset_images_dir, file_writer)
            detected_dataset_helper = DetectedCarDatasetHelper(car_dataset_writer=car_dataset_writer,
                                                               class_names=model.class_names,
                                                               iou_thresh=0.5,
                                                               objectness_thresh=0.9,
                                                               batch_size=batch_size,
                                                               debug=True)
            detect_and_process(model=model,
                               dataset=dataset,
                               processor=detected_dataset_helper.process_detections,
                               limit=10)
