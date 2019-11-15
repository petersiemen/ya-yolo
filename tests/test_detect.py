from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_detect_with_batches():
    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)
    batch_size = 2
    with torch.no_grad():
        model = Yolo(cfg_file=cfg_file, batch_size=batch_size)
        print()

        model.load_weights(weight_file)

        image_and_target_transform = Compose([
            SquashResize(416),
            CocoToTensor()
        ])

        dataset = SimpleCarDataset(root_dir='/home/peter/datasets/simple_cars/2019-08-23T10-22-54',
                                   transforms=image_and_target_transform, batch_size=2)

        to_file = os.path.join(HERE, 'output/cars.json')
        if os.path.exists(to_file):
            os.remove(to_file)

        with FileWriter(file_path=to_file) as file_writer:
            car_dataset_writer = DetectedSimpleCarDatasetWriter(file_writer)

            detect_cars(model=model,
                        ya_yolo_dataset=dataset,
                        class_names=class_names,
                        car_dataset_writer=car_dataset_writer,
                        limit=10,
                        batch_size=batch_size,
                        plot=True)
