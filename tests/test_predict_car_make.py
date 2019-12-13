from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_predict_car_make():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])
    batch_size = 1

    with open(
            os.path.join(os.environ['HOME'],
                         'datasets/detected-cars/more_than_4000_detected_per_make/makes.csv'), encoding="utf-8") as f:
        class_names = [make.strip() for make in f.readlines()]

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    model = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=batch_size, coreml_mode=False)

    model.load_state_dict(
        torch.load(os.path.join(HERE, '../models/yolo__num_classes_80__epoch_2_batch_7500.pt'), map_location=DEVICE))

    dataset = DetectedCareMakeDataset(
        json_file=os.path.join(os.environ['HOME'],
                               'datasets/detected-cars/more_than_4000_detected_per_make/test.json'),
        transforms=image_and_target_transform, batch_size=batch_size)
    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)
    limit = 10

    for batch_i, (images, ground_truth_boxes, _) in enumerate(data_loader):

        coordinates, class_scores, confidence = model(images)
        prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)

        detections = non_max_suppression(prediction=prediction,
                                         conf_thres=0.9,
                                         nms_thres=0.8
                                         )

        plot_batch(detections, ground_truth_boxes, images, dataset.class_names)
        if batch_i > limit:
            break

