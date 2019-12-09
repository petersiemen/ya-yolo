from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_predict_car_make():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])
    batch_size = 1

    with open('/home/peter/datasets/detected-cars/more_than_4000_detected_per_make/makes.csv') as f:
        class_names = [make.strip() for make in f.readlines()]

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    model = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=batch_size, coreml_mode=False)
    model.load_weights(weight_file)
    model.set_num_classes(len(class_names))
    model.set_class_names(class_names)

    model.load_state_dict(
        torch.load(os.path.join(HERE, '../models/yolo__num_classes_80__epoch_2_batch_7500.pt'), map_location=DEVICE))

    dataset = DetectedCareMakeDataset(
        json_file='/home/peter/datasets/detected-cars/more_than_4000_detected_per_make/test.json',
        transforms=image_and_target_transform, batch_size=batch_size)
    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)
    limit = 10

    for batch_i, (images, ground_truth_boxes, _) in enumerate(data_loader):
        plot_batch(None, ground_truth_boxes, images, dataset.class_names)
        if batch_i > limit:
            break


    # image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    # images, _ = image_and_target_transform(image, {})
    # images = images.unsqueeze(0)

    # coordinates, class_scores, confidence = model(images)
    # class_scores = torch.sigmoid(class_scores)
    # prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)
    # detections = non_max_suppression(prediction=prediction,
    #                                  conf_thres=0.9,
    #                                  nms_thres=0.5)
    # plot_batch(detections, None, images, class_names)
