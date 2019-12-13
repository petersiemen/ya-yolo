from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_save_and_reload():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])

    to_pil_image = transforms.Compose([
        ToPILImage()
    ])

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)
    yolo = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=1)
    yolo.load_weights(weight_file)

    yolo.save(dir=os.path.join(HERE, 'models'), name='test.pt')

    yolo_from_disc = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=1)
    yolo_from_disc.reload(dir=os.path.join(HERE, 'models'), name='test.pt')

    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    images, _ = image_and_target_transform(image, {})
    images = images.unsqueeze(0)

    coordinates, class_scores, confidence = yolo(images)
    prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)
    detections = non_max_suppression(prediction=prediction,
                                     conf_thres=0.9,
                                     nms_thres=0.5)
    plot_batch(detections,
               None,
               images,
               class_names)

    coordinates, class_score, confidence = yolo_from_disc(images)

    prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)
    detections = non_max_suppression(prediction=prediction,
                                     conf_thres=0.9,
                                     nms_thres=0.5)

    plot_batch(detections,
               None,
               images,
               class_names)
