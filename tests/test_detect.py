from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_forward_yolo():
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
    yolo = Yolo(cfg_file=cfg_file, batch_size=1)
    yolo.load_weights(weight_file)

    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    images,_  = image_and_target_transform(image, {})
    images = images.unsqueeze(0)

    coordinates, class_score, confidence = yolo(images)
    for b_i in range(coordinates.size(0)):
        boxes = nms_for_coordinates_and_class_scores_and_confidence(coordinates[b_i],
                                                                    class_score[b_i],
                                                                    confidence[b_i],
                                                                    0.6, 0.9)
        print(boxes)
        pil_image = to_pil_image(images[0])
        plot_boxes(pil_image, boxes, class_names, True)