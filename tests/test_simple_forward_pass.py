from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_forward_yolo():
    image_and_target_transform = Compose([
        SquashResize(416),
        CocoToTensor()
    ])
    # for b_i in range(coordinates.size(0)):
    #     boxes = detections[b_i].detach()
    #
    #     print(boxes)
    #     pil_image = to_pil_image(images[b_i])
    #     plot_boxes(pil_image, boxes, boxes, class_names)

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    yolo = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=1)
    class_names = yolo.class_names
    yolo.load_weights(weight_file)

    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    images, _ = image_and_target_transform(image, {})
    images = images.unsqueeze(0)

    coordinates, class_scores, confidence = yolo(images)
    class_scores = torch.nn.Softmax(dim=2)(class_scores)
    prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)
    detections = non_max_suppression(prediction=prediction,
                                     conf_thres=0.9,
                                     nms_thres=0.5)
    plot_batch(detections, None, images, class_names)
