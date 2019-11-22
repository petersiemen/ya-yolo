from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_predict_with_coreml():
    coreml_filename = os.path.join(HERE, 'output/yolo.mlmodel')
    namesfile = os.path.join(HERE, '../cfg/coco.names')
    class_names = load_class_names(namesfile)

    transform = transforms.Compose([
        SquashResize(416)
    ])

    model = coremltools.models.MLModel(coreml_filename)
    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    batch = transform({'image': image})

    output = model.predict(batch, usesCPUOnly=True)
    print(output)

    coordinates = output["boxes"]
    class_scores = output["scores"]
    confidence = output["confidence"]

    coordinates = torch.tensor(coordinates)
    class_scores = torch.tensor(class_scores)

    prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)
    detections = non_max_suppression(prediction=prediction,
                                     conf_thres=0.9,
                                     nms_thres=0.5)

    boxes = detections[0].detach()
    if len(boxes) > 0:
        boxes[..., :4] = xyxy2xywh(boxes[..., :4])

    plot_boxes(image, boxes, class_names, True)
