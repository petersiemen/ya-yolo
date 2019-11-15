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
    scores = output["scores"]
    confidence = output["confidence"]

    coordinates = torch.tensor(coordinates)
    scores = torch.tensor(scores)
    for b_i in range(coordinates.size(0)):
        boxes = nms_for_coordinates_and_class_scores_and_confidence(coordinates[b_i], scores[b_i], confidence[b_i], 0.6,
                                                                    0.9)
        plot_boxes(image, boxes, class_names, True)
