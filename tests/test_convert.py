from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
namesfile = os.path.join(HERE, '../cfg/coco.names')
onnx_filename = os.path.join(HERE, 'output/yolo.onnx')
coreml_filename = os.path.join(HERE, 'output/yolo.mlmodel')

def test_convert():
    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)

    yolo = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=1)
    yolo.load_weights(weight_file)
    pytorch_to_onnx(yolo, dummy_input, onnx_filename)
    onnx_to_coreml(onnx_filename, coreml_filename)



def test_predict_with_coreml_with_nms():
    class_names = load_class_names(namesfile)

    transform_resize = Compose([
        SquashResize(416),
       # CocoToTensor()
    ])

    transform_resize_and_to_tensor = Compose([
        SquashResize(416),
        CocoToTensor()
    ])
    model = coremltools.models.MLModel(coreml_filename)

    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    resized, _ = transform_resize(image, {})
    #images = images.unsqueeze(0)
    output = model.predict({'image': resized}, usesCPUOnly=True)
    print(output)

    coordinates = output["coordinates"]
    class_scores = output["class_scores"]
    confidence = output["confidence"]

    coordinates = torch.tensor(coordinates)
    class_scores = torch.tensor(class_scores)
    confidence = torch.tensor(confidence)
    class_scores = torch.sigmoid(class_scores)
    prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)

    detections = non_max_suppression(prediction=prediction,
                                     conf_thres=0.9,
                                     nms_thres=0.5)

    resized_and_tensor, _ = transform_resize_and_to_tensor(image, {})
    plot_batch(detections, None, resized_and_tensor.unsqueeze(0), class_names)

