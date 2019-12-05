from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))
cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
namesfile = os.path.join(HERE, '../cfg/coco.names')
onnx_filename = os.path.join(HERE, 'output/yolo.onnx')
coreml_filename = os.path.join(HERE, 'output/Yolo.mlmodel')
core_ml_with_nms_filename = os.path.join(HERE, 'output/YoloNms.mlmodel')


def test_convert_to_coreml_via_onnx():
    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)
    yolo = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=1, coreml_mode=True)
    yolo.load_weights(weight_file)
    pytorch_to_onnx(yolo, dummy_input, onnx_filename)
    onnx_to_coreml(onnx_filename, coreml_filename)


def test_predict_with_coreml_without_nms():
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
    # images = images.unsqueeze(0)
    output = model.predict({'image': resized}, usesCPUOnly=True)
    print(output)

    coordinates = output["coordinates"]
    class_scores = output["class_scores"]
    confidence = output["det_conf"]

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


def test_export_yolo_to_coreml_with_nms():
    onnx_model = onnx.load(onnx_filename)
    coreml_model = convert(
        onnx_model,
        target_ios="13",
        image_input_names=['image'],
        preprocessing_args={
            "image_scale": 1 / 255.0,
            "red_bias": 0,
            "green_bias": 0,
            "blue_bias": 0}
    )
    builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=coreml_model.get_spec())

    B = 1
    maxBoxes = 1
    class_names = load_class_names(namesfile)
    builder.add_nms(name='nms',
                    input_names=['coordinates', 'class_scores'],
                    output_names=['boxes', 'scores', 'indices', 'remaining'],
                    iou_threshold=0.5,
                    score_threshold=0.9,
                    max_boxes=maxBoxes,
                    per_class_suppression=False
                    )

    builder.set_output(output_names=['boxes', 'scores'],
                       output_dims=[
                           (B, maxBoxes, 4,),
                           (B, maxBoxes, 80,)
                       ])

    user_defined_metadata = {
        "classes": ",".join(class_names),
        "key": "ObjectDetector"
    }
    builder.spec.description.metadata.userDefined.update(user_defined_metadata)

    print(builder.spec.description)
    save_spec(builder.spec, core_ml_with_nms_filename)

  #  shutil.copy(core_ml_model_with_nms, core_ml_model_with_nms_in_ios_app)



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
    model = coremltools.models.MLModel(core_ml_with_nms_filename)

    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    resized, _ = transform_resize(image, {})
    # images = images.unsqueeze(0)
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
