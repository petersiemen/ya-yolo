from .context import *
import shutil
from coremltools.models.neural_network import flexible_shape_utils

HERE = os.path.dirname(os.path.realpath(__file__))
cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
namesfile = os.path.join(HERE, '../cfg/coco.names')
onnx_filename = os.path.join(HERE, 'output/yolo.onnx')
coreml_filename = os.path.join(HERE, 'output/Yolo.mlmodel')
core_ml_with_nms_filename = os.path.join(HERE, 'output/YoloNms.mlmodel')
core_ml_with_nms_filename_in_yolo_ios = os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/YoloNms.mlmodel')


def test_convert_to_onnx():
    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)
    yolo = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=1, coreml_mode=True)
    yolo.load_weights(weight_file)
    pytorch_to_onnx(yolo, dummy_input, onnx_filename)

def test_convert_to_coreml():
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
    output = model.predict({'image': resized}, usesCPUOnly=True)
    print(output)

    coordinates = output["boxes"]
    class_scores = output["scores"]
    confidence = output["objectness"]

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
    builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=coreml_model.get_spec(),
                                                                     disable_rank5_shape_mapping=True
                                                                     )

    class_names = load_class_names(namesfile)
    builder.add_nms(name='nms',
                    input_names=['boxes', 'scores'],
                    output_names=['coordinates', 'confidence', 'indices', 'remaining'],
                    iou_threshold=0.5,
                    score_threshold=0.9,
                    per_class_suppression=False
                    )

    builder.set_output(output_names=['coordinates', 'confidence'],
                       output_dims=[(0, 4), (0, 80)])

    del builder.spec.description.output[2]

    user_defined_metadata = {
        "classes": ",".join(class_names),
        "key": "ObjectDetector"
    }
    builder.spec.description.metadata.userDefined.update(user_defined_metadata)

    # flexible_shape_utils.set_multiarray_ndshape_range(spec=builder.spec,
    #                                                   feature_name="coordinates",
    #                                                   lower_bounds=[0, 4],
    #                                                   upper_bounds=[-1, 4])
    #
    # flexible_shape_utils.set_multiarray_ndshape_range(spec=builder.spec,
    #                                                   feature_name="confidence",
    #                                                   lower_bounds=[0, 80],
    #                                                   upper_bounds=[-1, 80])
    #
    # del builder.spec.description.output[0].type.multiArrayType.shape[0]
    # del builder.spec.description.output[0].type.multiArrayType.shape[0]
    # del builder.spec.description.output[1].type.multiArrayType.shape[0]
    # del builder.spec.description.output[1].type.multiArrayType.shape[0]

    print(builder.spec.description.output)
    save_spec(builder.spec, core_ml_with_nms_filename)

    shutil.copy(core_ml_with_nms_filename, core_ml_with_nms_filename_in_yolo_ios)


def test_predict_with_coreml_with_nms():
    class_names = load_class_names(namesfile)

    transform_resize = Compose([
        SquashResize(416),
    ])

    model = coremltools.models.MLModel(core_ml_with_nms_filename)

    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    resized, _ = transform_resize(image, {})
    # images = images.unsqueeze(0)
    output = model.predict({'image': resized}, usesCPUOnly=True)
    print(output)

    coordinates = output["coordinates"]
    class_scores = output["confidence"]
    # confidence = output["confidence"]

    print()
    coordinates = torch.tensor(coordinates)
    class_scores = torch.tensor(class_scores)
    # confidence = torch.tensor(confidence)
    # class_scores = torch.sigmoid(class_scores)
    # prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)

    # detections = non_max_suppression(prediction=prediction,
    #                                 conf_thres=0.9,
    #                                 nms_thres=0.5)

    # resized_and_tensor, _ = transform_resize_and_to_tensor(image, {})
    # plot_batch(detections, None, resized_and_tensor.unsqueeze(0), class_names)


def test_predict_dices():
    model = coremltools.models.MLModel(
        os.path.join(HERE,'../../yolo-ios/YoloIOS/Models/DiceDetector.mlmodel')
    )

    transform_resize = Compose([
        SquashResize(736),
    ])
    image = load_image_file(os.path.join(HERE, './images/dices.jpg'))
    resized, _ = transform_resize(image, {})
    # images = images.unsqueeze(0)
    output = model.predict({'image': resized}, usesCPUOnly=True)
    print(output)

    coordinates = output["coordinates"]
    class_scores = output["confidence"]
