from .context import *
import shutil
from coremltools.models.neural_network import flexible_shape_utils
import coremltools
from coremltools.models.pipeline import *

HERE = os.path.dirname(os.path.realpath(__file__))
# cfg_file = os.path.join(HERE, '../cfg/mini-yolov3.cfg')
cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
namesfile = os.path.join(HERE, '../cfg/coco.names')
onnx_filename = os.path.join(HERE, 'output/mini_yolo.onnx')
coreml_filename = os.path.join(HERE, 'output/MiniYolo.mlmodel')
coreml_nms_filename = os.path.join(HERE, 'output/NMS.mlmodel')
coreml_nms_filename_in_yolo_ios = os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/NMS.mlmodel')

coreml_filename_in_yolo_ios = os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/MiniYolo.mlmodel')

core_ml_with_nms_filename = os.path.join(HERE, 'output/MiniYoloNms.mlmodel')
core_ml_with_nms_filename_in_yolo_ios = os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/MiniYoloNms.mlmodel')

coreml_pipeline_filename = os.path.join(HERE, 'output/YoloPipeline.mlmodel')
coreml_pipeline_filename_in_yolo_ios = os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/YoloPipeline.mlmodel')
coreml_pipeline_filename_in_yolo_simple_ios = os.path.join(HERE, '../../yolo-simple-ios/YoloSimpleIOS/YoloPipeline.mlmodel')

def test_convert_to_onnx():
    # yolo model
    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)
    class_names = load_class_names(namesfile)
    yolo = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=1, coreml_mode=True)
    yolo.load_weights(weight_file)
    pytorch_to_onnx(yolo, dummy_input, onnx_filename)

    onnx_model = onnx.load(onnx_filename)
    coreml_model = convert(
        onnx_model,
        minimum_ios_deployment_target="13",
        image_input_names=['image'],
        preprocessing_args={
            "image_scale": 1 / 255.0,
            "red_bias": 0,
            "green_bias": 0,
            "blue_bias": 0}
    )

    yolo_model = coremltools.models.MLModel(coreml_model.get_spec())

    yolo_model.save(coreml_filename)
    shutil.copy(coreml_filename, coreml_filename_in_yolo_ios)

    # nms model
    nms_spec = coremltools.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 3
    # boxes
    yolo_boxes = yolo_model._spec.description.output[0].SerializeToString()
    nms_spec.description.input.add()
    nms_spec.description.input[0].ParseFromString(yolo_boxes)
    nms_spec.description.output.add()
    nms_spec.description.output[0].ParseFromString(yolo_boxes)
    nms_spec.description.output[0].name = "coordinates"
    # scores
    yolo_scores = yolo_model._spec.description.output[1].SerializeToString()
    nms_spec.description.input.add()
    nms_spec.description.input[1].ParseFromString(yolo_scores)
    nms_spec.description.output.add()
    nms_spec.description.output[1].ParseFromString(yolo_scores)
    nms_spec.description.output[1].name = "confidence"

    # coordinates
    ma_type = nms_spec.description.output[0].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = 4
    ma_type.shapeRange.sizeRanges[1].upperBound = 4
    del ma_type.shape[:]

    # confidence
    ma_type = nms_spec.description.output[1].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = 80
    ma_type.shapeRange.sizeRanges[1].upperBound = 80
    del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.coordinatesInputFeatureName = "boxes"
    nms.confidenceInputFeatureName = "scores"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    default_iou_threshold = 0.5
    nms.iouThreshold = default_iou_threshold
    default_confidence_threshold = 0.7
    nms.confidenceThreshold = default_confidence_threshold
    nms.pickTop.perClass = True
    labels = load_class_names(namesfile)
    nms.stringClassLabels.vector.extend(labels)

    nms_model = coremltools.models.MLModel(nms_spec)
    nms_model.save(coreml_nms_filename)

    shutil.copy(coreml_nms_filename, coreml_nms_filename_in_yolo_ios)

    # pipeline
    input_features = [("image", datatypes.Array(3, 416, 416)),
                      ("iouThreshold", datatypes.Double()),
                      ("confidenceThreshold", datatypes.Double())]

    output_features = ["coordinates", "confidence", ]
    pipeline = Pipeline(input_features, output_features)

    pipeline.add_model(yolo_model)
    pipeline.add_model(nms_model)

    # configure in and output of pipeline
    pipeline.spec.description.input[0].ParseFromString(
        yolo_model._spec.description.input[0].SerializeToString()
    )
    pipeline.spec.description.output[0].ParseFromString(
        nms_model._spec.description.output[0].SerializeToString()
    )
    pipeline.spec.description.output[1].ParseFromString(
        nms_model._spec.description.output[1].SerializeToString()
    )

    user_defined_metadata = {
        "classes": ",".join(labels),
        "iou_threshold": str(default_iou_threshold),
        "confidence_threshold": str(default_confidence_threshold)
    }
    pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)
    pipeline.spec.specificationVersion = 3

    final_model = coremltools.models.MLModel(pipeline.spec)
    final_model.save(coreml_pipeline_filename)
    shutil.copy(coreml_pipeline_filename, coreml_pipeline_filename_in_yolo_ios)


def test_predict_with_pipeline():
    with open(
            os.path.join(os.environ['HOME'],
                         'datasets/detected-cars/more_than_4000_detected_per_make/makes.csv'), encoding="utf-8") as f:
        class_names = [make.strip() for make in f.readlines()]

    transform_resize = Compose([
        SquashResize(416),
    ])
    to_tensor = Compose([
        ToTensor()
    ])

    model = coremltools.models.MLModel(coreml_pipeline_filename)

    image = load_image_file(os.path.join(HERE, './images/car.jpg'))
    resized, _ = transform_resize(image, {})
    output = model.predict({'image': resized}, usesCPUOnly=True)

    coordinates = output["coordinates"]
    confidence = output["confidence"]
    detections = coreml_output_to_detections(coordinates, confidence)
    tensor_image, _ = to_tensor(image, {})
    plot_batch([detections], None, tensor_image.unsqueeze(0), class_names)




def test_convert_pytorch_to_coreml():
    with open(
            os.path.join(os.environ['HOME'],
                         'datasets/detected-cars/more_than_4000_detected_per_make/makes.csv'), encoding="utf-8") as f:
        class_names = [make.strip() for make in f.readlines()]

    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)

    convert_pytorch_to_coreml(cfg_file=os.path.join(HERE, '../cfg/yolov3.cfg'),
                              class_names=class_names,
                              state_dict_file=os.path.join(os.environ['HOME'],
                                                           'datasets/models/car_makes/yolo__num_classes_80__epoch_2_batch_7500.pt'),
                              dummy_input=dummy_input,
                              coreml_pipeline_filename=os.path.join(HERE, 'output/YoloPipelineCarMakes.mlmodel'))


def test_predict_cars_with_pipeline():
    image_and_target_transform = Compose([
        SquashResize(416),
        ToTensor()
    ])
    to_pil_image = transforms.Compose([
        ToPILImage()
    ])

    batch_size = 1
    model = coremltools.models.MLModel(os.path.join(HERE, 'output/YoloPipelineCarMakes.mlmodel'))

    dataset = DetectedCareMakeDataset(
        json_file=os.path.join(os.environ['HOME'],
                               'datasets/detected-cars/more_than_4000_detected_per_make/test.json'),
        transforms=image_and_target_transform, batch_size=batch_size)
    data_loader = DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)
    limit = 10

    for batch_i, (images, ground_truth_boxes, _) in enumerate(data_loader):

        pil_image = to_pil_image(images.squeeze(0))
        output = model.predict({'image': pil_image, 'confidenceThreshold': 0.4}, usesCPUOnly=True)
        print(output)

        coordinates = output["coordinates"]
        confidence = output["confidence"]
        detections = coreml_output_to_detections(coordinates, confidence)

        plot_batch([detections], None, images, dataset.class_names)
        if batch_i > limit:
            break

    print()


