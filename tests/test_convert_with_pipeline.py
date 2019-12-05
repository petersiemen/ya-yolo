from .context import *
import shutil
from coremltools.models.neural_network import flexible_shape_utils
import coremltools

HERE = os.path.dirname(os.path.realpath(__file__))
cfg_file = os.path.join(HERE, '../cfg/mini-yolov3.cfg')
namesfile = os.path.join(HERE, '../cfg/coco.names')
onnx_filename = os.path.join(HERE, 'output/mini_yolo.onnx')
coreml_filename = os.path.join(HERE, 'output/MiniYolo.mlmodel')
coreml_nms_filename = os.path.join(HERE, 'output/NMS.mlmodel')
coreml_nms_filename_in_yolo_ios= os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/NMS.mlmodel')

coreml_filename_in_yolo_ios= os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/MiniYolo.mlmodel')

core_ml_with_nms_filename = os.path.join(HERE, 'output/MiniYoloNms.mlmodel')
core_ml_with_nms_filename_in_yolo_ios = os.path.join(HERE, '../../yolo-ios/YoloIOS/Models/MiniYoloNms.mlmodel')


def test_convert_to_onnx():

    # yolo model
    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)
    yolo = Yolo(cfg_file=cfg_file, namesfile=namesfile, batch_size=1, coreml_mode=True)
    pytorch_to_onnx(yolo, dummy_input, onnx_filename)

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

    yolo_model = coremltools.models.MLModel(coreml_model.get_spec())
    yolo_model.save(coreml_filename)
    shutil.copy(coreml_filename, coreml_filename_in_yolo_ios)



    # nms model
    nms_spec = coremltools.proto.Model_pb2.Model()
    nms_spec.specificationVersion = 3
    #boxes
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

    #coordinates
    ma_type = nms_spec.description.output[0].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = 4
    ma_type.shapeRange.sizeRanges[1].upperBound = 4
    del ma_type.shape[:]

    #confidence
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
    #nms.iouThresholdInputFeatureName = "iouThreshold"
    #nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = 0.5
    nms.confidenceThreshold = 0.7
    nms.pickTop.perClass = True
    labels = load_class_names(namesfile)
    nms.stringClassLabels.vector.extend(labels)

    nms_model = coremltools.models.MLModel(nms_spec)
    nms_model.save(coreml_nms_filename)

    shutil.copy(coreml_nms_filename, coreml_nms_filename_in_yolo_ios)
