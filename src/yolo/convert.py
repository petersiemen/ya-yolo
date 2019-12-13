import torch.onnx
import os
import tempfile

import coremltools
import onnx
import torch.onnx
from coremltools.models import datatypes
from coremltools.models.pipeline import Pipeline
from coremltools.models.utils import save_spec
from onnx_coreml import convert

from logging_config import *
from yolo.utils import load_class_names
from yolo.yolo import Yolo
from device import DEVICE

logger = logging.getLogger(__name__)


def pytorch_to_onnx(model, dummy_input, onnx_filename):
    torch.onnx.export(model,
                      dummy_input,
                      onnx_filename,
                      verbose=True,
                      input_names=["image"],
                      output_names=["boxes", "scores"])
    onnx_model = onnx.load(onnx_filename)
    #onnx.checker.check_model(onnx_model)


def onnx_to_coreml(onnx_filename, core_ml_filename):
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
    builder = coremltools.models.neural_network.NeuralNetworkBuilder(spec=coreml_model.get_spec())

    save_spec(builder.spec, core_ml_filename)

    coremltools.models.neural_network.printer.print_network_spec(builder.spec)


def convert_pytorch_to_coreml(cfg_file, class_names, state_dict_file, dummy_input, coreml_pipeline_filename):
    _path = os.path.dirname(coreml_pipeline_filename)
    assert os.path.isdir(_path), f"path {_path} does not exist"

    if os.path.exists(coreml_pipeline_filename):
        logger.warning(f'file {coreml_pipeline_filename} exists.')

    HERE = os.path.dirname(os.path.realpath(__file__))

    _onnx_filename = os.path.join(HERE, '../../tests/output/yolo-cars.onnx') #tempfile.NamedTemporaryFile(prefix='onny-yolo')
    _coreml_filename = os.path.join(HERE, '../../tests/output/yolo-cars.mlmodel') #tempfile.NamedTemporaryFile(prefix='coreml-yolo')
    _coreml_nms_filename =os.path.join(HERE, '../../tests/output/yolo-cars-nms.mlmodel')# tempfile.NamedTemporaryFile(prefix='coreml-nms')

    yolo = Yolo(cfg_file=cfg_file, class_names=class_names, batch_size=1, coreml_mode=True)

    yolo.load_state_dict(
        torch.load(state_dict_file,
                   map_location=DEVICE))

    pytorch_to_onnx(yolo, dummy_input, _onnx_filename)

    onnx_model = onnx.load(_onnx_filename)
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

    yolo_model.save(_coreml_filename)

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
    ma_type.shapeRange.sizeRanges[1].lowerBound = len(class_names)
    ma_type.shapeRange.sizeRanges[1].upperBound = len(class_names)
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
    nms.stringClassLabels.vector.extend(class_names)

    nms_model = coremltools.models.MLModel(nms_spec)
    nms_model.save(_coreml_nms_filename)

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
        "classes": ",".join(class_names),
        "iou_threshold": str(default_iou_threshold),
        "confidence_threshold": str(default_confidence_threshold)
    }
    pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)
    pipeline.spec.specificationVersion = 3

    final_model = coremltools.models.MLModel(pipeline.spec)
    final_model.save(coreml_pipeline_filename)
