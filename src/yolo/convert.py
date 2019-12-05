import onnx
from onnx import onnx_pb
from onnx_coreml import convert
import torch.onnx
import coremltools
from coremltools.models import MLModel, datatypes
from coremltools.models.pipeline import Pipeline
import onnx
from onnx import onnx_pb
from onnx_coreml import convert
from coremltools.models.utils import save_spec


def pytorch_to_onnx(model, dummy_input, onnx_filename):
    torch.onnx.export(model,
                      dummy_input,
                      onnx_filename,
                      verbose=True,
                      input_names=["image"],
                      output_names=["boxes", "scores"])
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)


def onnx_to_coreml(onnx_filename, core_ml_filename):
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

    save_spec(builder.spec, core_ml_filename)

    coremltools.models.neural_network.printer.print_network_spec(builder.spec)
