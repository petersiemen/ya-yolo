from .context import *

HERE = os.path.dirname(os.path.realpath(__file__))


def test_convert():
    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)
    onnx_filename = os.path.join(HERE, 'output/yolo.onnx')
    coreml_filename = os.path.join(HERE, 'output/yolo.mlmodel')

    cfg_file = os.path.join(HERE, '../cfg/yolov3.cfg')
    weight_file = os.path.join(HERE, '../cfg/yolov3.weights')
    yolo = Yolo(cfg_file=cfg_file, batch_size=1)
    yolo.load_weights(weight_file)
    pytorch_to_onnx(yolo, dummy_input, onnx_filename)

    onnx_to_coreml(onnx_filename, coreml_filename)

