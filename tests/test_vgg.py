from torchvision import models
import torch
import os
import torch.onnx
import onnx
import coremltools
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

from .context import to_pil_image
from .context import load_class_names

HERE = os.path.dirname(os.path.realpath(__file__))
vgg_onnx_filename = os.path.join(HERE, 'output/vgg.onnx')
vgg_coreml_filename = os.path.join(HERE, 'output/vgg.mlmodel')


class FixedSizeAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride, padding=0):
        super().__init__()
        self.p = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.p(x)


def test_convert_vgg_to_onnx():
    vgg16 = models.vgg16(pretrained=True)
    print(vgg16)

    vgg16.avgpool = FixedSizeAvgPool2d(kernel_size=(7, 7), stride=1, padding=3)
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)

    torch.onnx.export(vgg16,
                      dummy_input,
                      vgg_onnx_filename,
                      verbose=True,
                      input_names=["image"],
                      output_names=["predictions"])


def test_convert_onnx_to_coreml():
    onnx_model = onnx.load(vgg_onnx_filename)
    onnx.checker.check_model(onnx_model)
    labels = load_class_names(
        os.path.join(os.environ['HOME'], 'workspace/ya-yolo/src/datasets/imagenet1000_clsidx_to_labels.txt'))

    from onnx_coreml import convert

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


    vgg_model = coremltools.models.MLModel(coreml_model.get_spec())
    vgg_model._spec.neuralNetworkClassifier.stringClassLabels.vector.extend(labels)
    vgg_model.save(vgg_coreml_filename)


def test_adaptive_average_pooling():
    m = nn.AdaptiveAvgPool2d((7, 7))
    input = torch.randn(1, 512, 13, 13)
    output = m(input)
    print(output.shape)

    my = FixedSizeAvgPool2d(kernel_size=(7, 7), stride=1)
    my_output = my(input)
    print(my_output.shape)


def test_adaptive_average_pooling_with_512_7_7():
    m = nn.AdaptiveAvgPool2d((7, 7))
    input = torch.randn(1, 512, 7, 7)
    output = m(input)
    print(output.shape)

    my = FixedSizeAvgPool2d(kernel_size=(7, 7), stride=1, padding=3)
    my_output = my(input)
    print(my_output.shape)


def test_detect_flowers():
    test_dir = os.path.join(os.environ['HOME'], 'datasets/flower_photos')
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.ToTensor()])
    test_data = datasets.ImageFolder(test_dir, transform=data_transform)
    batch_size = 1
    num_workers = 0
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)

    vgg16 = models.vgg16(pretrained=True)
    print(vgg16)

    vgg16.avgpool = FixedSizeAvgPool2d(kernel_size=(7, 7), stride=1, padding=3)
    # iterate over test data
    limit = 5
    for i, (data, target) in enumerate(test_loader):
        output = vgg16(data)

        pil_image = to_pil_image(data.squeeze(0))
        plt.imshow(pil_image)
        plt.show()

        if i > limit:
            break
