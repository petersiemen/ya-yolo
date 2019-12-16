import torch
import torch.onnx
import os
import onnx
import coremltools
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
resnet_onnx_filename =  os.path.join(HERE, 'output/resnet.onnx')
resnet_coreml_filename =  os.path.join(HERE, 'output/resnet.mlmodel')

def test_resnet_to_onnx():
    model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)

    print(model)

    dummy_input = torch.randn(1, 3, 416, 416, requires_grad=True)
    dummy_input = torch.randn(256, 2048, 1, 1,requires_grad=True)

    torch.onnx.export(model.classifier,
                      dummy_input,
                      resnet_onnx_filename,
                      verbose=True,
                      input_names=["image"],
                      output_names=["predictions"])

def test_resnet_to_coreml():
    onnx_model = onnx.load(resnet_onnx_filename)
    onnx.checker.check_model(onnx_model)

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

    yolo_model = coremltools.models.MLModel(coreml_model.get_spec())

    yolo_model.save(resnet_coreml_filename)


def test_resnet_run():
    import torch
    model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)
    model.eval()

    import urllib
    url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try:
        urllib.URLopener().retrieve(url, filename)
    except:
        urllib.request.urlretrieve(url, filename)

    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    plt.imshow(input_image)
    plt.show()
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r.putpalette(colors)


    plt.imshow(r)
    plt.show()
