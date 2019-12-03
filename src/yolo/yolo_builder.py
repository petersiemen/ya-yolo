from yolo.layers import *

from yolo.utils import parse_cfg


class YoloBuilder:

    @staticmethod
    def run(cfg_file, batch_size, coreml_mode=False):
        blocks = parse_cfg(cfg_file)

        grid_sizes = []
        models = nn.ModuleList()
        start_width = start_height = width = height = channels = None
        for idx, block in enumerate(blocks):
            if block['type'] == 'net':
                start_width = width = int(block['width'])
                start_height = height = int(block['height'])
                assert start_width == start_height, "Can only operate on square images. ({}x{}) is not square".format(
                    start_width, start_height)
                channels = 3
            elif block['type'] == 'convolutional':
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                padding = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                layer = ConvolutionalLayer(layer_idx=idx-1, in_channels=channels, height=height, width=width,
                                           out_channels=filters, kernel_size=kernel_size, stride=stride,
                                           padding=padding, batch_normalize=batch_normalize, activation=activation
                                           )
                channels, height, width = layer.get_output_chw()
                models.append(layer)


            elif block['type'] == 'shortcut':
                _from = int(block['from'])
                layer = ShortcutLayer(_from=_from, in_channels=channels, height=height, width=width)
                channels, height, width = layer.get_output_chw()
                models.append(layer)
            elif block['type'] == 'route':
                layers = block['layers'].split(",")
                if len(layers) == 1:
                    layer_1 = int(layers[0])
                    channels, height, width = models[layer_1].get_output_chw()
                    layer = RouteLayer(layer_1=layer_1, in_channels_layer_1=channels, height=height, width=width)
                elif len(layers) == 2:
                    layer_1 = int(layers[0])
                    layer_2 = int(layers[1])
                    channels_layer_1, height_1, width_1 = models[layer_1].get_output_chw()
                    channels_layer_2, height_2, width_2 = models[layer_2].get_output_chw()
                    assert width_1 == width_2, "Cannot create RouteLayer from 2 layers ({},{}) with different widths ({},{})".format(
                        layer_1, layer_2, width_1, width_2)
                    assert height_1 == height_2, "Cannot create RouteLayer from 2 layers ({},{}) with different heights ({},{})".format(
                        layer_1, layer_2, height_1, height_2)
                    layer = RouteLayer(layer_1=layer_1, in_channels_layer_1=channels_layer_1, height=height,
                                       width=width,
                                       layer_2=layer_2, in_channels_layer_2=channels_layer_2)
                else:
                    raise Exception('Cannot build RouteLayer from {}\n{}'.format(layers, block))
                channels, height, width = layer.get_output_chw()
                models.append(layer)
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                layer = UpsampleLayer(scale_factor=stride, in_channels=channels, height=height, width=width)
                channels, height, width = layer.get_output_chw()
                models.append(layer)
            elif block['type'] == 'yolo':
                anchors = block['anchors'].split(',')
                anchor_mask = [int(i) for i in block['mask'].split(',')]
                anchors = [float(i) for i in anchors]
                num_anchors = int(block['num'])
                num_classes = int(block['classes'])
                anchor_step = len(anchors) // num_anchors
                _, prev_height, prev_width = models[-1].get_output_chw()
                stride = start_width / prev_width
                masked_anchors = []
                for m in anchor_mask:
                    masked_anchors += anchors[m * anchor_step:(m + 1) * anchor_step]
                masked_anchors = [anchor / stride for anchor in masked_anchors]

                assert width == height, "Can only create EagerYoloLayers for square grids. ({}x{}) is not square".format(
                    width, height)
                grid_sizes.append(width)

                layer = EagerYoloLayer(batch_size=batch_size, anchors=masked_anchors, num_classes=num_classes,
                                       in_channels=channels,
                                       width=width, height=height, coreml_mode=coreml_mode)
                models.append(layer)

        return models, grid_sizes
