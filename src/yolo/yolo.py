import numpy as np

import os
from yolo.layers import *
from yolo.utils import load_conv, load_conv_bn
from yolo.yolo_builder import YoloBuilder
from yolo.utils import load_class_names
from logging_config import *

logger = logging.getLogger(__name__)


class Yolo(nn.Module):
    def __init__(self, cfg_file, class_names, batch_size, coreml_mode=False):
        super(Yolo, self).__init__()
        self.models, self.grid_sizes = YoloBuilder.run(cfg_file, batch_size, coreml_mode)
        self.output_tensor_length = self.get_output_tensor_length()
        self.coreml_mode = coreml_mode

        self.class_names = class_names
        self.num_classes = len(class_names)
        self._adjust_layers_to_num_classes()

    def freeze_parameters(self):
        yolo_indices = [idx for idx, model in enumerate(self.models) if isinstance(model, EagerYoloLayer)]
        conv_before_yolo = [idx -1 for idx in yolo_indices]

        for idx in range(len(self.models)):
            if idx in yolo_indices or idx in conv_before_yolo:
                logger.info(f"Not freezing layer {idx}. Either its a yolo layer or a convolution before")
            else:
                for param in self.models[idx].parameters():
                    param.requires_grad = False


    def get_trainable_parameters(self):
        yolo_indices = [idx for idx, model in enumerate(self.models) if isinstance(model, EagerYoloLayer)]
        for idx in yolo_indices:
            yield from self.models[idx - 1].parameters()

    def _adjust_layers_to_num_classes(self):
        num_classes = len(self.class_names)
        logger.info(f"Update num_classes to {num_classes}")

        outchannels_of_conv_layer_as_input_to_yolo = 3 * (num_classes + 5)
        yolo_indices = [idx for idx, model in enumerate(self.models) if isinstance(model, EagerYoloLayer)]

        for idx in yolo_indices:
            old = self.models[idx - 1]
            # update num_classes and in_channels in EagerYoloLayer
            self.models[idx].num_classes = num_classes
            self.models[idx].in_channels = outchannels_of_conv_layer_as_input_to_yolo

            # update convolutional layer feeding into EagerYoloLayer
            self.models[idx - 1] = ConvolutionalLayer(layer_idx=old.layer_idx,
                                                      in_channels=old.in_channels,
                                                      height=old.height,
                                                      width=old.width,
                                                      out_channels=outchannels_of_conv_layer_as_input_to_yolo,
                                                      kernel_size=old.kernel_size,
                                                      stride=old.stride,
                                                      padding=old.padding,
                                                      batch_normalize=old.batch_normalize,
                                                      activation=old.activation)

    def save(self, dir, name):
        torch.save(self.state_dict(), os.path.join(dir, name))

    def reload(self, dir, name):
        self.load_state_dict(torch.load(os.path.join(dir, name)))

    def get_output_tensor_length(self):
        return np.sum([
            model.get_output_tensor_length()
            for _, model in enumerate(self.models) if isinstance(model, EagerYoloLayer)])

    def forward(self, x):
        outputs = dict()
        coordinates = []
        class_scores = []
        confidence = []

        for idx, model in enumerate(self.models):
            if isinstance(model, ConvolutionalLayer) or isinstance(model, UpsampleLayer):
                x = model(x)
                outputs[idx] = x
            elif isinstance(model, RouteLayer):
                layer_1 = model.layer_1
                layer_2 = model.layer_2
                if layer_2 is None:
                    x = outputs[idx + layer_1]
                    outputs[idx] = model(x)
                else:
                    x1 = outputs[idx + layer_1]
                    x2 = outputs[layer_2]
                    x = torch.cat((x1, x2), 1)
                    outputs[idx] = x
            elif isinstance(model, ShortcutLayer):
                _from = model._from
                x1 = outputs[idx + _from]
                x2 = outputs[idx - 1]
                x = x1 + x2
                x = model(x)
                outputs[idx] = x
            elif isinstance(model, EagerYoloLayer):
                if model.coreml_mode:
                    co, cs = model(x)
                    coordinates.append(co)
                    class_scores.append(cs)
                else:
                    co, cs, pc = model(x)
                    coordinates.append(co)
                    class_scores.append(cs)
                    confidence.append(pc)

        if self.coreml_mode:
            return torch.cat(coordinates, 1).to(dtype=torch.double).view(-1, 4), \
                   torch.cat(class_scores, 1).to(dtype=torch.double).view(-1, self.num_classes)#, \
                   #torch.cat(confidence, 1).to(dtype=torch.double)  # .to(dtype=torch.double)
        else:
            return torch.cat(coordinates, 1), \
                   torch.cat(class_scores, 1), \
                   torch.cat(confidence, 1)

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        print('Loading weights. Please Wait...\n')
        start = 0
        for idx, model in enumerate(self.models):
            if start >= buf.size:
                break
            if isinstance(model, ConvolutionalLayer):
                if model.batch_normalize:
                    start = load_conv_bn(buf, start, model.models[0], model.models[1])
                else:
                    start = load_conv(buf, start, model.models[0])

    def print_network(self):
        print('\nlayer     filters    size              input                output')
        for idx, model in enumerate(self.models):
            if isinstance(model, ConvolutionalLayer):
                ind = idx
                kernel_size = model.kernel_size
                stride = model.stride
                prev_width = model.width
                prev_height = model.height
                prev_filters = model.in_channels
                filters, height, width = model.get_output_chw()
                print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                    ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters,
                    width, height, filters))
            elif isinstance(model, UpsampleLayer):
                ind = idx
                stride = model.scale_factor
                prev_filters = model.in_channels
                prev_width = model.width
                prev_height = model.height
                filters, height, width = model.get_output_chw()
                print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                    ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            elif isinstance(model, RouteLayer):
                ind = idx
                layer_1 = model.layer_1 + ind
                layer_2 = model.layer_2
                if layer_2 is None:
                    print('%5d %-6s %d' % (ind, 'route', layer_1))
                else:
                    print('%5d %-6s %d %d' % (ind, 'route', layer_1, layer_2))
            elif isinstance(model, ShortcutLayer):
                ind = idx
                from_id = model._from
                from_id = from_id if from_id > 0 else from_id + ind
                print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            elif isinstance(model, EagerYoloLayer):
                ind = idx
                anchors = model.anchors
                width = model.width
                height = model.height
                channels = model.in_channels
                print('{} {} {}  {}x{}x{}'.format(ind, 'detection', anchors, width, height, channels))
            else:
                raise Exception('Unknown layer', model)
