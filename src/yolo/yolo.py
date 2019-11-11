import numpy as np

from yolo.layers import *
from yolo.utils import load_conv, load_conv_bn
from yolo.yolo_builder import YoloBuilder


class Yolo(nn.Module):
    def __init__(self, cfg_file, batch_size=1):
        super(Yolo, self).__init__()
        self.models, self.grid_sizes = YoloBuilder.run(cfg_file, batch_size)
        self.output_tensor_length = self.get_output_tensor_length()

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
                co, cs, pc = model(x)
                coordinates.append(co)
                class_scores.append(cs)
                confidence.append(pc)

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
