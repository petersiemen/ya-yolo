import torch
import torch.nn as nn

from device import DEVICE


class ConvolutionalLayer(nn.Module):
    def __init__(self, layer_idx, in_channels, height, width, out_channels, kernel_size, stride, padding,
                 batch_normalize=1,
                 activation=None):
        super(ConvolutionalLayer, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_normalize = batch_normalize

        modules = nn.Sequential()

        if batch_normalize:
            modules.add_module(f"conv_{layer_idx}",
                               nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         bias=False)
                               )
            modules.add_module(f"batch_norm_{layer_idx}",
                               nn.BatchNorm2d(out_channels)
                               )
        else:
            modules.add_module(f"conv_{layer_idx}",
                               nn.Conv2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding)
                               )

        if activation == 'leaky':
            modules.add_module(f"relu_{layer_idx}",
                               nn.LeakyReLU(0.1, inplace=True))

        self.models = modules

    def __repr__(self):
        _, out_width, out_height = self.get_output_chw()
        return super(ConvolutionalLayer, self).__repr__() + " --  input: ({}x{}x{}), output: ({}x{}x{})".format(
            self.in_channels,
            self.height,
            self.width,
            self.out_channels,
            out_height,
            out_width
        )

    def get_output_chw(self):
        height = (self.height + 2 * self.padding - self.kernel_size) // self.stride + 1
        width = (self.width + 2 * self.padding - self.kernel_size) // self.stride + 1
        return self.out_channels, height, width

    def forward(self, x):
        for i, model in enumerate(self.models):
            x = model(x)
        return x


class ShortcutLayer(nn.Module):
    """
    It means the output of this layer is obtained by adding feature maps
    from the previous layer and the '_from' layer backwards from this layer

    Hence the values for width, height, channels are the same as the previous layer left them.
    """

    def __init__(self, _from, in_channels, height, width):
        super(ShortcutLayer, self).__init__()
        self._from = _from
        self.in_channels = in_channels
        self.height = height
        self.width = width

    def __repr__(self):
        out_channels, out_height, out_width, = self.get_output_chw()
        return "ShortcutLayer(_from: {} and (-1), input: ({}x{}x{}), output: ({}x{}x{}))".format(self._from,
                                                                                                 self.in_channels,
                                                                                                 self.width,
                                                                                                 self.height,
                                                                                                 out_channels,
                                                                                                 out_height,
                                                                                                 out_width
                                                                                                 )

    def get_output_chw(self):
        return self.in_channels, self.height, self.width

    def forward(self, x):
        """
        forward needs to be implemented by the network that uses this layer by
        adding up the output from the previous layer and the layer 'current_id + self._from'
        :param x:
        :return:
        """
        return x


class RouteLayer(nn.Module):
    def __init__(self, layer_1, in_channels_layer_1, height, width, layer_2=None,
                 in_channels_layer_2=None):
        super(RouteLayer, self).__init__()
        self.layer_1 = layer_1
        self.in_channels_layer_1 = in_channels_layer_1
        self.height = height
        self.width = width
        self.layer_2 = layer_2
        self.in_channels_layer_2 = in_channels_layer_2

    def __repr__(self):
        out_channels, out_height, out_width = self.get_output_chw()
        return "RouteLayer(layer_1: {}, layer_2: {}, input: ({}x{}x{}), output: ({}x{}x{}))".format(self.layer_1,
                                                                                                    self.layer_2,
                                                                                                    self.in_channels_layer_1,
                                                                                                    self.height,
                                                                                                    self.width,
                                                                                                    out_channels,
                                                                                                    out_height,
                                                                                                    out_width
                                                                                                    )

    def get_output_chw(self):
        if self.layer_2 is None:
            return self.in_channels_layer_1, self.height, self.width
        else:
            return self.in_channels_layer_1 + self.in_channels_layer_2, self.height, self.width

    def forward(self, x):
        """
        needs to be implemented by the network that uses this layer by
        concatenating the output of the layers indentified by 'self.layer_1' and 'self.layer_2' along the Channels dimension
        Same as in the Shortcut layer these indices count backwards from the current position
        :param x:
        :return:
        """
        return x


class UpsampleLayer(nn.Module):
    def __init__(self, scale_factor, in_channels, height, width):
        assert height == width, "Can only operate on square images. {}x{} is not square".format(height, width)
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.model = nn.Upsample(size=self.width * self.scale_factor)
        # self.model = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # ws = hs = self.scale_factor
        # B = x.data.size(0)
        # C = x.data.size(1)
        # H = x.data.size(2)
        # W = x.data.size(3)
        # x = x.view(B, C, H, 1, W, 1).expand(B, C, H, hs, W, ws).contiguous().view(B, C, H * hs, W * ws)

        return self.model(x)
        # return x

    def __repr__(self):
        out_channels, out_height, out_width = self.get_output_chw()
        return "UpsampleLayer(scale_factor: {}, input: ({}x{}x{}), output: ({}x{}x{}))".format(self.scale_factor,
                                                                                               self.in_channels,
                                                                                               self.height,
                                                                                               self.width,
                                                                                               out_channels,
                                                                                               out_height,
                                                                                               out_width
                                                                                               )

    def get_output_chw(self):
        return self.in_channels, self.height * self.scale_factor, self.width * self.scale_factor


class EagerYoloLayer(nn.Module):
    def __init__(self, batch_size, anchors, num_classes, in_channels, height, width):
        super(EagerYoloLayer, self).__init__()
        self.batch_size = batch_size
        self.anchors = anchors
        self.num_anchors = int(len(anchors) / 2)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.height = height
        self.width = width

        self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = self.create_grid()

    def __repr__(self):
        out_channels, out_height, out_width = self.get_output_chw()
        return "EagerYoloLayer(num_classes: {}, anchors: {}, input: ({}x{}x{})))".format(self.num_classes,
                                                                                         self.anchors,
                                                                                         self.in_channels,
                                                                                         self.height,
                                                                                         self.width)

    def create_grid(self):
        """
        Creates grid (grid_x,grid_y,anchor_w,anchor_h) for a given grid_size and anchors
        :return:
        """
        w = self.width
        h = self.height
        batch_size = self.batch_size
        anchor_step = 2
        num_anchors = self.num_anchors
        anchors = self.anchors
        grid_x = torch.linspace(0, w - 1, w).repeat(h, 1).repeat(batch_size * num_anchors, 1, 1).view(
            batch_size * num_anchors * h * w).to(DEVICE)
        grid_y = torch.linspace(0, h - 1, h).repeat(w, 1).t().repeat(batch_size * num_anchors, 1, 1).view(
            batch_size * num_anchors * h * w).to(DEVICE)

        anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, h * w).view(batch_size * num_anchors * h * w).to(
            DEVICE)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, h * w).view(batch_size * num_anchors * h * w).to(
            DEVICE)

        return grid_x, grid_y, anchor_w, anchor_h

    def get_output_chw(self):
        return self.in_channels, self.height, self.width

    def get_output_tensor_length(self):
        return self.batch_size * self.in_channels * self.height * self.width

    def forward(self, x):
        output = x.view(self.batch_size * self.num_anchors, 5 + self.num_classes, self.height * self.width) \
            .transpose(0, 1) \
            .contiguous() \
            .view(5 + self.num_classes,
                  self.batch_size * self.num_anchors * self.height * self.width)

        xs = (torch.sigmoid(
            output[0]) + self.grid_x) / self.width
        ys = (torch.sigmoid(output[1]) + self.grid_y) / self.height
        ws = (torch.exp(output[2]) * self.anchor_w) / self.width
        hs = (torch.exp(output[3]) * self.anchor_h) / self.height
        det_confs = torch.sigmoid(output[4])

        coordinates = torch.cat((xs.view(self.batch_size, self.num_anchors * self.height * self.width, 1),
                                 ys.view(self.batch_size, self.num_anchors * self.height * self.width, 1),
                                 ws.view(self.batch_size, self.num_anchors * self.height * self.width, 1),
                                 hs.view(self.batch_size, self.num_anchors * self.height * self.width, 1)), 2)

        # according to YOLOv3: An Incremental Improvement
        #
        # 2.2. Class Prediction
        # Each box predicts the classes the bounding box may contain using multilabel classification. We do not use a softmax
        # as we have found it is unnecessary for good performance, instead we simply use independent logistic classifiers.
        # During training we use binary cross-entropy loss for the class predictions.
        #
        # note: we do not compute the sigmoid on the class_scores here but rather use torch.nn.BCEWithLogitsLoss() since
        # it is numerically more stable than computing sigmoid separately and then using torch.nn.BCELoss()
        class_scores = output[5:5 + self.num_classes].transpose(0, 1) \
            .view(self.batch_size, self.num_anchors * self.height * self.width,
                  self.num_classes)

        return coordinates, class_scores, det_confs.view(self.batch_size, self.num_anchors * self.height * self.width)
