import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_cfg(cfg_file):
    blocks = []
    fp = open(cfg_file, 'r')
    block = None
    line = fp.readline()
    while line != '':
        line = line.rstrip()
        if line == '' or line[0] == '#' or line[0] == ';':
            line = fp.readline()
            continue
        elif line[0] == '[':
            if block:
                blocks.append(block)
            block = dict()
            block['type'] = line.lstrip('[').rstrip(']')
            # set default value
            if block['type'] == 'convolutional':
                block['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            key = key.strip()
            if key == 'type':
                key = '_type'
            value = value.strip()
            block[key] = value
        line = fp.readline()

    if block:
        blocks.append(block)
    fp.close()
    return blocks


def print_cfg(blocks):
    print('layer     filters    size              input                output')
    prev_width = 416
    prev_height = 416
    prev_filters = 3
    out_filters = []
    out_widths = []
    out_heights = []
    ind = -2
    for block in blocks:
        ind = ind + 1
        if block['type'] == 'net':
            prev_width = int(block['width'])
            prev_height = int(block['height'])
            continue
        elif block['type'] == 'convolutional':
            filters = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            is_pad = int(block['pad'])
            pad = (kernel_size - 1) // 2 if is_pad else 0
            width = (prev_width + 2 * pad - kernel_size) // stride + 1
            height = (prev_height + 2 * pad - kernel_size) // stride + 1
            print('%5d %-6s %4d  %d x %d / %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'conv', filters, kernel_size, kernel_size, stride, prev_width, prev_height, prev_filters, width,
                height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'upsample':
            stride = int(block['stride'])
            filters = prev_filters
            width = prev_width * stride
            height = prev_height * stride
            print('%5d %-6s           * %d   %3d x %3d x%4d   ->   %3d x %3d x%4d' % (
                ind, 'upsample', stride, prev_width, prev_height, prev_filters, width, height, filters))
            prev_width = width
            prev_height = height
            prev_filters = filters
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                print('%5d %-6s %d' % (ind, 'route', layers[0]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                prev_filters = out_filters[layers[0]]
            elif len(layers) == 2:
                print('%5d %-6s %d %d' % (ind, 'route', layers[0], layers[1]))
                prev_width = out_widths[layers[0]]
                prev_height = out_heights[layers[0]]
                assert (prev_width == out_widths[layers[1]])
                assert (prev_height == out_heights[layers[1]])
                prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] in ['region', 'yolo']:
            print('%5d %-6s' % (ind, 'detection'))
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        elif block['type'] == 'shortcut':
            from_id = int(block['from'])
            from_id = from_id if from_id > 0 else from_id + ind
            print('%5d %-6s %d' % (ind, 'shortcut', from_id))
            prev_width = out_widths[from_id]
            prev_height = out_heights[from_id]
            prev_filters = out_filters[from_id]
            out_widths.append(prev_width)
            out_heights.append(prev_height)
            out_filters.append(prev_filters)
        else:
            print('unknown type %s' % (block['type']))


def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()
    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data))
    start = start + num_w
    return start


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()
    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight.data))
    start = start + num_w
    return start


def load_class_names(namesfile):
    # Create an empty list to hold the object classes
    class_names = []

    # Open the file containing the COCO object classes in read-only mode
    with open(namesfile, 'r') as fp:
        # The coco.names file contains only one object class per line.
        # Read the file line by line and save all the lines in a list.
        lines = fp.readlines()

    # Get the object class names
    for line in lines:
        # Make a copy of each line with any trailing whitespace removed
        line = line.rstrip()

        # Save the object class name into class_names
        class_names.append(line)

    return class_names


def plot_boxes(img, boxes, class_names, plot_labels, color=None):
    # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))

        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]

        return int(r * 255)

    # Get the width and height of the image
    width = img.width
    height = img.height

    # Create a figure and plot the image
    fig, a = plt.subplots(1, 1)
    a.imshow(img)

    # Plot the bounding boxes and corresponding labels on top of the image
    for i in range(len(boxes)):

        # Get the ith bounding box
        box = boxes[i]

        # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
        # of the bounding box relative to the size of the image.
        x1 = int(np.around((box[0] - box[2] / 2.0) * width))
        y1 = int(np.around((box[1] - box[3] / 2.0) * height))
        x2 = int(np.around((box[0] + box[2] / 2.0) * width))
        y2 = int(np.around((box[1] + box[3] / 2.0) * height))

        # Set the default rgb value to red
        rgb = (1, 0, 0)

        # Use the same color to plot the bounding boxes of the same object class
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue = get_color(0, offset, classes) / 255

            # If a color is given then set rgb to the given color instead
            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color

        # Calculate the width and height of the bounding box relative to the size of the image.
        width_x = x2 - x1
        width_y = y1 - y2

        # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
        # lower-left corner of the bounding box relative to the size of the image.
        rect = patches.Rectangle((x1, y2),
                                 width_x, width_y,
                                 linewidth=2,
                                 edgecolor=rgb,
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)

        # If plot_labels = True then plot the corresponding label
        if plot_labels:
            # Create a string with the object class name and the corresponding object class probability
            conf_tx = class_names[int(cls_id)] + ': {:.1f}'.format(cls_conf)

            # Define x and y offsets for the labels
            lxc = (width * 0.266) / 100
            lyc = (height * 1.180) / 100

            # Draw the labels on top of the image
            a.text(x1 + lxc, y1 - lyc, conf_tx, fontsize=24, color='k',
                   bbox=dict(facecolor=rgb, edgecolor=rgb, alpha=0.8))

    plt.show()


def nms_for_coordinates_and_class_scores_and_confidence(coordinates,
                                                        class_scores,
                                                        confidence,
                                                        iou_thresh,
                                                        score_threshold):
    assert coordinates.size(0) == confidence.size(0), "coordinates and objectnesses must match size in dimension 1"
    assert coordinates.size(0) == class_scores.size(0), "coordinates and objectnesses must match size in dimension 1"

    # Get the detection confidence of each predicted bounding box
    cls_max_confs, cls_max_ids = torch.max(class_scores, 1)

    # Sort the indices of the bounding boxes by detection confidence value in descending order.
    # We ignore the first returned element since we are only interested in the sorted indices
    # sorted_cnfs, sortIds = torch.sort(cls_max_confs, descending=True)
    sorted_cnfs, sortIds = torch.sort(confidence, descending=True)

    # Create an empty list to hold the best bounding boxes after
    # Non-Maximal Suppression (NMS) is performed
    best_boxes = []
    # Perform Non-Maximal Suppression
    for i in range(len(sortIds)):

        # Get the bounding box with the highest detection confidence first
        box_i = coordinates[sortIds[i]].detach().numpy().tolist()
        det_conf = confidence[sortIds[i]].detach().item()
        class_id = cls_max_ids[sortIds[i]].detach().item()
        class_cnf = cls_max_confs[sortIds[i]].detach().item()
        box_i.append(det_conf)
        box_i.append(class_cnf)
        box_i.append(class_id)
        box_i.append(sortIds[i].detach().item())

        # Check that the detection confidence is not zero
        if det_conf > score_threshold:
            # Save the bounding box
            best_boxes.append(box_i)

            # Go through the rest of the bounding boxes in the list and calculate their IOU with
            # respect to the previous selected box_i.
            for j in range(i + 1, len(sortIds)):
                box_j = coordinates[sortIds[j]]

                # If the IOU of box_i and box_j is higher than the given IOU threshold set
                # box_j's detection confidence to zero.
                if boxes_iou(box_i, box_j) > iou_thresh:
                    confidence[sortIds[j]] = 0

    return best_boxes


def boxes_iou(box1, box2):
    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]

    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1 / 2.0, box2[0] - width_box2 / 2.0)
    Mx = max(box1[0] + width_box1 / 2.0, box2[0] + width_box2 / 2.0)

    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx

    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1 / 2.0, box2[1] - height_box2 / 2.0)
    My = max(box1[1] + height_box1 / 2.0, box2[1] + height_box2 / 2.0)

    # Calculate the height of the union of the two bounding boxes
    union_height = My - my

    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height

    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height

    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate the IOU
    iou = intersection_area / union_area

    return iou
