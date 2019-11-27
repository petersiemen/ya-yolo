import matplotlib.patches as patches
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import ToPILImage

from logging_config import *
from metrics.utils import *

logger = logging.getLogger(__name__)

to_pil_image = transforms.Compose([
    ToPILImage()
])


def _process_batch(detections_for_batch, ground_truth_boxes_for_batch, images, class_names, processor,
                   image_paths=None):
    batch_size = len(ground_truth_boxes_for_batch)
    for b_i in range(batch_size):
        pil_image = to_pil_image(images[b_i].cpu())
        detections_for_image = detections_for_batch[b_i]
        ground_truth_boxes_for_image = ground_truth_boxes_for_batch[b_i]

        fig, a = _create_fig_from_pil_image(pil_image, detections_for_image, ground_truth_boxes_for_image, class_names)
        if image_paths is not None:
            processor(fig, a, image_paths[b_i])
        else:
            processor(fig, a)


def _create_fig_from_pil_image(pil_image, detected_boxes, ground_truth_boxes, class_names):
    """
     0  1  2  3  4          5          6
    [x, y, w, h, det_conf,  cls_conf,  cls_id]
    """

    # Get the width and height of the image
    width = pil_image.width
    height = pil_image.height

    # Create a figure and plot the image
    fig, a = plt.subplots(1, 1)
    a.imshow(pil_image)

    for i in range(len(ground_truth_boxes)):
        _plot_rect_on_fig(a, ground_truth_boxes[i].detach().cpu(), 'green', class_names, width, height)

    for i in range(len(detected_boxes)):
        _plot_rect_on_fig(a, detected_boxes[i].detach().cpu(), 'blue', class_names, width, height)

    return fig, a


def plot_batch(detections_for_batch, ground_truth_boxes_for_batch, images, classnames):
    def plot_image(fig, a):
        plt.show()

    _process_batch(detections_for_batch, ground_truth_boxes_for_batch, images, classnames, plot_image)


def save_batch(image_paths, images_results_dir, detections_for_batch, ground_truth_boxes_for_batch, images, classnames):
    def save_image(fig, a, image_path):
        full_image_path = os.path.join(images_results_dir, os.path.basename(image_path))
        fig.savefig(full_image_path)

    _process_batch(detections_for_batch, ground_truth_boxes_for_batch, images, classnames, save_image, image_paths)


def _plot_rect_on_fig(a, box, color, class_names, width, height):
    # Plot the bounding boxes and corresponding labels on top of the image

    # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
    # of the bounding box relative to the size of the image.
    x1 = int(np.around((box[0] - box[2] / 2.0) * width))
    y1 = int(np.around((box[1] - box[3] / 2.0) * height))
    x2 = int(np.around((box[0] + box[2] / 2.0) * width))
    y2 = int(np.around((box[1] + box[3] / 2.0) * height))

    det_conf = box[4]
    cls_conf = box[5]
    cls_id = box[6]
    classes = len(class_names)
    offset = cls_id * 123457 % classes

    # Calculate the width and height of the bounding box relative to the size of the image.
    width_x = x2 - x1
    width_y = y1 - y2

    # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
    # lower-left corner of the bounding box relative to the size of the image.
    rect = patches.Rectangle((x1, y2),
                             width_x, width_y,
                             linewidth=2,
                             edgecolor=color,
                             facecolor='none')

    # Draw the bounding box on top of the image
    a.add_patch(rect)

    # Create a string with the object class name and the corresponding object class probability
    conf_tx = class_names[int(cls_id)] + ', det_conf: {:.1f}'.format(det_conf) + ' / cls_conf:{:.1f}'.format(
        cls_conf)

    # Define x and y offsets for the labels
    lxc = (width * 0.266) / 100
    lyc = (height * 1.180) / 100

    # Draw the labels on top of the image
    a.text(x1 + lxc, y1 - lyc, conf_tx, fontsize=10, color='k',
           bbox=dict(facecolor=color, edgecolor=color, alpha=0.8))
