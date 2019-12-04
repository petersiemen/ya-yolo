import collections

import torchvision.transforms.functional as F
from PIL import Image, ImageOps
from torchvision import transforms


class Compose(object):
    """Composes several transforms for images and targets together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.SquashResize(416),
        >>>     transforms.CocoToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class SquashResize(object):
    def __init__(self, size):
        assert isinstance(size, int)
        self.size = size

    def __call__(self, img, target):
        image = img.resize((self.size, self.size))

        return image, target


class PadToFit(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target):

        w, h = image.size
        if h > w:
            new_h, new_w = int(self.size), int(self.size * w / h)
        else:
            new_h, new_w = int(self.size * h / w), int(self.size)
        top = bottom = int((self.size - new_h) / 2)
        left = right = int((self.size - new_w) / 2)
        if new_w + left + right + 1 == self.size:
            left += 1
        elif new_w + left + right - 1 == self.size:
            left -= 1
        if new_h + top + bottom + 1 == self.size:
            top += 1
        elif new_h + top + bottom - 1 == self.size:
            top -= 1

        assert new_w + left + right == self.size
        assert new_h + top + bottom == self.size

        resized = F.resize(image, (new_h, new_w), self.interpolation)
        return ImageOps.expand(resized, (left, top, right, bottom))

#
# class AbsoluteToRelativeBoundingBoxForCoco(object):
#
#     def __call__(self, img, target):
#         orig_width, orig_height = img.size
#         for i in range(len(target)):
#             if 'bbox' in target[i]:
#                 target[i]['bbox'][0] = target[i]['bbox'][0] / orig_width  # x
#                 target[i]['bbox'][1] = target[i]['bbox'][1] / orig_height  # y
#                 target[i]['bbox'][2] = target[i]['bbox'][2] / orig_width  # w
#                 target[i]['bbox'][3] = target[i]['bbox'][3] / orig_height  # h
#
#         return img, target

#
# class ConvertXandYToCenterOfBoundingBoxForCoco(object):
#
#     def __call__(self, img, target):
#         for i in range(len(target)):
#             # x-coordinate of upper left  corner -> to x-coordinate of center of box
#             target[i]['bbox'][0] = target[i]['bbox'][2] / 2 + target[i]['bbox'][0]
#             # y-coordinate of upper left  corner -> to y-coordinate of center of box
#             target[i]['bbox'][1] = target[i]['bbox'][3] / 2 + target[i]['bbox'][1]  # upper left corner
#         return img, target


class CocoToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, target):
        return F.to_tensor(img), target
