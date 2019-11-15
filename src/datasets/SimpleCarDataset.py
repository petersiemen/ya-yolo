import os
import glob
import json
from datasets.YaYoloCustomDataset import YaYoloCustomDataset
from exif import load_image_file
from logging_config import *

logger = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.realpath(__file__))


class DetectedSimpleCarDatasetWriter():
    def __init__(self, file_writer):
        self.file_writer = file_writer
        logger.info('Init {}.'.format(self))

    def append(self, image_path, make, model, bounding_box):
        self.file_writer.append(
            json.dumps({'image': image_path,
                        'make': make,
                        'model': model,
                        'bounding_box': bounding_box
                        })
        )

    def __repr__(self):
        return 'DetectedSimpleCarDatasetWriter({})'.format(self.file_writer.fd.name)


class SimpleCarDataset(YaYoloCustomDataset):

    def __init__(self, root_dir, transforms, batch_size):
        self.root_dir = root_dir
        self.transforms = transforms
        self.batch_size = batch_size

        self.annotations = []
        self.image_paths = []
        for filename in glob.glob(os.path.join(root_dir, 'feeds/*.json')):
            with open(filename) as f:
                for line in f:
                    obj = json.loads(line)
                    make = obj['make']
                    model = obj['model']
                    for image in obj['images']:
                        image_path = os.path.join(root_dir, 'images', image['path'])
                        self.image_paths.append(image_path)
                        # we append a list of annotations for every image here because
                        # the YaYoloDataset convention is that it should be possible to attach multiple targets per image
                        self.annotations.append([
                            {
                                'make': make,
                                'model': model
                            }])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        target = self.annotations[index]
        image_path = self.image_paths[index]
        image = load_image_file(image_path)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, image_path

    def __len__(self):
        return len(self.annotations)
