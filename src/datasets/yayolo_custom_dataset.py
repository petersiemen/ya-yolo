import os

from datasets.yayolo_dataset import YaYoloDataset

HERE = os.path.dirname(os.path.realpath(__file__))


class YaYoloCustomDataset(YaYoloDataset):

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this!")