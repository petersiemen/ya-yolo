from torch.utils.data import Dataset


class YaYoloDataset(Dataset):

    def get_ground_truth_boxes(self, annotations):
        raise NotImplementedError("Subclasses should implement this!")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        raise NotImplementedError("Subclasses should implement this!")
