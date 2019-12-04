import torch
from torchvision.datasets import ImageNet


class YaYoloImageNetDataset(ImageNet):
    def __init__(self, root, batch_size, transforms, split='train', download=False):
        ImageNet.__init__(self,
                          root=root, split=split, download=download,
                          transforms=transforms)
        self.batch_size = batch_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return ImageNet.__len__(self)
