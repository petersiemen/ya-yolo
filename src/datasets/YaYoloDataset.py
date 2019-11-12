from torch.utils.data import DataLoader

class YaYoloDataset():

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def get_ground_truth_boxes(self, annotations):
        raise NotImplementedError("Subclasses should implement this!")

    def get_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def get_classnames(self):
        raise NotImplementedError("Subclasses should implement this!")
