from torch.utils.data import Dataset


class YaYoloDataset(Dataset):

    def get_ground_truth_boxes(self, annotations):
        raise NotImplementedError("Subclasses should implement this!")
