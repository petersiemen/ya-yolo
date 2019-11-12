import torch
from datasets.YaYoloDataset import YaYoloDataset


class YoloCocoDataset(YaYoloDataset):

    def __init__(self, dataset, batch_size):
        super(YoloCocoDataset, self).__init__(dataset, batch_size)

    def get_ground_truth_boxes(self, annotations):
        boxes_for_batch = []
        for b_i in range(self.batch_size):
            boxes_for_image = []
            for o_i in range(len(annotations)):
                bbox_coordinates = annotations[o_i]['bbox']

                x = bbox_coordinates[0][b_i]
                y = bbox_coordinates[1][b_i]
                w = bbox_coordinates[2][b_i]
                h = bbox_coordinates[3][b_i]
                box = [x, y, w, h, 1, 1, annotations[o_i]['category_id'][b_i], 1]
                boxes_for_image.append(box)

            boxes_for_batch.append(boxes_for_image)

        ground_truth_boxes = torch.tensor(boxes_for_batch)
        return ground_truth_boxes
