from metrics.utils import image_path_to_image_id
from metrics.bounding_box import BoundingBox


class GroundTruth:

    def __init__(self, file_id, class_id, bounding_box):
        self.file_id = file_id
        self.class_id = class_id
        self.bounding_box = bounding_box

    def __repr__(self):
        return f"GroundTruth({self.file_id}, {self.class_id}, {self.bounding_box})"

    @staticmethod
    def from_ground_truths(image_paths, ground_truths):
        batch_size = len(image_paths)
        for image_i in range(batch_size):
            image_id = image_path_to_image_id(image_paths[image_i])
            for ground_truth in ground_truths[image_i]:
                ground_truth = ground_truth.detach().cpu().numpy()
                yield GroundTruth(file_id=image_id, class_id=int(ground_truth[-1]), bounding_box=BoundingBox.from_xywh(
                    x=ground_truth[0], y=ground_truth[1], w=ground_truth[2], h=ground_truth[3]
                ))
