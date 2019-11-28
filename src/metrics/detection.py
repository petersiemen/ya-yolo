from metrics.utils import image_path_to_image_id
from metrics.bounding_box import BoundingBox

class Detection:
    def __init__(self, file_id, class_id, confidence, bounding_box):
        self.file_id = file_id
        self.class_id = class_id
        self.confidence = confidence
        self.bounding_box = bounding_box

        self.classification = None

    def set_classification(self, classification):
        self.classification = classification

    def __repr__(self):
        return f"Detection({self.file_id}, {self.class_id}, {self.confidence}, {self.bounding_box}, {self.classification})"

    @staticmethod
    def from_detections(image_paths, detections):
        batch_size = len(detections)
        for image_i in range(batch_size):
            image_id = image_path_to_image_id(image_paths[image_i])
            for detection in detections[image_i]:
                detection = detection.detach().cpu().numpy()
                confidence = detection[4]# * detection[5]
                yield Detection(file_id=image_id, class_id=int(detection[-1]), confidence=confidence,
                                bounding_box=BoundingBox.from_xywh(
                                    x=detection[0], y=detection[1], w=detection[2], h=detection[3]
                                ))
