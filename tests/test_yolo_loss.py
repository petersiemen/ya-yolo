from .context import *


def test_yolo_loss__for_no_annotated_objects():
    boxes_with_highest_iou = torch.rand(2, 1, 0)
    confidence_with_highest_iou = torch.rand(2, 1, 0)
    no_object_confidences = [torch.ones(10647), torch.ones(10647)]
    class_scores_for_ground_truth_boxes = torch.rand(2, 2, 0, 80)
    ground_truth_boxes = torch.rand(2, 1, 0)

    yolo_loss = YoloLoss()
    loss = yolo_loss.loss(boxes_with_highest_iou,
                          confidence_with_highest_iou,
                          no_object_confidences,
                          class_scores_for_ground_truth_boxes,
                          ground_truth_boxes)

    assert loss == 1.0
