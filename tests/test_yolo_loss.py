from .context import *


def test_get_indices_for_center_of_ground_truth_bounding_boxes__for_no_annotated_objects():
    grid_sizes = [13, 26, 52]
    ground_truth_boxes = torch.rand(2, 0, 0)
    batch_indices_of_ground_truth_boxes = get_indices_for_center_of_ground_truth_bounding_boxes(ground_truth_boxes,
                                                                                                grid_sizes)

    assert batch_indices_of_ground_truth_boxes.shape == (2, 0)


def test_get_indices_for_highest_iou_with_ground_truth_bounding_box__for_no_annotated_objects():
    batch_indices_of_ground_truth_boxes = torch.rand(2, 0)
    ground_truth_boxes = torch.rand(2, 0, 0)
    coordinates = torch.rand([2, 10647, 4])
    confidence = torch.rand([2, 10647])
    class_scores = torch.rand([2, 10647, 80])
    batch_indices_with_highest_iou = get_indices_for_highest_iou_with_ground_truth_bounding_box(
        batch_indices_of_ground_truth_boxes,
        ground_truth_boxes,
        coordinates)

    assert batch_indices_with_highest_iou.shape == (2, 0)

    boxes_with_highest_iou = select_boxes(coordinates, batch_indices_with_highest_iou)
    confidence_with_highest_iou = select_confidence(confidence, batch_indices_with_highest_iou)
    no_object_confidences = negative_select_confidences(confidence, batch_indices_with_highest_iou)
    class_scores_for_ground_truth_boxes = select_class_scores(class_scores,
                                                              batch_indices_of_ground_truth_boxes)

    yolo_loss = YoloLoss()
    loss = yolo_loss.loss(boxes_with_highest_iou,
                          confidence_with_highest_iou,
                          no_object_confidences,
                          class_scores_for_ground_truth_boxes,
                          ground_truth_boxes)

    assert loss > 0.1


def test_yolo_loss__for_no_annotated_objects():
    boxes_with_highest_iou = torch.rand(2, 0)
    confidence_with_highest_iou = torch.rand(2, 1, 0)
    no_object_confidences = [torch.ones(10647), torch.ones(10647)]
    class_scores_for_ground_truth_boxes = torch.rand(2, 2, 0, 80)
    ground_truth_boxes = torch.rand(2, 0, 0)

    yolo_loss = YoloLoss()
    loss = yolo_loss.loss(boxes_with_highest_iou,
                          confidence_with_highest_iou,
                          no_object_confidences,
                          class_scores_for_ground_truth_boxes,
                          ground_truth_boxes)

    assert loss == 1.0
