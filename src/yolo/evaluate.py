import torch
from torch.utils.data import DataLoader
from device import DEVICE
from logging_config import *
from yolo.utils import non_max_suppression
from yolo.utils import get_batch_statistics


logger = logging.getLogger(__name__)


def evaluate(model,
             ya_yolo_dataset,
             summary_writer,
             lambda_coord=5,
             lambda_no_obj=0.5,
             limit=None,
             debug=False):
    model.eval()
    with torch.no_grad():
        data_loader = DataLoader(ya_yolo_dataset, batch_size=ya_yolo_dataset.batch_size, shuffle=False)
        batch_size = ya_yolo_dataset.batch_size
        class_names = model.class_names

        running_loss = 0.0

        for batch_i, (images, annotations, image_paths) in enumerate(data_loader):
            print(batch_i)

            images = images.to(DEVICE)
            ground_truth_boxes = ya_yolo_dataset.get_ground_truth_boxes(annotations).to(DEVICE)

            coordinates, class_scores, confidence = model(images)
            prediction = torch.cat((coordinates, confidence.unsqueeze(-1), class_scores), -1)

            detections = non_max_suppression(prediction=prediction,
                                             conf_thres=0.5,
                                             nms_thres=0.5)


            get_batch_statistics(detections, ground_truth_boxes, 0.9)
            print(detections)

            if limit is not None and batch_i >= limit:
                logger.info(f"Stop evaluation here after {batch_i} batches")
                return
