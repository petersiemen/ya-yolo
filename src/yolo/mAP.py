import os


class MeanAveragePrecisionHelper():
    def __init__(self, out_dir):
        self.detection_results_dir = os.path.join(out_dir, "detection-results")
        self.ground_truth_dir = os.path.join(out_dir, "ground-truth")
