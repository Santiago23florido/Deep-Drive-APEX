from enum import Enum


class DetectionStatus(Enum):
    ONLY_GREEN = 1
    ONLY_RED = 2
    BOTH = 3
    NONE = 4


def extract_info(_frame, _width, _height):
    # Fallback stub for simulation: no detection.
    avg_r = 0.0
    avg_g = 0.0
    ratio_r = 0.0
    ratio_g = 0.0
    detection_status = DetectionStatus.NONE
    processing_results = {}
    return avg_r, avg_g, ratio_r, ratio_g, detection_status, processing_results
