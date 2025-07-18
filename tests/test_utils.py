import numpy as np
from seeding.utils import simple_nms


def test_simple_nms_basic():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    scores = [0.9, 0.8, 0.7]
    keep = simple_nms(boxes, scores, iou_threshold=0.5)
    assert keep == [0, 2]
