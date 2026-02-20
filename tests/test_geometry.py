import numpy as np

from seeding.utils import clip_bbox_to_image, rotate_image_and_boxes


def _assert_bbox_close(actual, expected, tol=1):
    assert actual is not None
    assert expected is not None
    assert abs(actual[0] - expected[0]) <= tol
    assert abs(actual[1] - expected[1]) <= tol
    assert abs(actual[2] - expected[2]) <= tol
    assert abs(actual[3] - expected[3]) <= tol


def test_clip_bbox_to_image_clamps_and_normalizes():
    clipped = clip_bbox_to_image((-5, 8, 12, -2), width=10, height=10)
    assert clipped == (0, 0, 10, 8)


def test_clip_bbox_to_image_returns_none_for_outside_bbox():
    clipped = clip_bbox_to_image((30, 30, 40, 40), width=20, height=20)
    assert clipped is None


def test_rotate_image_and_boxes_roundtrip_for_90_degree_steps():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    boxes = [(10, 8, 26, 20)]

    rotated_image, rotated_boxes = rotate_image_and_boxes(
        image,
        boxes,
        angle=90,
    )
    restored_image, restored_boxes = rotate_image_and_boxes(
        rotated_image,
        [box for box in rotated_boxes if box is not None],
        angle=-90,
    )

    assert restored_image.shape == image.shape
    _assert_bbox_close(restored_boxes[0], boxes[0])
