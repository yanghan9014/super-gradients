"""Board framing guarantees for KeypointsBoardFramedCrop on synthetic images."""

import random

import numpy as np
import pytest

from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.transforms.keypoints import KeypointsBoardFramedCrop

GREY = 127


def sample(width, height, board_box, pieces=()):
    """A uniform image (value 200) with a board whose nine landmarks span `board_box` (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = board_box
    xs, ys = [x0, x0, x1, x1, (x0 + x1) / 2, x0, x1, (x0 + x1) / 2, (x0 + x1) / 2], [y1, y0, y1, y0, (y0 + y1) / 2, (y0 + y1) / 2, (y0 + y1) / 2, y1, y0]
    joints = np.zeros((1 + len(pieces), 9, 3), dtype=np.float32)
    joints[0, :, 0], joints[0, :, 1], joints[0, :, 2] = xs, ys, 1
    for i, (x, y) in enumerate(pieces, start=1):
        joints[i, 0] = [x, y, 1]
    return PoseEstimationSample(
        image=np.full((height, width, 3), 200, dtype=np.uint8),
        mask=np.ones((height, width), dtype=np.float32),
        joints=joints,
        areas=np.full(len(joints), 100.0, dtype=np.float32),
        bboxes_xywh=np.tile(np.array([[x0, y0, x1 - x0, y1 - y0]], dtype=np.float32), (len(joints), 1)),
        is_crowd=np.zeros(len(joints), dtype=bool),
        labels=np.array([12] + [3] * len(pieces)),
        additional_samples=None,
    )


def board_span(result):
    board = result.joints[0]
    assert (board[:, 2] > 0).all(), "a board landmark left the frame"
    return board[:, 0].min(), board[:, 1].min(), board[:, 0].max(), board[:, 1].max()


def assert_board_inside_with_margin(result, fraction=0.85):
    x0, y0, x1, y1 = board_span(result)
    margin_x, margin_y = 640 * (1 - fraction) / 2, 480 * (1 - fraction) / 2
    assert x0 >= margin_x - 1e-3 and x1 <= 640 - margin_x + 1e-3
    assert y0 >= margin_y - 1e-3 and y1 <= 480 - margin_y + 1e-3


def has_padding(result):
    return (result.image == GREY).any()


def test_board_that_fits_gets_the_full_height_window_with_no_padding():
    result = KeypointsBoardFramedCrop().apply_to_sample(sample(1920, 1080, (700, 300, 1300, 800)))
    assert result.image.shape == (480, 640, 3)
    assert not has_padding(result)
    # The largest 4:3 window (1440x1080) scales by 640/1440; the board keeps that size.
    x0, _, x1, _ = board_span(result)
    assert x1 - x0 == pytest.approx(600 * 640 / 1440, abs=0.5)
    assert_board_inside_with_margin(result)


def test_close_up_enlarges_the_window_and_pads_only_what_is_missing():
    # Board spans 1800 of 1920 px: no 4:3 window inside the image can hold it with room.
    result = KeypointsBoardFramedCrop().apply_to_sample(sample(1920, 1080, (60, 100, 1860, 1000)))
    assert result.image.shape == (480, 640, 3)
    assert has_padding(result)
    assert_board_inside_with_margin(result)
    x0, _, x1, _ = board_span(result)
    assert x1 - x0 == pytest.approx(640 * 0.85, abs=0.5)


def test_portrait_close_up_keeps_the_whole_board():
    result = KeypointsBoardFramedCrop().apply_to_sample(sample(1080, 1920, (50, 500, 1030, 1500)))
    assert_board_inside_with_margin(result)
    assert has_padding(result)


def test_board_near_the_edge_keeps_its_margin_even_if_that_needs_padding():
    result = KeypointsBoardFramedCrop().apply_to_sample(sample(1920, 1080, (1400, 300, 1915, 800)))
    assert_board_inside_with_margin(result)


@pytest.mark.parametrize("size, box", [((1920, 1080), (700, 300, 1300, 800)), ((1920, 1080), (60, 100, 1860, 1000)), ((1080, 1920), (50, 500, 1030, 1500))])
def test_random_framing_always_keeps_the_board_whole_with_room(size, box):
    transform = KeypointsBoardFramedCrop(randomize=True)
    random.seed(0)
    spans = []
    for _ in range(200):
        result = transform.apply_to_sample(sample(*size, box))
        assert result.image.shape == (480, 640, 3)
        assert_board_inside_with_margin(result)
        x0, y0, x1, y1 = board_span(result)
        spans.append(max((x1 - x0) / 640, (y1 - y0) / 480))
    assert max(spans) <= 0.85 + 1e-3
    assert len({round(s, 2) for s in spans}) > 1 or max(spans) < 0.5  # sizes vary unless the base window caps them


def test_random_framing_never_zooms_in_beyond_the_largest_window():
    transform = KeypointsBoardFramedCrop(randomize=True)
    random.seed(1)
    for _ in range(50):
        x0, _, x1, _ = board_span(transform.apply_to_sample(sample(1920, 1080, (900, 500, 1000, 580))))
        assert x1 - x0 <= 100 * 640 / 1440 + 0.5


def test_pieces_outside_the_window_become_invisible_and_labels_follow_the_image():
    result = KeypointsBoardFramedCrop().apply_to_sample(sample(1920, 1080, (700, 300, 1300, 800), pieces=[(1000, 550), (10, 10)]))
    inside, outside = result.joints[1, 0], result.joints[2, 0]
    assert inside[2] == 1 and 0 <= inside[0] < 640 and 0 <= inside[1] < 480
    assert outside[2] == 0


def test_image_without_a_board_gets_the_app_centre_crop():
    s = sample(1920, 1080, (0, 0, 1, 1))
    s.labels = np.array([3])
    s.joints = s.joints[:1].copy()
    s.joints[0, :, 2] = 0
    s.joints[0, 0] = [960, 540, 1]
    result = KeypointsBoardFramedCrop().apply_to_sample(s)
    assert not has_padding(result)
    assert result.joints[0, 0, 0] == pytest.approx(320, abs=0.5) and result.joints[0, 0, 1] == pytest.approx(240, abs=0.5)


def test_outputs_are_float32_whatever_the_dataset_loaded():
    s = sample(1920, 1080, (700, 300, 1300, 800))
    s.joints, s.bboxes_xywh, s.areas = s.joints.astype(np.float64), s.bboxes_xywh.astype(np.float64), s.areas.astype(np.float64)
    result = KeypointsBoardFramedCrop(randomize=True).apply_to_sample(s)
    assert result.joints.dtype == result.bboxes_xywh.dtype == result.areas.dtype == np.float32


def test_inference_equivalent_is_resize_and_pad_to_the_output_size():
    steps = KeypointsBoardFramedCrop().get_equivalent_preprocessing()
    assert [next(iter(s)) for s in steps] == ["KeypointsLongestMaxSizeRescale", "KeypointsBottomRightPadding"]
