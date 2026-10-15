"""Deterministic validation framing: whole photo, or a crop around boards smaller than the app accepts."""

import copy

import cv2
import numpy as np
import pytest

from super_gradients.training.transforms.keypoints import KeypointsBoardFocusLetterbox
from super_gradients.training.transforms.keypoints.keypoints_board_focus_letterbox import board_focus_window, letterbox_image
from tests.unit_tests.keypoints_board_framed_crop_test import sample


def board(origin, square):
    """A fronto-parallel board with `square`-pixel squares whose a8 corner is at `origin`, as (x0, y0, x1, y1)."""
    x, y = origin
    return x, y, x + 8 * square, y + 8 * square


def window(width, height, box):
    return board_focus_window(sample(width, height, box).joints[0], (width, height))


def test_board_meeting_the_app_minimum_keeps_the_whole_photo():
    # 1920x1080 letterboxes at 1/3: 100px squares give a board of 0.23 of the frame.
    assert window(1920, 1080, board((500, 100), 100)) == (0, 0, 1920, 1080)


def test_small_board_is_cropped_to_two_squares_of_margin_widened_to_4_by_3():
    # 40px squares: 0.02 of the frame. Margin box 480x480 around the board, widened to 640x480.
    assert window(1920, 1080, board((800, 400), 40)) == (640, 320, 1280, 800)


def test_window_slides_inside_the_photo_instead_of_padding_for_margin():
    # Centred on the board the window would start at x=-120.
    assert window(1920, 1080, board((40, 400), 40)) == (0, 320, 640, 800)


def test_axis_the_window_cannot_fit_keeps_the_whole_photo():
    # The 320px-wide window cannot fit in a 300px-wide photo; vertically it still crops.
    assert window(300, 1080, board((60, 400), 20)) == (0, 360, 300, 600)


def test_without_a_usable_board_the_whole_photo_is_kept():
    joints = sample(1920, 1080, board((800, 400), 40)).joints[0]
    joints[3:, 2] = 0  # only three landmarks left
    assert board_focus_window(joints, (1920, 1080)) == (0, 0, 1920, 1080)
    assert board_focus_window(None, (1920, 1080)) == (0, 0, 1920, 1080)


def test_letterbox_image_area_averages_and_centres():
    image = np.random.default_rng(0).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    canvas, (scale_x, scale_y, left, top) = letterbox_image(image, (640, 480))
    assert canvas.shape == (480, 640, 3) and (scale_x, scale_y, left, top) == (1 / 3, 1 / 3, 0, 60)
    assert (canvas[:60] == 0).all() and (canvas[420:] == 0).all()
    np.testing.assert_array_equal(canvas[60:420], cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))


def test_transform_crops_maps_annotations_and_drops_pieces_outside_the_crop():
    box = board((800, 400), 40)
    # One piece on the board, one beside it inside the margin, one far outside the crop.
    source = sample(1920, 1080, box, pieces=[(820, 420), (1250, 560), (100, 100)])
    transform = KeypointsBoardFocusLetterbox()
    result = transform.apply_to_sample(copy.deepcopy(source))
    assert result.image.shape == (480, 640, 3)
    assert list(result.labels) == [12, 3, 3]
    np.testing.assert_allclose(result.joints[1:, 0, :2], [[180, 100], [610, 240]])
    # Window (640, 320)-(1280, 800) maps 1:1.
    np.testing.assert_allclose(result.joints[0, 1, :2], [160, 80])
    again = transform.apply_to_sample(copy.deepcopy(source))
    np.testing.assert_array_equal(again.image, result.image)
    np.testing.assert_array_equal(again.joints, result.joints)


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (640, 480), (4032, 3024)])
def test_large_board_frames_exactly_like_the_whole_photo_letterbox(width, height):
    source = sample(width, height, (10, 10, width - 10, height - 10))
    result = KeypointsBoardFocusLetterbox().apply_to_sample(copy.deepcopy(source))
    expected, (scale_x, scale_y, left, top) = letterbox_image(source.image, (640, 480))
    np.testing.assert_array_equal(result.image, expected)
    np.testing.assert_allclose(result.joints[0, :, 0], source.joints[0, :, 0] * scale_x + left, rtol=1e-6)
