"""Regression coverage for framing before rotation and bounded landmark loss."""

import copy
import math
import random
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from super_gradients.training.transforms.keypoints import KeypointsBoardFramedCrop, KeypointsBoardSafeGeometry
from tests.unit_tests.keypoints_board_framed_crop_test import sample


def visible(result):
    return (result.joints[0, :, 2] > 0).sum()


def translation(dx, dy=0):
    return np.array([[1.0, 0, dx], [0, 1, dy], [0, 0, 1]])


def test_portrait_board_fits_after_framing_before_ninety_degree_rotation():
    source = sample(1080, 1920, (100, 300, 980, 1620))
    old = cv2.getRotationMatrix2D((539.5, 959.5), 90, 1)
    old_points = cv2.transform(source.joints[:, :, :2], old)[0]
    assert ((old_points[:, 0] < 0) | (old_points[:, 0] >= 1080)).any()
    framed = KeypointsBoardFramedCrop(image_pad_value=0).apply_to_sample(source)
    matrix = np.eye(3)
    matrix[:2] = cv2.getRotationMatrix2D((319.5, 239.5), 90, 1)
    transform = KeypointsBoardSafeGeometry()
    with patch.object(transform, "_sample_matrix", return_value=matrix) as draw:
        result = transform.apply_to_sample(framed)
    assert draw.call_count == 1
    assert visible(result) == 9
    assert result.image.shape == (480, 640, 3)


def test_bad_candidate_is_rejected_before_image_is_rendered():
    source = sample(640, 480, (120, 100, 500, 380))
    transform = KeypointsBoardSafeGeometry()
    with patch.object(transform, "_sample_matrix", side_effect=[translation(1000), np.eye(3)]) as draw:
        with patch("cv2.warpPerspective", wraps=cv2.warpPerspective) as warp:
            result = transform.apply_to_sample(copy.deepcopy(source))
    assert draw.call_count == 2
    assert warp.call_count == 2  # One image warp and one mask warp, only for the accepted matrix.
    np.testing.assert_array_equal(result.image, source.image)
    np.testing.assert_array_equal(result.joints, source.joints)


def test_bounded_attempts_fall_back_to_identity():
    source = sample(640, 480, (120, 100, 500, 380))
    transform = KeypointsBoardSafeGeometry(max_attempts=3)
    with patch.object(transform, "_sample_matrix", return_value=translation(1000)) as draw:
        result = transform.apply_to_sample(copy.deepcopy(source))
    assert draw.call_count == 3
    np.testing.assert_array_equal(result.image, source.image)
    np.testing.assert_array_equal(result.joints, source.joints)


def test_modest_loss_is_allowed_and_old_invisible_landmarks_stay_invisible():
    source = sample(640, 480, (120, 100, 500, 380))
    source.joints[0, 0, 0] = 630
    source.joints[0, 1, 2] = 0
    transform = KeypointsBoardSafeGeometry()
    with patch.object(transform, "_sample_matrix", return_value=translation(20)) as draw:
        result = transform.apply_to_sample(source)
    assert draw.call_count == 1
    assert visible(result) == 7  # One new loss out of eight initially visible.
    assert result.joints[0, 0, 2] == result.joints[0, 1, 2] == 0


def test_image_mask_piece_labels_boxes_and_areas_follow_the_same_matrix():
    source = sample(640, 480, (120, 100, 500, 380), pieces=[(300, 200)])
    source.image[198:203, 298:303] = [10, 20, 30]
    source.mask[198:203, 298:303] = 0
    expected_joints = source.joints.copy()
    expected_joints[:, :, :2] += [20, 10]
    expected_boxes = source.bboxes_xywh.copy()
    expected_boxes[:, :2] += [20, 10]
    transform = KeypointsBoardSafeGeometry()
    with patch.object(transform, "_sample_matrix", return_value=translation(20, 10)):
        result = transform.apply_to_sample(source)
    np.testing.assert_array_equal(result.image[210, 320], [10, 20, 30])
    assert result.mask[210, 320] == 0
    np.testing.assert_array_equal(result.image[0, 0], [0, 0, 0])
    assert result.mask[0, 0] == 1
    np.testing.assert_array_equal(result.labels, [12, 3])
    np.testing.assert_allclose(result.joints, expected_joints)
    np.testing.assert_allclose(result.bboxes_xywh, expected_boxes)
    np.testing.assert_allclose(result.areas, [100, 100])


@pytest.mark.parametrize("initially_visible", [2, 4, 7, 8, 9])
def test_random_pipeline_preserves_required_fraction(initially_visible):
    random.seed(47)
    transform = KeypointsBoardSafeGeometry()
    crop = KeypointsBoardFramedCrop(randomize=True, image_pad_value=0)
    for _ in range(40):
        source = sample(1080, 1920, (100, 300, 980, 1620))
        source.joints[0, initially_visible:, 2] = 0
        result = transform.apply_to_sample(crop.apply_to_sample(source))
        assert visible(result) >= max(math.ceil(initially_visible * 0.75), min(initially_visible, 4))
        assert result.image.shape == (480, 640, 3)


def test_each_board_is_checked_separately():
    source = sample(640, 480, (100, 100, 400, 350), pieces=[(600, 200)])
    source.labels[1] = 12
    source.joints[1] = source.joints[0]
    source.joints[1, :, 0] += 200
    transform = KeypointsBoardSafeGeometry(max_attempts=1)
    with patch.object(transform, "_sample_matrix", return_value=translation(50)):
        result = transform.apply_to_sample(source)
    assert (result.joints[:, :, 2] > 0).all()  # Second board forces identity fallback.


def test_empty_annotations_still_allow_geometry():
    source = sample(640, 480, (120, 100, 500, 380))
    source.filter_by_mask(np.array([False]))
    transform = KeypointsBoardSafeGeometry()
    with patch.object(transform, "_sample_matrix", return_value=translation(20)) as draw:
        result = transform.apply_to_sample(source)
    assert draw.call_count == 1
    assert result.joints.shape == (0, 9, 3)
    assert result.image.shape == (480, 640, 3)


@pytest.mark.parametrize("kwargs", [{"min_retained_fraction": 0}, {"max_attempts": 0}, {"distortion_scale": 0.5}, {"max_shear": 90}])
def test_invalid_parameters_fail_early(kwargs):
    with pytest.raises(ValueError):
        KeypointsBoardSafeGeometry(**kwargs)
