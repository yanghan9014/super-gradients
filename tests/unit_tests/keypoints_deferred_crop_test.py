"""KeypointsBoardFramedCrop(defer_crop=True) + KeypointsBoardSafeGeometry: one resample instead of two, same content."""

import copy
import random

import cv2
import numpy as np
import pytest

from super_gradients.training.transforms.keypoints import KeypointsBoardFramedCrop, KeypointsBoardSafeGeometry
from tests.unit_tests.keypoints_board_framed_crop_test import sample


def textured(source, seed=0):
    """Replace the flat test image with fine texture (no zeros), so sharpness and padding are measurable."""
    noise = np.random.default_rng(seed).integers(0, 256, source.image.shape, dtype=np.uint8)
    source.image = np.clip(cv2.GaussianBlur(noise, (0, 0), 1.5).astype(int) * 2 - 128, 20, 255).astype(np.uint8)
    return source


def run(source, defer, seed, interpolation=cv2.INTER_LINEAR):
    random.seed(seed)
    crop = KeypointsBoardFramedCrop(randomize=True, preserve_out_of_frame_board_annotations=True, defer_crop=defer)
    geometry = KeypointsBoardSafeGeometry(max_rotation=90, max_shear=5, distortion_scale=0.15, perspective_prob=0.5, interpolation_mode=interpolation)
    return geometry.apply_to_sample(crop.apply_to_sample(copy.deepcopy(source)))


def laplacian_variance(image):
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("width,height,box", [(1920, 1080, (500, 150, 1300, 900)), (1080, 1920, (100, 600, 900, 1300)), (4032, 3024, (1500, 1000, 2600, 2000))])
def test_deferred_crop_keeps_annotations_and_padding_of_cutting_first(seed, width, height, box):
    source = textured(sample(width, height, box, pieces=[((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)]))
    cut_first, deferred = run(source, False, seed), run(source, True, seed)
    assert deferred.image.shape == cut_first.image.shape == (480, 640, 3)
    assert deferred.mask.shape == (480, 640) and deferred.pending_crop is None
    np.testing.assert_array_equal(deferred.labels, cut_first.labels)
    np.testing.assert_allclose(deferred.joints, cut_first.joints, atol=1.0)
    np.testing.assert_allclose(deferred.bboxes_xywh, cut_first.bboxes_xywh, atol=2.0)
    # Same padding (black) region, up to a pixel of edge rounding.
    padding_disagrees = (deferred.image.max(axis=2) == 0) != (cut_first.image.max(axis=2) == 0)
    assert padding_disagrees.mean() < 0.01


def test_deferred_crop_is_sharper():
    source = textured(sample(1920, 1080, (500, 150, 1300, 900)))
    ratios = [laplacian_variance(run(source, True, s, cv2.INTER_CUBIC).image) / laplacian_variance(run(source, False, s).image) for s in range(5)]
    assert min(ratios) > 1.15


def test_geometry_without_a_pending_crop_is_unchanged():
    source = textured(sample(640, 480, (100, 60, 540, 420)))
    random.seed(3)
    expected = KeypointsBoardSafeGeometry().apply_to_sample(copy.deepcopy(source))
    random.seed(3)
    actual = KeypointsBoardSafeGeometry(interpolation_mode=cv2.INTER_LINEAR).apply_to_sample(copy.deepcopy(source))
    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.joints, expected.joints)
