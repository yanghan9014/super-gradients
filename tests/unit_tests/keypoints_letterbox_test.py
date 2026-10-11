import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from super_gradients.training.transforms.keypoints import KeypointsBoardFramedCrop, KeypointsBoardSafeGeometry, KeypointsCompose, KeypointsLetterbox
from super_gradients.training.processing.processing import ComposeProcessing, KeypointsCenterPadding, KeypointsLongestMaxSizeRescale
from tests.unit_tests.keypoints_board_framed_crop_test import sample


def test_appearance_changes_preserve_annotations_and_precede_black_padding():
    from super_gradients.training.transforms.keypoints import KeypointsImageOnlyAugmentation

    source = sample(1920, 1080, (100, 100, 1700, 950))
    source.joints[0, 0] = [900, -30, 2]
    joints = source.joints.copy()
    appearance = KeypointsImageOnlyAugmentation([{"GaussNoise": {"var_limit": [50, 50], "p": 1}}])
    changed = appearance.apply_to_sample(copy.deepcopy(source))
    np.testing.assert_array_equal(changed.joints, joints)
    assert not np.array_equal(changed.image, source.image)
    final = KeypointsCompose([appearance, KeypointsLetterbox()], preserve_out_of_frame_board_annotations=True).apply_to_sample(source)
    assert (final.image[:60] == 0).all() and (final.image[-60:] == 0).all()
    assert final.joints[0, 0, 2] == 2
    with pytest.raises(ValueError, match="image-only"):
        KeypointsImageOnlyAugmentation([{"Affine": {"rotate": 45, "p": 1}}])


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920), (640, 480), (4032, 3024), (821, 689), (1000, 667)])
def test_whole_image_centered_with_padding_on_at_most_one_axis(width, height):
    source = sample(width, height, (10, 10, width - 10, height - 10))
    # Distinct edge strips expose any cropping; the content has no black pixels.
    source.image[:10] = [250, 20, 20]
    source.image[-10:] = [20, 250, 20]
    source.image[:, :10] = [20, 20, 250]
    source.image[:, -10:] = [250, 250, 20]
    result = KeypointsLetterbox().apply_to_sample(copy.deepcopy(source))
    scale = min(640 / width, 480 / height)
    w, h = round(width * scale), round(height * scale)
    left, top = (640 - w) // 2, (480 - h) // 2
    assert w == 640 or h == 480
    assert abs(left - (640 - w - left)) <= 1
    assert abs(top - (480 - h - top)) <= 1
    expected = np.zeros((480, 640, 3), dtype=np.uint8)
    expected[top : top + h, left : left + w] = cv2.resize(source.image, (w, h))
    np.testing.assert_array_equal(result.image, expected)
    np.testing.assert_allclose(result.joints[:, :, :2], source.joints[:, :, :2] * scale + [left, top], rtol=1e-6)
    np.testing.assert_array_equal(result.joints[:, :, 2], source.joints[:, :, 2])


def test_known_off_photo_board_points_enter_padding_but_missing_points_and_off_photo_pieces_do_not():
    source = sample(1920, 1080, (100, 100, 1700, 950), pieces=[(900, -30)])
    source.joints[0, 0] = [900, -30, 2]  # Known corner becomes (300, 50) in top padding.
    source.joints[0, 1] = [900, -30, 0]  # Same coordinates, but genuinely unannotated.
    source.joints[0, 2] = [900, -300, 2]  # Remains outside the final 640x480 canvas.
    transform = KeypointsCompose([KeypointsLetterbox()], preserve_out_of_frame_board_annotations=True)
    result = transform.apply_to_sample(copy.deepcopy(source))
    np.testing.assert_allclose(result.joints[0, 0], [300, 50, 2])
    assert result.joints[0, 1, 2] == result.joints[0, 2, 2] == result.joints[1, 0, 2] == 0
    assert (result.image[50, 300] == 0).all()
    legacy = KeypointsCompose([KeypointsLetterbox()]).apply_to_sample(copy.deepcopy(source))
    assert legacy.joints[0, 0, 2] == 0  # Historical recipes keep their input-clipping policy.


def test_training_preserves_known_board_point_until_geometry_brings_it_into_canvas():
    source = sample(640, 480, (80, 80, 500, 400))
    source.joints[0, 0] = [-20, 200, 2]
    source.joints[0, 1] = [-20, 200, 0]
    crop = KeypointsBoardFramedCrop(preserve_out_of_frame_board_annotations=True)
    crop.window = lambda _: (0, 0, 640)
    geometry = KeypointsBoardSafeGeometry()
    geometry._sample_matrix = lambda *args: np.array([[1.0, 0, 40], [0, 1, 0], [0, 0, 1]])
    result = KeypointsCompose([crop, geometry], preserve_out_of_frame_board_annotations=True).apply_to_sample(source)
    np.testing.assert_allclose(result.joints[0, 0], [20, 200, 2])
    assert result.joints[0, 1, 2] == 0


def test_validation_and_inference_pixels_match_and_predictions_map_back_to_source():
    source = sample(1000, 667, (10, 20, 980, 620))
    source.image = np.random.default_rng(42).integers(0, 256, source.image.shape, dtype=np.uint8)
    transform = KeypointsLetterbox()
    result = transform.apply_to_sample(copy.deepcopy(source))
    processing = ComposeProcessing([KeypointsLongestMaxSizeRescale((480, 640)), KeypointsCenterPadding((480, 640), 0)])
    image, metadata = processing.preprocess_image(source.image)
    np.testing.assert_array_equal(image, result.image)
    boxes = result.bboxes_xywh.copy()
    boxes[:, 2:] += boxes[:, :2]
    prediction = SimpleNamespace(poses=result.joints.copy(), bboxes_xyxy=boxes)
    prediction = processing.postprocess_predictions(prediction, metadata)
    np.testing.assert_allclose(prediction.poses[:, :, :2], source.joints[:, :, :2], atol=1e-4)
    expected_boxes = source.bboxes_xywh.copy()
    expected_boxes[:, 2:] += expected_boxes[:, :2]
    np.testing.assert_allclose(prediction.bboxes_xyxy, expected_boxes, atol=1e-4)


def test_dataset_getitem_uses_new_annotation_policy(tmp_path):
    import json
    from super_gradients.training.datasets import ChessPoseEstimationDataset

    cv2.imwrite(str(tmp_path / "image.png"), np.full((1080, 1920, 3), 200, dtype=np.uint8))
    categories = [{"id": i, "name": str(i), "keypoints": ["p"] if i < 12 else [str(j) for j in range(9)]} for i in range(13)]
    points = [900, -30, 2] + [0, 0, 0] * 8
    data = {
        "images": [{"id": 1, "file_name": "image.png", "width": 1920, "height": 1080}],
        "categories": categories,
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 12, "bbox": [100, 100, 500, 500], "area": 250000, "keypoints": points, "num_keypoints": 1, "iscrowd": 0}
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(data))
    dataset = ChessPoseEstimationDataset(
        data_dir=str(tmp_path),
        images_dir=".",
        json_file="annotations.json",
        include_empty_samples=True,
        transforms=[{"KeypointsLetterbox": {}}],
        edge_links=[[0, 1]],
        edge_colors=None,
        keypoint_colors=None,
        preserve_out_of_frame_board_annotations=True,
    )
    np.testing.assert_allclose(dataset[0].joints[0, 0], [300, 50, 2])
