import json
import random

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from super_gradients.training.datasets.pose_estimation_datasets.chess_pose_estimation_dataset import ChessPoseEstimationDataset
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import (
    flat_collate_tensors_with_batch_index,
    undo_flat_collate_tensors_with_batch_index,
)
from super_gradients.training.losses.chess_yolo_nas_pose_loss import ChessYoloNASPoseLoss
from super_gradients.training.metrics.board_homography_metrics import BOARD_REFERENCE_POINTS


def unpack_with_per_image_loop(targets, batch_size):
    """The previous implementation, kept as the reference."""
    target_boxes, target_joints, target_class_labels = targets
    image_index, gt_bbox = target_boxes[:, 0], target_boxes[:, 1:5]
    per_image_targets = undo_flat_collate_tensors_with_batch_index(target_joints, batch_size)
    per_image_label_ids = undo_flat_collate_tensors_with_batch_index(target_class_labels, batch_size)
    boxes, masks, classes, max_boxes = [], [], [], 0
    for i in range(batch_size):
        mask = image_index == i
        boxes.append(gt_bbox[mask, :])
        masks.append(gt_bbox[mask, :].sum(dim=1, keepdims=True) > 0)
        classes.append(per_image_label_ids[i].long())
        max_boxes = max(max_boxes, mask.sum().item())
    pad = lambda t: F.pad(t, (0, 0, 0, max_boxes - len(t)), value=0)  # noqa: E731
    return {
        "gt_class": torch.stack([pad(t) for t in classes]),
        "gt_bbox": torch.stack([pad(t) for t in boxes]),
        "pad_gt_mask": torch.stack([pad(t) for t in masks]),
        "gt_poses": torch.stack([F.pad(t, (0, 0, 0, 0, 0, max_boxes - len(t))) for t in per_image_targets]),
    }


@pytest.mark.parametrize("counts", [[3, 0, 5, 1], [0, 0], [2], [0, 4, 0]])
def test_vectorised_target_unpacking_matches_per_image_loop(counts):
    generator = torch.Generator().manual_seed(0)
    boxes = [torch.rand(n, 4, generator=generator) * 100 for n in counts]
    boxes[0][:1] = 0  # an all-zero box must stay masked out
    joints = [torch.rand(n, 9, 3, generator=generator) for n in counts]
    labels = [torch.randint(0, 13, (n, 1), generator=generator) for n in counts]
    targets = tuple(flat_collate_tensors_with_batch_index(t) for t in (boxes, joints, labels))
    loss = ChessYoloNASPoseLoss(board_oks_sigmas=[0.1] * 9, piece_oks_sigma=[0.5], num_classes=13)
    expected = unpack_with_per_image_loop(targets, len(counts))
    actual = loss._unpack_flat_targets(targets, len(counts))
    for key, value in expected.items():
        assert actual[key].dtype == value.dtype, key
        torch.testing.assert_close(actual[key], value, msg=key)


def write_dataset(tmp_path, width, height, square, origin, piece_squares):
    """A fronto-parallel board with `square`-pixel squares whose a8 corner is at `origin`, plus pieces at square centres."""
    (tmp_path / "images").mkdir()
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.stack([127 + 100 * np.sin(xx / 37), 127 + 100 * np.cos(yy / 23), (xx + yy) % 255], axis=-1).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "a.jpg"), image)
    landmarks = np.asarray(origin) + BOARD_REFERENCE_POINTS * square
    pad = [0, 0, 0] * 8
    annotations = [{"id": 1, "image_id": 1, "category_id": 12, "iscrowd": 0, "bbox": [0, 0, 10, 10], "area": 100,
                    "keypoints": [v for x, y in landmarks for v in (x, y, 2)]}]
    for i, (file, rank) in enumerate(piece_squares):
        x, y = origin[0] + (file + 0.5) * square, origin[1] + (rank + 0.5) * square
        annotations.append({"id": i + 2, "image_id": 1, "category_id": 3, "iscrowd": 0, "bbox": [x - 32, y - 32, 64, 64], "area": 4096,
                            "keypoints": [x, y, 2] + pad})
    categories = [{"id": c, "name": str(c), "keypoints": [str(k) for k in range(9)], "skeleton": []} for c in range(13)]
    images = [{"id": 1, "file_name": "a.jpg", "width": width, "height": height}]
    (tmp_path / "ann.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))


def dataset(tmp_path, **kwargs):
    return ChessPoseEstimationDataset(data_dir=str(tmp_path), images_dir="images", json_file="ann.json", include_empty_samples=True,
                                      transforms=[], edge_links=[[0, 1]], edge_colors=None, keypoint_colors=None, **kwargs)


@pytest.mark.parametrize("width,height,expected", [(1920, 1080, (853, 480)), (4032, 3024, (640, 480)), (1080, 1920, (640, 1138)), (600, 400, (600, 400))])
def test_images_shrink_to_cover_load_size_and_annotations_follow(tmp_path, width, height, expected):
    square = min(width, height) / 12
    write_dataset(tmp_path, width, height, square, (square, square), [(2, 3)])
    full = dataset(tmp_path).load_sample(0)
    small = dataset(tmp_path, load_size=(640, 480)).load_sample(0)
    assert small.image.shape == (expected[1], expected[0], 3) and small.mask.shape == small.image.shape[:2]
    assert small.mask.dtype == np.uint8
    scale = np.array(expected) / (width, height)
    np.testing.assert_allclose(small.joints[..., :2], full.joints[..., :2] * scale, rtol=1e-6)
    np.testing.assert_allclose(small.areas, full.areas * scale.prod(), rtol=1e-6)
    # Same picture, area-averaged: close to a direct area resize of the full image.
    reference = cv2.resize(full.image, expected, interpolation=cv2.INTER_AREA).astype(float)
    assert np.abs(small.image.astype(float) - reference).mean() < 4


def test_piece_boxes_cover_one_square_centred_on_the_piece(tmp_path):
    write_dataset(tmp_path, 1920, 1080, 100, (500, 100), [(2, 3), (7, 0)])
    sample = dataset(tmp_path, piece_box_squares=1.0).load_sample(0)
    pieces = sample.labels.reshape(-1) != 12
    np.testing.assert_allclose(sample.bboxes_xywh[pieces], [[700, 400, 100, 100], [1200, 100, 100, 100]], atol=1e-6)
    np.testing.assert_allclose(sample.areas[pieces], 10000, rtol=1e-6)
    unchanged = dataset(tmp_path).load_sample(0)
    np.testing.assert_allclose(unchanged.bboxes_xywh[pieces][:, 2:], 64)


def test_lookup_table_colour_augmentations_match_full_image_arithmetic():
    from super_gradients.training.samples import PoseEstimationSample
    from super_gradients.training.transforms.keypoints import KeypointsBrightnessContrast
    from super_gradients.training.transforms.transforms import augment_hsv

    image = np.random.default_rng(1).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    for seed in range(20):
        # Brightness/contrast: previous float path vs lookup table.
        random.seed(seed)
        transform = KeypointsBrightnessContrast(prob=1.0, brightness_range=(0.4, 1.6), contrast_range=(0.8, 1.2))
        sample = PoseEstimationSample(image=image.copy(), mask=np.ones(image.shape[:2]), joints=np.zeros((0, 9, 3)), areas=None, bboxes_xywh=None,
                                      is_crowd=None, additional_samples=None)
        actual = transform.apply_to_sample(sample).image
        random.seed(seed)
        random.random()  # the transform's probability draw
        contrast, brightness = random.uniform(0.8, 1.2), random.uniform(0.4, 1.6)
        mean = image.astype(np.float32).mean(axis=(0, 1))
        expected = np.clip((image.astype(np.float32) - mean) * contrast + mean * brightness, 0, 255).astype(np.uint8)
        # The mean is summed in a different order, which can move a value across an integer boundary.
        assert np.abs(actual.astype(int) - expected).max() <= 1 and (actual != expected).mean() < 0.01

        # HSV: previous int16 path vs lookup table, with the same random draws.
        np.random.seed(seed)
        actual = augment_hsv(image.copy(), 40, 20, 20)
        np.random.seed(seed)
        gains = (np.random.uniform(-1, 1, 3) * [40, 20, 20] * np.random.randint(0, 2, 3)).astype(np.int16)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + gains[0]) % 180
        hsv[..., 1:] = np.clip(hsv[..., 1:] + gains[1:], 0, 255)
        np.testing.assert_array_equal(actual, cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
