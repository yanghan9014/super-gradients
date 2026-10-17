from types import SimpleNamespace

import numpy as np
import pytest
import torch

from super_gradients.training.datasets.pose_estimation_datasets.chess_pose_estimation_dataset import _count_on_board_pieces
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import ChessYoloNASPoseCollateFN
from super_gradients.training.losses.chess_yolo_nas_pose_loss import (
    BOARD_180_PERMUTATION, ChessYoloNASPoseLoss, YoloNASPoseBoxesAssignmentResult, batch_pose_oks,
)
from super_gradients.training.metrics.board_homography_metrics import BOARD_REFERENCE_POINTS
from super_gradients.training.samples import ChessPoseEstimationSample


def annotation(count, outside=0):
    joints = np.zeros((count + outside + 1, 9, 3), dtype=np.float32)
    joints[0, :, :2] = BOARD_REFERENCE_POINTS * 10 + 10
    joints[0, :, 2] = 2
    joints[1:, 0] = [25, 25, 2]
    if outside:
        joints[-outside:, 0] = [120, 25, 2]
    return SimpleNamespace(ann_class_labels=np.array([12] + [0] * (count + outside)), ann_keypoints=joints)


@pytest.mark.parametrize("count", [0, 6, 7, 32])
def test_original_count_excludes_off_board_pieces_and_survives_filtering_and_collation(count):
    ann = annotation(count, outside=2)
    counts = _count_on_board_pieces(ann, 12)
    assert counts[0] == count and (counts[1:] == -1).all()
    sample = ChessPoseEstimationSample(
        image=np.zeros((100, 100, 3), dtype=np.float32), mask=np.ones((100, 100)),
        joints=ann.ann_keypoints, labels=ann.ann_class_labels, bboxes_xywh=np.tile([10, 10, 80, 80], (len(counts), 1)),
        board_piece_counts=counts,
    )
    sample.filter_by_mask(sample.labels == 12)  # Every piece removed by a crop: original count still controls rotation.
    _, targets, extras = ChessYoloNASPoseCollateFN(include_board_piece_counts=True)([sample])
    assert len(targets) == 4 and targets[3].tolist() == [[0, count]]
    assert extras["gt_samples"][0].board_piece_counts.tolist() == [count]


def test_unknown_or_degenerate_geometry_keeps_count_unknown():
    ann = annotation(6)
    ann.ann_keypoints[0, 3:, 2] = 0
    assert _count_on_board_pieces(ann, 12)[0] == -1
    ann = annotation(6)
    ann.ann_keypoints[0, :, :2] = [10, 10]
    assert _count_on_board_pieces(ann, 12)[0] == -1
    ann = annotation(6)
    ann.ann_keypoints[1, 0, 2] = 0
    assert _count_on_board_pieces(ann, 12)[0] == -1


def example(counts=(6, 7)):
    n = len(counts)
    coords = torch.tensor(BOARD_REFERENCE_POINTS, dtype=torch.float32) * 4 + 4
    gt = torch.cat([coords, torch.full((9, 1), 2.0)], dim=-1)
    boxes = torch.tensor([[i, 0, 0, 40, 40] for i in range(n)], dtype=torch.float32)
    joints = torch.stack([torch.cat([torch.full((9, 1), float(i)), gt], dim=1) for i in range(n)])
    labels = torch.tensor([[i, 12] for i in range(n)])
    counts = torch.tensor([[i, c] for i, c in enumerate(counts)])
    predictions = (
        torch.zeros(n, 1, 13, requires_grad=True), torch.zeros(n, 1, 68, requires_grad=True),
        (coords[BOARD_180_PERMUTATION] + 0.1).repeat(n, 1, 1, 1).requires_grad_(), torch.zeros(n, 1, 9, requires_grad=True),
        torch.tensor([[0., 0, 40, 40]]), torch.tensor([[20., 20.]]), [1], torch.tensor([[4.]]),
    )
    return ((), predictions), (boxes, joints, labels, counts)


def loss(**kwargs):
    return ChessYoloNASPoseLoss(
        board_oks_sigmas=[0.1] * 9, piece_oks_sigma=[0.35], num_classes=13, bbox_assigner_topk=1,
        assigner_multiply_by_pose_oks=True, ignore_unannotated_board_keypoints=True,
        rotation_invariant_start_epoch=0, rotation_invariant_max_pieces=6, **kwargs,
    )


def test_threshold_is_inclusive_configurable_and_applies_from_first_epoch():
    criterion = loss()
    outputs, targets = example((0, 6, 7, -1))
    unpacked = criterion._unpack_flat_targets(targets, 4)
    assert criterion._rotation_eligibility(unpacked).flatten().tolist() == [True, True, False, False]
    criterion.rotation_invariant_max_pieces = 7
    assert criterion._rotation_eligibility(unpacked).flatten().tolist() == [True, True, True, False]
    criterion.rotation_invariant_start_epoch = 10
    assert not criterion._rotation_eligibility(unpacked).any()
    criterion.current_epoch = 10
    assert criterion._rotation_eligibility(unpacked).sum() == 3
    del unpacked["board_piece_counts"]
    with pytest.raises(ValueError, match="original board piece counts"):
        criterion._rotation_eligibility(unpacked)


def test_assignment_and_regression_accept_same_whole_board_rotation_only_when_eligible():
    outputs, targets = example()
    criterion = loss(log_pose_diagnostics=True, rescale_pose_loss_with_assigned_score=True)
    original_targets = tuple(t.clone() for t in targets)
    total, logged = criterion(outputs, targets)
    assert torch.isfinite(total)
    diagnostics = dict(zip(criterion.component_names, logged))
    assert diagnostics["rotation_eligible_fraction"] == 0.5
    assert diagnostics["rotation_flipped_assigned_fraction"] == 1
    for term in ("cls", "iou", "dfl", "pose_cls", "pose_reg"):
        torch.testing.assert_close(diagnostics[f"board_loss_{term}"] + diagnostics[f"piece_loss_{term}"], diagnostics[f"loss_{term}"])
    total.backward()
    assert torch.isfinite(outputs[1][2].grad).all()
    for actual, expected in zip(targets, original_targets):
        torch.testing.assert_close(actual, expected)  # Validation GT and strict metrics remain canonical.
    unpacked = criterion._unpack_flat_targets(targets, 2)
    oks = batch_pose_oks(
        unpacked["gt_poses"], outputs[1][2], unpacked["gt_bbox"], criterion.oks_sigmas,
        gt_labels=unpacked["gt_class"], piece_sigma=criterion.piece_oks_sigma,
        rotation_eligible=criterion._rotation_eligibility(unpacked),
    )
    assert oks[0, 0, 0] > 0.99 and oks[1, 0, 0] < 0.2
    relaxed_outputs, relaxed_targets = example((6,))
    relaxed_reg = criterion(relaxed_outputs, relaxed_targets)[1][4]
    strict_outputs, strict_targets = example((7,))
    strict_reg = criterion(strict_outputs, strict_targets)[1][4]
    assert strict_reg > relaxed_reg * 10


def test_missing_annotation_mask_is_permuted_with_coordinates():
    criterion = loss()
    gt = torch.tensor(BOARD_REFERENCE_POINTS, dtype=torch.float32)[None]
    visible = torch.full((1, 9, 1), 2.0)
    visible[:, 0] = 0
    prediction = gt[:, BOARD_180_PERMUTATION].clone()
    chosen, mask = criterion._resolve_board_orientation(prediction, gt, visible)
    torch.testing.assert_close(chosen, prediction)
    assert mask[0, 3] == 0 and mask[0, 0] == 2


@pytest.mark.parametrize("weighted", [True, False])
def test_multiplier_doubles_only_board_localization_gradients(weighted):
    poses = torch.zeros(1, 2, 9, 3)
    poses[..., :2] = 15
    poses[0, 0, :, 2] = 2
    poses[0, 1, 0, 2] = 2
    assignment = YoloNASPoseBoxesAssignmentResult(
        assigned_labels=torch.tensor([[12, 0]]), assigned_bboxes=torch.tensor([[[0., 0, 40, 40], [0., 0, 40, 40]]]),
        assigned_poses=poses, assigned_scores=torch.nn.functional.one_hot(torch.tensor([[12, 0]]), 13).float(),
        assigned_gt_index=torch.tensor([[0, 1]]), assigned_crowd=torch.zeros(1, 2),
    )
    results = []
    for multiplier in [1., 2.]:
        criterion = loss(board_localization_loss_multiplier=multiplier, rescale_pose_loss_with_assigned_score=weighted, log_pose_diagnostics=True)
        dist = torch.zeros(1, 2, 68, requires_grad=True)
        boxes = torch.tensor([[[1., 1, 4, 4], [1., 1, 4, 4]]], requires_grad=True)
        coords = torch.full((1, 2, 9, 2), 17., requires_grad=True)
        logits = torch.zeros(1, 2, 9, requires_grad=True)
        losses = criterion._bbox_loss(dist, boxes, coords, logits, torch.full((2, 1), 8.), torch.tensor([[2., 2], [2., 2]]), assignment, torch.tensor(2.), 16)
        results.append(torch.autograd.grad(sum(losses), (dist, boxes, coords, logits)))
        for index, term, weight in [(0, "iou", criterion.iou_loss_weight), (1, "dfl", criterion.dfl_loss_weight), (2, "pose_cls", criterion.pose_cls_loss_weight), (3, "pose_reg", criterion.pose_reg_loss_weight)]:
            torch.testing.assert_close(criterion._pose_diagnostics[f"board_loss_{term}"] + criterion._pose_diagnostics[f"piece_loss_{term}"], losses[index] * weight)
    for i in range(3):
        torch.testing.assert_close(results[1][i][0, 0], 2 * results[0][i][0, 0])
        torch.testing.assert_close(results[1][i][0, 1], results[0][i][0, 1])
        assert results[0][i][0, 0].abs().sum() > 0
    torch.testing.assert_close(results[1][3], results[0][3])
