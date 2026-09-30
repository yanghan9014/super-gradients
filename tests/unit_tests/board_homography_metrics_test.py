"""Geometric regression tests independent of training data/checkpoints."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from super_gradients.training.losses.chess_yolo_nas_pose_loss import BOARD_180_PERMUTATION
from super_gradients.training.metrics.board_homography_metrics import BOARD_REFERENCE_POINTS, compute_board_homography_matching
from super_gradients.training.metrics.chess_pose_estimation_metrics import ChessPoseEstimationMetrics
from super_gradients.module_interfaces.pose_estimation_post_prediction_callback import ChessPoseEstimationPredictions


AFFINE_BOARD = np.array([[32.0, 0, 64], [0, 32.0, 96], [0, 0, 1]])
PERSPECTIVE_BOARD = np.array([[57.0, 12, -80], [6, 40, 25], [0.055, 0.025, 1]])


def project(points, homography):
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ homography.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def board_points(homography=AFFINE_BOARD, shift=(0, 0)):
    # These predictions induce H_pred(p) = H_annotation(p) + shift.
    return project(BOARD_REFERENCE_POINTS - np.asarray(shift), homography)


def matches(predictions, targets=None, visibility=None, point_scores=None, scores=None, tolerance=0.25, top_k=50):
    predictions = np.asarray(predictions).reshape(-1, 9, 2)
    targets = np.asarray([board_points()] if targets is None else targets).reshape(-1, 9, 2)
    return compute_board_homography_matching(
        predicted_poses=torch.tensor(predictions, dtype=torch.float32),
        predicted_scores=torch.tensor(np.ones(len(predictions)) if scores is None else scores, dtype=torch.float32),
        targets=torch.tensor(targets, dtype=torch.float32),
        targets_visibilities=torch.tensor(np.ones(targets.shape[:2]) if visibility is None else np.asarray(visibility)),
        tolerance=tolerance,
        top_k=top_k,
        predicted_pose_scores=None if point_scores is None else torch.tensor(point_scores),
    )


@pytest.mark.parametrize("homography", [AFFINE_BOARD, PERSPECTIVE_BOARD])
@pytest.mark.parametrize(
    "shift, expected",
    [
        ((0, 0), True),
        ((0.20, 0.20), True),
        ((0.25, 0.25), True),
        ((-0.25, -0.25), True),
        ((0.2501, 0), False),
        ((0, 0.2501), False),
        ((-0.26, 0), False),
        ((0, -0.26), False),
    ],
)
def test_rectified_axis_limits(homography, shift, expected):
    strict, relaxed = matches([board_points(homography, shift)], targets=[board_points(homography)])
    assert strict.preds_matched.item() == expected
    assert relaxed.preds_matched.item() == expected


def test_tolerance_is_configurable_in_square_units():
    predicted = [board_points(shift=(0.4, 0.4))]
    assert not matches(predicted)[0].preds_matched.item()
    assert matches(predicted, tolerance=0.5)[0].preds_matched.item()
    assert not matches([board_points(shift=(0.001, 0))], tolerance=0.0001)[0].preds_matched.item()


def test_uses_worst_reference_point_instead_of_average():
    # Drift is zero at x=0 and 0.32 square at x=8; the average is only 0.16.
    stretched = BOARD_REFERENCE_POINTS.copy()
    stretched[:, 0] /= 1.04
    predicted = project(stretched, PERSPECTIVE_BOARD)
    strict, _ = matches([predicted], targets=[board_points(PERSPECTIVE_BOARD)])
    assert not strict.preds_matched.item()


def test_cropped_board_reconstructs_missing_edge_from_annotation():
    targets = board_points(PERSPECTIVE_BOARD)
    visible = np.array([True, True, False, False, True, True, False, True, True])
    targets[~visible] = 0  # COCO's unannotated/out-of-frame placeholders.
    point_scores = visible.astype(float)[None]
    assert matches([board_points(PERSPECTIVE_BOARD)], [targets], [visible], point_scores)[0].preds_matched.item()

    # The visible left/middle points move <=0.16. The inferred right edge moves
    # 0.32 and must fail even though it is outside the annotated crop.
    stretched = BOARD_REFERENCE_POINTS.copy()
    stretched[:, 0] /= 1.04
    predicted = project(stretched, PERSPECTIVE_BOARD)
    assert not matches([predicted], [targets], [visible], point_scores)[0].preds_matched.item()


def test_relaxed_accepts_only_whole_board_180_degree_relabelling():
    target = board_points(PERSPECTIVE_BOARD)
    strict, relaxed = matches([target[BOARD_180_PERMUTATION]], [target])
    assert not strict.preds_matched.item()
    assert relaxed.preds_matched.item()

    rotated_90 = np.column_stack((8 - BOARD_REFERENCE_POINTS[:, 1], BOARD_REFERENCE_POINTS[:, 0]))
    strict, relaxed = matches([project(rotated_90, PERSPECTIVE_BOARD)], [target])
    assert not strict.preds_matched.item()
    assert not relaxed.preds_matched.item()


@pytest.mark.parametrize("invalid", [np.zeros((9, 2)), np.full((9, 2), np.nan), np.full((9, 2), np.inf), np.tile(np.arange(9)[:, None], (1, 2))])
def test_invalid_prediction_never_matches(invalid):
    strict, relaxed = matches([invalid])
    assert not strict.preds_matched.item()
    assert not relaxed.preds_matched.item()


def test_insufficient_annotation_is_unmatched_and_still_in_recall_denominator():
    strict, relaxed = matches([board_points()], visibility=[[1, 1, 1, 0, 0, 0, 0, 0, 0]])
    assert strict.num_targets == relaxed.num_targets == 1
    assert not strict.preds_matched.item()
    assert not relaxed.preds_matched.item()


def test_keypoint_confidence_controls_prediction_homography():
    scores = np.array([[0.9, 0.9, 0.9, 0.69, 0, 0, 0, 0, 0]])
    assert not matches([board_points()], point_scores=scores)[0].preds_matched.item()
    scores[0, 3] = 0.7
    assert matches([board_points()], point_scores=scores)[0].preds_matched.item()


def test_homogeneous_division_by_zero_is_a_miss():
    invalid_projection = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, -64.0]])
    with patch("super_gradients.training.metrics.board_homography_metrics._fit_homography", side_effect=[invalid_projection, np.linalg.inv(AFFINE_BOARD)]):
        strict, relaxed = matches([board_points()])
    assert not strict.preds_matched.item()
    assert not relaxed.preds_matched.item()


def test_matching_is_one_to_one_confidence_ordered_and_capped():
    strict, _ = matches([board_points(), board_points()], scores=[0.4, 0.9])
    assert strict.preds_scores.tolist() == pytest.approx([0.9, 0.4])
    assert strict.preds_matched[:, 0].tolist() == [True, False]
    strict, _ = matches([board_points(), board_points()], scores=[0.4, 0.9], top_k=1)
    assert strict.preds_matched.shape == (1, 1)


def test_multiple_boards_match_independently_including_relaxed_orientation():
    targets = [board_points(), board_points(PERSPECTIVE_BOARD)]
    predictions = [targets[1][BOARD_180_PERMUTATION], targets[0]]
    strict, relaxed = matches(predictions, targets, scores=[0.9, 0.8])
    assert strict.preds_matched[:, 0].tolist() == [False, True]
    assert relaxed.preds_matched[:, 0].tolist() == [True, True]


def make_metric(**kwargs):
    params = dict(post_prediction_callback=lambda outputs: outputs, num_joints=9, oks_sigmas=[0.5] * 9, board_class_id=12)
    params.update(kwargs)
    return ChessPoseEstimationMetrics(**params)


def update_metric(metric, poses, targets=None, scores=None, labels=None):
    poses = np.asarray(poses).reshape(-1, 9, 2)
    targets = np.asarray([board_points()] if targets is None else targets).reshape(-1, 9, 2)
    class_scores = np.zeros((len(poses), 13), dtype=np.float32)
    class_scores[np.arange(len(poses)), 12 if labels is None else np.asarray(labels)] = 0.9 if scores is None else scores
    prediction = SimpleNamespace(poses=poses, scores=class_scores, pose_scores=np.ones(poses.shape[:2]))
    sample = SimpleNamespace(
        joints=np.concatenate((targets, np.ones((*targets.shape[:2], 1))), axis=-1),
        bboxes_xywh=np.tile([64, 96, 256, 256], (len(targets), 1)),
        areas=np.full(len(targets), 256**2),
        labels=np.full(len(targets), 12),
    )
    metric.update(preds=[prediction], target=None, gt_samples=[sample])


def test_overall_oks_metrics_unchanged_while_board_fails():
    baseline = make_metric(board_class_id=None, iou_thresholds_to_report=[0.5, 0.75])
    geometric = make_metric(iou_thresholds_to_report=[0.5, 0.75])
    loose = make_metric(board_max_square_error=0.5, iou_thresholds_to_report=[0.5, 0.75])
    for metric in (baseline, geometric, loose):
        update_metric(metric, [board_points(shift=(0.26, 0))])
    expected = baseline.compute()
    for result in (geometric.compute(), loose.compute()):
        for name, value in expected.items():
            assert result[name] == value
    assert expected["AP"] == expected["AR"] == 1
    assert geometric.compute()["Board_AP"] == geometric.compute()["Board_AR"] == 0
    assert loose.compute()["Board_AP"] == loose.compute()["Board_AR"] == 1


def test_mixed_piece_and_board_classes_keep_the_same_overall_ap_ar():
    baseline = make_metric(board_class_id=None)
    geometric = make_metric()
    target = np.concatenate((board_points(), np.ones((9, 1))), axis=1)
    piece = np.zeros((9, 3))
    piece[0] = [180, 220, 2]
    sample = SimpleNamespace(
        joints=np.stack((target, piece)),
        bboxes_xywh=np.array([[64, 96, 256, 256], [170, 200, 30, 40]]),
        areas=np.array([256**2, 1200]),
        labels=np.array([12, 3]),
    )
    class_scores = np.zeros((3, 13))
    class_scores[0, 12], class_scores[1, 3], class_scores[2, 3] = 0.9, 0.8, 0.95
    false_piece = piece[:, :2] + 100
    prediction = SimpleNamespace(
        poses=np.stack((board_points(shift=(0, 0.3)), piece[:, :2], false_piece)),
        scores=class_scores,
        pose_scores=np.ones((3, 9)),
    )
    for metric in (baseline, geometric):
        metric.update(preds=[prediction], target=None, gt_samples=[sample])
    assert baseline.compute()["AP"] == geometric.compute()["AP"] == pytest.approx(0.75)
    assert baseline.compute()["AR"] == geometric.compute()["AR"] == 1
    assert geometric.compute()["Board_AP"] == 0


def test_board_ap_uses_confidence_ranking_and_ar_counts_missing_boards():
    metric = make_metric()
    update_metric(metric, [board_points(shift=(1, 0))], targets=[], scores=[0.95])  # high-confidence false positive
    update_metric(metric, [board_points()], scores=[0.8])  # one true positive
    update_metric(metric, [])  # one missed target
    result = metric.compute()
    assert result["Board_AR"] == pytest.approx(0.5)
    # Precision=1/2 at recall 0..0.50 (51 samples), then zero at unreachable recall.
    assert result["Board_AP"] == pytest.approx(0.5 * 51 / 101)
    assert result["Board_Relaxed_AP"] == result["Board_AP"]


def test_wrong_class_cannot_match_board():
    metric = make_metric()
    update_metric(metric, [board_points()], labels=[0])
    result = metric.compute()
    assert result["Board_AR"] == result["Board_AP"] == 0


def test_empty_predictions_and_reset_clear_all_board_state():
    metric = make_metric()
    assert all(value == -1 for value in metric.compute().values())
    update_metric(metric, [])
    assert metric.compute()["Board_AR"] == 0
    metric.reset()
    update_metric(metric, [board_points()[BOARD_180_PERMUTATION]])
    assert metric.compute()["Board_AP"] == 0
    assert metric.compute()["Board_Relaxed_AP"] == 1
    metric.reset()
    assert not metric.predictions and not metric.board_predictions and not metric.board_relaxed_predictions
    assert all(value == -1 for value in metric.compute().values())


def test_sample_update_forwards_keypoint_scores():
    metric = make_metric()
    prediction = ChessPoseEstimationPredictions(
        poses=torch.tensor(board_points()[None], dtype=torch.float32),
        pose_scores=torch.zeros((1, 9)),
        scores=torch.nn.functional.one_hot(torch.tensor([12]), 13).float(),
        labels=torch.tensor([12]),
        bboxes_xyxy=torch.tensor([[64, 96, 320, 352]]),
    )
    sample = SimpleNamespace(
        joints=np.concatenate((board_points(), np.ones((9, 1))), axis=1)[None],
        bboxes_xywh=np.array([[64, 96, 256, 256]]),
        areas=np.array([256**2]),
        labels=np.array([12]),
    )
    metric.update(preds=[prediction], target=None, gt_samples=[sample])
    assert metric.compute()["AP"] == 1
    assert metric.compute()["Board_AP"] == 0


def test_distributed_sync_includes_both_board_states():
    metric = make_metric()
    update_metric(metric, [board_points()])
    metric.is_distributed, metric.world_size, metric.rank = True, 2, 0

    def gather(states, local):
        states[:] = [local, local]

    with patch("torch.distributed.all_gather_object", side_effect=gather):
        metric._sync_dist()
    assert len(metric.predictions) == len(metric.board_predictions) == len(metric.board_relaxed_predictions) == 2


@pytest.mark.parametrize("tolerance", [0, -0.1, float("inf"), float("nan")])
def test_invalid_tolerance_is_rejected(tolerance):
    with pytest.raises(ValueError, match="board_max_square_error"):
        make_metric(board_max_square_error=tolerance)
