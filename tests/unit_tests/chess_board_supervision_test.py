"""Board annotation presence is independent of image boundaries and physical visibility."""

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from super_gradients.training.losses.chess_yolo_nas_pose_loss import ChessYoloNASPoseLoss, YoloNASPoseBoxesAssignmentResult
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.chess_yolo_nas_pose_ndfl_heads import ChessYoloNASPoseNDFLHeads
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.chess_yolo_nas_pose_post_prediction_callback import ChessYoloNASPosePostPredictionCallback
from super_gradients.training.transforms.keypoints import (
    KeypointsBoardFocusLetterbox, KeypointsBoardFramedCrop, KeypointsBoardSafeGeometry, KeypointsCompose, KeypointsRandomAffineTransform,
)
from tests.unit_tests.keypoints_board_framed_crop_test import sample


@pytest.mark.parametrize("defer", [False, True])
def test_known_points_remain_supervised_after_crop_and_geometry(defer):
    source = sample(640, 480, (80, 80, 500, 400), pieces=[(630, 200)])
    source.joints[0, 0] = [-100, 200, 2]  # Already outside source.
    source.joints[0, 1] = [630, 200, 2]  # Geometry moves this outside output.
    source.joints[0, 2] = [100, 200, 0]  # Missing annotation, even though in frame.
    crop = KeypointsBoardFramedCrop(preserve_out_of_frame_board_annotations=True, defer_crop=defer)
    crop.window = lambda _: (0, 0, 640)
    geometry = KeypointsBoardSafeGeometry()
    geometry._sample_matrix = lambda *args: np.array([[1.0, 0, 40], [0, 1, 0], [0, 0, 1]])
    result = KeypointsCompose([crop, geometry], preserve_out_of_frame_board_annotations=True).apply_to_sample(source)
    np.testing.assert_allclose(result.joints[0, 0], [-60, 200, 2])
    np.testing.assert_allclose(result.joints[0, 1], [670, 200, 2])
    assert result.joints[0, 2, 2] == 0
    assert result.joints[1, 0, 2] == 0
    np.testing.assert_array_equal(result.sanitize_sample().joints, result.joints)


def test_sanitize_preserves_only_finite_board_annotations_and_keeps_legacy_default():
    source = sample(640, 480, (80, 80, 500, 400), pieces=[(-10, 100)])
    source.joints[0, :4] = [[-10, 100, 2], [700, 100, 1], [-10, 100, 0], [np.nan, 100, 2]]
    legacy = copy.deepcopy(source).sanitize_sample()
    assert (legacy.joints[0, :3, 2] == 0).all()
    source.preserve_out_of_frame_board_annotations = True
    result = source.sanitize_sample()
    np.testing.assert_array_equal(result.joints[0, :4, 2], [2, 1, 0, 0])
    assert np.isfinite(result.joints).all()
    assert result.joints[1, 0, 2] == 0


def test_affine_respects_annotation_policy_and_direct_helper_keeps_legacy_masking():
    source = sample(640, 480, (80, 80, 500, 400), pieces=[(630, 200)])
    source.joints[0, 0] = [630, 200, 2]
    matrix = np.array([[1.0, 0, 40], [0, 1, 0]])
    transform = KeypointsRandomAffineTransform(0, 1, 1, 0, 0, 1, prob=1)
    transform._get_affine_matrix = lambda *args: matrix
    legacy = transform.apply_to_keypoints(source.joints, matrix, (480, 640))
    assert legacy[0, 0, 2] == legacy[1, 0, 2] == 0
    result = KeypointsCompose([transform], preserve_out_of_frame_board_annotations=True).apply_to_sample(source)
    np.testing.assert_array_equal(result.joints[0, 0], [670, 200, 2])
    assert result.joints[1, 0, 2] == 0


def test_validation_focus_letterbox_keeps_final_outside_board_annotations():
    source = sample(1920, 1080, (100, 100, 1700, 950))
    source.joints[0, 0] = [900, -300, 2]
    source.joints[0, 1] = [900, -300, 0]
    result = KeypointsCompose([KeypointsBoardFocusLetterbox()], preserve_out_of_frame_board_annotations=True).apply_to_sample(source)
    assert result.joints[0, 0, 1] < 0 and result.joints[0, 0, 2] == 2
    assert result.joints[0, 1, 2] == 0


@pytest.mark.parametrize("kind", ["bce", "focal"])
@pytest.mark.parametrize("weighted", [False, True])
def test_missing_board_points_have_zero_gradient_and_known_points_are_normalized(kind, weighted):
    criterion = ChessYoloNASPoseLoss(board_oks_sigmas=[0.1] * 9, pose_classification_loss_type=kind)
    logits = torch.zeros(1, 9, 1, requires_grad=True)
    coords = torch.zeros(1, 9, 2, requires_grad=True)
    targets = torch.full((1, 9, 2), -10.0)  # An out-of-frame coordinate is a valid regression target.
    visibility = torch.zeros(1, 9, 1)
    visibility[:, :2] = 2
    kwargs = dict(area=torch.tensor([[10000.0]]), sigmas=criterion.oks_sigmas)
    if weighted:
        kwargs.update(assigned_scores=torch.tensor([[0.5]]), assigned_scores_sum=torch.tensor(2.0))
    reg, cls = criterion._keypoint_loss(coords, targets, logits, visibility, ignore_unannotated_keypoints=True, **kwargs)
    expected = F.binary_cross_entropy_with_logits(logits[:, :2], torch.ones_like(logits[:, :2]))
    if kind == "focal":
        expected *= 0.25 * 0.5 ** 2
    if weighted:
        expected *= 0.25
    torch.testing.assert_close(cls, expected)
    (reg + cls).backward()
    assert (logits.grad[:, :2] < 0).all()
    assert (coords.grad[:, :2] > 0).all()
    assert torch.count_nonzero(logits.grad[:, 2:]) == torch.count_nonzero(coords.grad[:, 2:]) == 0

    # Unknown coordinate/logit values cannot change either loss.
    changed_coords, changed_logits = coords.detach().clone(), logits.detach().clone()
    changed_coords[:, 2:] = 1e4
    changed_logits[:, 2:] = -100
    changed = criterion._keypoint_loss(changed_coords, targets, changed_logits, visibility, ignore_unannotated_keypoints=True, **kwargs)
    torch.testing.assert_close(changed, (reg, cls))


@pytest.mark.parametrize("kind", ["bce", "focal"])
def test_all_missing_zero_and_all_known_unchanged(kind):
    criterion = ChessYoloNASPoseLoss(board_oks_sigmas=[0.1] * 9, pose_classification_loss_type=kind)
    logits = torch.randn(2, 9, 1, requires_grad=True)
    coords = torch.randn(2, 9, 2, requires_grad=True)
    args = (coords, torch.ones_like(coords), logits)
    kwargs = dict(area=torch.ones(2, 1) * 10000, sigmas=criterion.oks_sigmas)
    missing = torch.zeros_like(logits)
    reg, cls = criterion._keypoint_loss(*args, missing, ignore_unannotated_keypoints=True, **kwargs)
    assert reg.item() == cls.item() == 0
    (reg + cls).backward()
    assert torch.count_nonzero(coords.grad) == torch.count_nonzero(logits.grad) == 0
    known = torch.ones_like(logits) * 2
    torch.testing.assert_close(
        criterion._keypoint_loss(*args, known, ignore_unannotated_keypoints=True, **kwargs),
        criterion._keypoint_loss(*args, known, **kwargs),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_mixed_board_and_piece_loss_routes_ignore_mask_only_to_boards(enabled):
    criterion = ChessYoloNASPoseLoss(
        board_oks_sigmas=[0.1] * 9, piece_oks_sigma=[0.1], num_classes=13,
        ignore_unannotated_board_keypoints=enabled, iou_loss_weight=0, dfl_loss_weight=0,
    )
    poses = torch.zeros(1, 2, 9, 3)
    poses[0, 0, 0] = torch.tensor([-10.0, 100.0, 2.0])
    boxes = torch.tensor([[[0.0, 0, 640, 480], [0, 0, 640, 480]]])
    assignment = YoloNASPoseBoxesAssignmentResult(
        assigned_labels=torch.tensor([[12, 0]]), assigned_bboxes=boxes, assigned_poses=poses,
        assigned_scores=F.one_hot(torch.tensor([[12, 0]]), 13).float(),
        assigned_gt_index=torch.tensor([[0, 1]]), assigned_crowd=torch.zeros(1, 2),
    )
    logits = torch.zeros(1, 2, 9, requires_grad=True)
    coords = torch.zeros(1, 2, 9, 2, requires_grad=True)
    _, _, cls, reg = criterion._bbox_loss(
        torch.zeros(1, 2, 8), boxes, coords, logits, torch.ones(2, 1), torch.zeros(2, 2), assignment, torch.tensor(2.0), 1,
    )
    (cls + reg).backward()
    assert logits.grad[0, 0, 0] < 0
    assert coords.grad[0, 0, 0].abs().sum() > 0
    if enabled:
        assert torch.count_nonzero(logits.grad[0, 0, 1:]) == 0
    else:
        assert (logits.grad[0, 0, 1:] > 0).all()
    assert logits.grad[0, 1, 0] > 0  # Piece negative-confidence behavior is unchanged.
    assert torch.count_nonzero(logits.grad[0, 1, 1:]) == 0


@pytest.mark.parametrize("offset", [(-5.0, 20.0), (20.0, -5.0)])
def test_existing_head_decodes_out_of_frame_coordinates_in_train_and_eval(offset):
    model = ChessYoloNASPoseNDFLHeads(
        num_classes=13, in_channels=(8,), heads_list=[{"ChessYoloNASPoseDFLHead": {
            "bbox_inter_channels": 8, "pose_inter_channels": 8, "pose_regression_blocks": 1,
            "shared_stem": False, "pose_conf_in_class_head": False, "pose_block_use_repvgg": False,
            "width_mult": 1, "first_conv_group_size": 0, "stride": 8,
        }}],
    )
    with torch.no_grad():
        model.head1.pose_pred.weight.zero_()
        model.head1.pose_pred.bias.copy_(torch.tensor([*offset, 0.0] * 9))
    for training in (True, False):
        model.train(training)
        decoded, raw = model([torch.zeros(1, 8, 2, 2)])  # 16x16 image extent.
        xy = decoded[2]
        assert ((xy < 0) | (xy > 16)).all()
        torch.testing.assert_close(xy, raw[2])
        # Board selection must not clamp the coordinates either.
        scores = torch.zeros_like(decoded[1])
        scores[..., 12] = 0.9
        callback = ChessYoloNASPosePostPredictionCallback(
            pose_confidence_threshold=0.05, nms_iou_threshold=0.7, pre_nms_max_predictions=300, post_nms_max_predictions=50,
        )
        prediction = callback(((decoded[0], scores, xy, decoded[3]),))[0]
        assert prediction.labels.tolist() == [12]
        assert ((prediction.poses < 0) | (prediction.poses > 16)).all()
