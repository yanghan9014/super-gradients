"""Separate presence/type/color contract, supervision, and lossless postprocessing."""
import math

import pytest
import torch
from super_gradients.training.losses.chess_yolo_nas_pose_loss import ChessYoloNASPoseLoss
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.chess_yolo_nas_pose_ndfl_heads import ChessYoloNASPoseNDFLHeads
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.chess_yolo_nas_pose_post_prediction_callback import ChessYoloNASPosePostPredictionCallback
from super_gradients.training.utils.chess_classification import chess_joint_scores
from tests.unit_tests.chess_factorized_color_test import features, targets


def heads():
    return ChessYoloNASPoseNDFLHeads(
        num_classes=13, in_channels=(8, 8, 8), factorized_piece_color=True, piece_objectness=True,
        heads_list=[{"ChessYoloNASPoseDFLHead": {
            "bbox_inter_channels": 8, "pose_inter_channels": 8, "pose_regression_blocks": 1,
            "shared_stem": False, "pose_conf_in_class_head": False, "pose_block_use_repvgg": False,
            "width_mult": 1, "first_conv_group_size": 0, "stride": stride,
        }} for stride in (8, 16, 32)])


def criterion(**kwargs):
    return ChessYoloNASPoseLoss(num_classes=13, board_oks_sigmas=[0.1]*9, piece_oks_sigma=[0.35],
        factorized_piece_color=True, piece_objectness=True, log_pose_diagnostics=True, **kwargs)


def test_probabilities_sum_to_objectness_and_board_is_unchanged():
    types, colors, obj = torch.randn(2, 5, 7), torch.randn(2, 5, 2), torch.randn(2, 5, 1)
    scores = chess_joint_scores(types, colors, obj)
    torch.testing.assert_close(scores[..., :12].sum(-1), obj.sigmoid()[..., 0])
    torch.testing.assert_close(scores[..., 12], types[..., 6].sigmoid())
    torch.testing.assert_close(scores[..., :6] + scores[..., 6:12], obj.sigmoid()*types[..., :6].softmax(-1))


@pytest.mark.parametrize('identity', ['type', 'color'])
def test_identity_predictions_do_not_change_assignment_quality_or_geometry_loss(identity):
    torch.manual_seed(0)
    decoded, raw = heads()(features())
    raw = list(raw)
    # Provide usable piece geometry so equality cannot pass on all-background or
    # zero-quality assignments. Each box extends one stride around its anchor.
    distribution = torch.full_like(raw[1], -20).reshape(*raw[1].shape[:2], 4, -1)
    distribution[..., 1] = 20
    raw[1] = distribution.flatten(-2)
    raw[2] = raw[5][None, :, None, :].expand(2, -1, 9, -1).clone()
    raw[0] = torch.zeros_like(raw[0], requires_grad=True)
    raw[8] = torch.zeros_like(raw[8], requires_grad=True)
    loss = criterion(assigner_multiply_by_pose_oks=True)
    seen = []
    handle = loss.assigner.register_forward_hook(
        lambda module, args, kwargs, result: seen.append((kwargs['pred_scores'], result)),
        with_kwargs=True,
    )
    try:
        _, original_logs = loss((decoded, tuple(raw)), targets())
        changed = list(raw)
        if identity == 'type':
            changed[0] = torch.cat((torch.randn_like(raw[0][..., :6]) * 8, raw[0][..., 6:7]), -1)
        else:
            changed[8] = torch.randn_like(raw[8]) * 8
        _, changed_logs = loss((decoded, tuple(changed)), targets())
    finally:
        handle.remove()

    original_scores, original = seen[0]
    changed_scores, altered = seen[1]
    assert not original_scores.requires_grad
    torch.testing.assert_close(original_scores[..., :12], raw[9].detach().sigmoid().expand(-1, -1, 12))
    torch.testing.assert_close(original_scores[..., 12], raw[0].detach()[..., 6].sigmoid())
    torch.testing.assert_close(changed_scores, original_scores)
    assert original.assigned_scores[..., :12].sum() > 0
    assert original.assigned_scores[..., 12].sum() > 0
    for name, value in vars(original).items():
        if value is not None:
            torch.testing.assert_close(getattr(altered, name), value)
    # Identity still receives supervision; only assignment and geometry are independent.
    assert not torch.isclose(changed_logs[0], original_logs[0])
    torch.testing.assert_close(changed_logs[1:5], original_logs[1:5])


@pytest.mark.parametrize('kind', ['bce', 'focal'])
def test_quality_targets_and_weighted_identity_supervision_keep_board_loss(kind):
    loss = criterion(classification_loss_type=kind)
    types = torch.zeros(1, 4, 7, requires_grad=True)
    colors = torch.zeros(1, 4, 2, requires_grad=True)
    obj = torch.zeros(1, 4, 1, requires_grad=True)
    labels = torch.tensor([[3, 9, 12, 13]])
    crowd = torch.zeros_like(labels, dtype=torch.bool)
    scores = torch.zeros(1, 4, 13)
    scores[0, 0, 3], scores[0, 1, 9], scores[0, 2, 12] = .01, .9, .7
    total = loss._classification_loss(types, scores, scores.sum(), colors, obj, labels, crowd)
    total.backward()
    # At zero logits, low-quality assignments push confidence down and strong
    # assignments push it up. Normalize by total quality, including the board.
    quality = torch.tensor([.01, .9, 0., 0.])
    expected_gradient = (.5 - quality) / scores.sum()
    if kind == 'focal':
        # SG's quality focal form is |sigmoid(logit) - q|^2 * BCE(logit, q).
        expected_gradient *= (.5 - quality).square() + math.log(2) / 2
    torch.testing.assert_close(obj.grad.flatten(), expected_gradient)
    assert (obj.grad[0, 2:] > 0).all()
    assert torch.count_nonzero(types.grad[0, 2:, :6]) == 0
    assert torch.count_nonzero(colors.grad[0, 2:]) == 0
    torch.testing.assert_close(types.grad[0, 0, :6] / .01, types.grad[0, 1, :6] / .9)
    torch.testing.assert_close(types.grad[0, 0, 3], torch.tensor(-5 / 6 * .01 / 1.61))
    torch.testing.assert_close(colors.grad[0, 0, 0], torch.tensor(-.5 * .01 / 1.61))
    assert colors.grad[0, 0, 0] < 0 < colors.grad[0, 1, 0]
    old = ChessYoloNASPoseLoss(num_classes=13, board_oks_sigmas=[.1]*9,
        factorized_piece_color=True, classification_loss_type=kind, log_pose_diagnostics=True)
    old._classification_loss(types.detach(), scores, scores.sum(), colors.detach())
    torch.testing.assert_close(old._pose_diagnostics['board_loss_cls'], loss._pose_diagnostics['board_loss_cls'])
    d = loss._pose_diagnostics
    torch.testing.assert_close(total.detach(), d['board_loss_cls']+d['piece_loss_cls'])
    torch.testing.assert_close(d['piece_loss_cls'], d['piece_loss_objectness']+d['piece_loss_type']+d['piece_loss_color'])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_default_objectness_focal_suppresses_easy_examples_and_keeps_hard_negatives(dtype):
    labels = torch.tensor([[0, 13, 13, 0]])
    crowd = torch.tensor([[False, False, False, True]])
    scores = torch.zeros(1, 4, 13)
    scores[0, 0, 0] = 1.
    gradients = []
    for options in ({}, {'classification_loss_type': 'bce'}):
        loss = criterion(**options)
        types = torch.zeros(1, 4, 7, dtype=dtype, requires_grad=True)
        colors = torch.zeros(1, 4, 2, dtype=dtype, requires_grad=True)
        # Easy positive, easy negative, hard negative, ignored crowd candidate.
        obj = torch.tensor([[[12.], [-12.], [12.], [0.]]], dtype=dtype, requires_grad=True)
        total = loss._classification_loss(types, scores, scores.sum().clamp(min=1), colors, obj, labels, crowd)
        total.backward()
        assert torch.isfinite(total)
        assert all(torch.isfinite(t.grad).all() for t in (types, colors, obj))
        gradients.append(obj.grad.flatten().float())
    focal, bce = gradients
    assert (bce[:2].abs() > 0).all()
    assert (focal[:2].abs() < bce[:2].abs() * 1e-8).all()
    assert focal[2] > .9 and bce[2] > .9
    assert focal[3] == 0 and bce[3] == 0


@pytest.mark.parametrize('kind', ['bce', 'focal'])
def test_objectness_matches_soft_quality_instead_of_forcing_one(kind):
    loss = criterion(classification_loss_type=kind)
    quality = torch.tensor([.1, .8])
    obj = torch.logit(quality).reshape(1, 2, 1).requires_grad_()
    scores = torch.zeros(1, 2, 13)
    scores[0, :, 0] = quality
    total = loss._classification_loss(
        torch.zeros(1, 2, 7), scores, scores.sum().clamp(min=1),
        torch.zeros(1, 2, 2), obj, torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 2, dtype=torch.bool),
    )
    total.backward()
    torch.testing.assert_close(obj.grad, torch.zeros_like(obj), atol=1e-7, rtol=0)


@pytest.mark.parametrize('kind', ['bce', 'focal'])
def test_zero_quality_assignments_and_crowd_do_not_supervise_identity(kind):
    loss = criterion(classification_loss_type=kind)
    types = torch.zeros(1, 3, 7, requires_grad=True)
    colors = torch.zeros(1, 3, 2, requires_grad=True)
    obj = torch.zeros(1, 3, 1, requires_grad=True)
    labels = torch.tensor([[0, 12, 1]])
    crowd = torch.tensor([[False, False, True]])
    scores = torch.zeros(1, 3, 13)
    scores[0, 1, 12] = .3
    # A geometrically assigned piece can have zero quality. It must not receive
    # hard-positive objectness or identity supervision merely from its label.
    total = loss._classification_loss(types, scores, scores.sum().clamp(min=1), colors, obj, labels, crowd)
    total.backward()
    assert torch.isfinite(total)
    assert obj.grad[0, 0, 0] > 0
    assert obj.grad[0, 2, 0] == 0
    assert torch.count_nonzero(types.grad[..., :6]) == 0
    assert torch.count_nonzero(colors.grad) == 0


@pytest.mark.parametrize('empty', [False, True])
def test_assignment_backward_and_export_roundtrip(empty):
    model = heads()
    output = model(features())
    assert len(output[0]) == 7 and len(output[1]) == 10
    assert [x.shape[-1] for x in output[0][4:]] == [1, 6, 2]
    loss = criterion(assigner_multiply_by_pose_oks=True)
    total, logs = loss(output, targets(empty))
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(logs).all()
    assert len(logs) == len(loss.component_names)
    for scale in (model.head1, model.head2, model.head3):
        assert torch.isfinite(scale.objectness_pred.weight.grad).all()
        if empty:
            assert not torch.count_nonzero(scale.color_pred.weight.grad)
            assert not torch.count_nonzero(scale.cls_pred.weight.grad[:6])
    restored = heads()
    restored.load_state_dict(model.state_dict(), strict=True)
    model.eval(); model.prep_model_for_conversion()
    inputs = features()
    with torch.no_grad():
        decoded = model(inputs)
        exported = torch.export.export(model, (inputs,))
        torch.testing.assert_close(exported.module()(inputs), decoded)


def test_ambiguous_identity_survives_objectness_filter_and_keeps_vectors():
    callback = ChessYoloNASPosePostPredictionCallback(pose_confidence_threshold=.5,
        nms_iou_threshold=.7, pre_nms_max_predictions=10, post_nms_max_predictions=10)
    boxes = torch.tensor([[[0., 0, 10, 10], [20, 20, 30, 30]]])
    types, colors = torch.zeros(1, 2, 7), torch.zeros(1, 2, 2)
    types[..., 6] = -10
    obj = torch.tensor([[[3.], [-3.]]])
    scores = chess_joint_scores(types, colors, obj)
    assert scores[..., :12].max() < .1
    probabilities = (obj.sigmoid(), types[..., :6].softmax(-1), colors.softmax(-1))
    result = callback(((boxes, scores, torch.zeros(1, 2, 9, 2), torch.ones(1, 2, 9), *probabilities), ()))[0]
    assert len(result.labels) == 1
    torch.testing.assert_close(result.piece_objectness, probabilities[0][0, :1])
    torch.testing.assert_close(result.piece_type_probabilities, probabilities[1][0, :1])
    torch.testing.assert_close(result.piece_color_probabilities, probabilities[2][0, :1])


def test_full_recipe_model_backward_and_export():
    from pathlib import Path
    from omegaconf import OmegaConf
    from super_gradients.training import models
    recipes = Path(__file__).resolve().parents[2] / 'src/super_gradients/recipes'
    arch = OmegaConf.load(recipes / 'arch_params/chess_yolo_nas_pose_s_arch_params.yaml')
    model = models.get('yolo_nas_pose_s', arch_params=arch, num_classes=13, pretrained_weights=None)
    x = torch.randn(2, 3, 64, 64)
    total, logs = criterion()(model(x), targets())
    total.backward()
    assert torch.isfinite(total) and torch.isfinite(logs).all()
    assert model.heads.head1.objectness_pred.weight.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        before = model(x)[0]
        model.prep_model_for_conversion(input_size=(64, 64))
        assert len(model(x)) == 7
        torch.testing.assert_close(model(x), before, rtol=1e-4, atol=1e-4)
        exported = torch.export.export(model, (x,))
        torch.testing.assert_close(exported.module()(x), model(x))


def test_prediction_pipeline_preserves_separate_fields():
    from super_gradients.training.pipelines.pipelines import PoseEstimationPipeline
    model = heads().eval()
    with torch.no_grad():
        output = model(features())
    pipeline = PoseEstimationPipeline.__new__(PoseEstimationPipeline)
    pipeline.post_prediction_callback = ChessYoloNASPosePostPredictionCallback(
        pose_confidence_threshold=0, nms_iou_threshold=.7, pre_nms_max_predictions=10, post_nms_max_predictions=10)
    expected = pipeline.post_prediction_callback(output)
    decoded = pipeline._decode_model_output(output, torch.zeros(2, 3, 64, 64))
    for actual, reference in zip(decoded, expected):
        for name in ('piece_objectness', 'piece_type_probabilities', 'piece_color_probabilities'):
            torch.testing.assert_close(torch.from_numpy(getattr(actual, name)), getattr(reference, name))
