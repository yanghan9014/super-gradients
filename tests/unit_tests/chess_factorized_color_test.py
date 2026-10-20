"""Type sharing, positive-only color supervision, assignment and inference compatibility."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from super_gradients.training.losses.chess_yolo_nas_pose_loss import ChessYoloNASPoseLoss
from super_gradients.training.metrics.board_homography_metrics import BOARD_REFERENCE_POINTS
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.chess_yolo_nas_pose_ndfl_heads import ChessYoloNASPoseNDFLHeads
from super_gradients.training.utils.chess_classification import chess_joint_scores


def heads(factorized=True):
    return ChessYoloNASPoseNDFLHeads(
        num_classes=13, in_channels=(8, 8, 8), factorized_piece_color=factorized,
        heads_list=[{"ChessYoloNASPoseDFLHead": {
            "bbox_inter_channels": 8, "pose_inter_channels": 8, "pose_regression_blocks": 1,
            "shared_stem": False, "pose_conf_in_class_head": False, "pose_block_use_repvgg": False,
            "width_mult": 1, "first_conv_group_size": 0, "stride": stride,
        }} for stride in (8, 16, 32)],
    )


def criterion(**kwargs):
    return ChessYoloNASPoseLoss(
        num_classes=13, board_oks_sigmas=[0.1] * 9, piece_oks_sigma=[0.35],
        factorized_piece_color=True, **kwargs,
    )


def features():
    return [torch.randn(2, 8, size, size) for size in (8, 4, 2)]


def targets(empty=False):
    if empty:
        return torch.empty(0, 5), torch.empty(0, 9, 4), torch.empty(0, 2, dtype=torch.long)
    # The second image is a negative sample. Include both sides, multiple types and a board.
    boxes = torch.tensor([[0, 8, 8, 24, 24], [0, 32, 8, 48, 24], [0, 8, 32, 24, 48], [0, 0, 0, 64, 64.]], dtype=torch.float32)
    labels = torch.tensor([[0, 0], [0, 6], [0, 9], [0, 12]])
    joints = torch.zeros(4, 9, 4)
    joints[:3, 0, 1:] = torch.tensor([[16., 16, 2], [40., 16, 2], [16., 40, 2]])
    joints[3, :, 1:3] = torch.tensor(BOARD_REFERENCE_POINTS) * 8
    joints[3, :, 3] = 2
    return boxes, joints, labels


def test_joint_scores_preserve_all_labels_and_type_mass_and_independent_board():
    types = torch.full((1, 12, 7), -8.)
    colors = torch.full((1, 12, 2), -8.)
    for label in range(12):
        types[0, label, label % 6] = 8
        colors[0, label, label // 6] = 8
    scores = chess_joint_scores(types, colors)
    assert scores.argmax(-1).tolist() == [list(range(12))]
    torch.testing.assert_close(scores[..., :6] + scores[..., 6:12], types[..., :6].sigmoid())
    torch.testing.assert_close(scores[..., 12], types[..., 6].sigmoid())
    swapped = chess_joint_scores(types, colors.flip(-1))
    torch.testing.assert_close(swapped[..., :6], scores[..., 6:12])
    torch.testing.assert_close(swapped[..., 12], scores[..., 12])


@pytest.mark.parametrize("kind", ["focal", "bce"])
def test_same_shape_shares_detection_gradient_and_color_only_supervises_pieces(kind):
    loss = criterion(classification_loss_type=kind, log_pose_diagnostics=True)
    types = torch.zeros(1, 4, 7, requires_grad=True)
    colors = torch.zeros(1, 4, 2, requires_grad=True)
    assigned = torch.zeros(1, 4, 13)
    assigned[0, 0, 0] = assigned[0, 1, 6] = assigned[0, 2, 12] = 1
    result = loss._classification_loss(types, assigned, assigned.sum(), colors)
    result.backward()
    torch.testing.assert_close(types.grad[0, 0], types.grad[0, 1])
    assert colors.grad[0, 0, 0] < 0 < colors.grad[0, 0, 1]
    torch.testing.assert_close(colors.grad[0, 1], -colors.grad[0, 0])
    assert torch.count_nonzero(colors.grad[0, 2:]) == 0  # Board and background.
    assert (types.grad[0, 3] > 0).all()  # Background still suppresses detection.
    diagnostics = loss._pose_diagnostics
    torch.testing.assert_close(diagnostics["piece_loss_cls"], diagnostics["piece_loss_type"] + diagnostics["piece_loss_color"])
    torch.testing.assert_close(result, diagnostics["piece_loss_cls"] + diagnostics["board_loss_cls"])


def test_color_loss_uses_assignment_quality_and_handles_confident_half_precision_errors():
    loss = criterion(piece_color_loss_weight=2, log_pose_diagnostics=True)
    types = torch.zeros(1, 3, 7, requires_grad=True)
    colors = torch.tensor([[[-10000., 10000.], [0, 0], [10000, -10000]]], dtype=torch.float16, requires_grad=True)
    assigned = torch.zeros(1, 3, 13)
    assigned[0, 0, 0] = 0.25
    assigned[0, 1, 6] = 0.75
    assigned[0, 2, 12] = 1
    result = loss._classification_loss(types, assigned, assigned.sum(), colors)
    expected_color = (20000 * 0.25 + torch.log(torch.tensor(2.)) * 0.75) / 2 * 2
    torch.testing.assert_close(loss._pose_diagnostics["piece_loss_color"], expected_color)
    result.backward()
    assert torch.isfinite(result) and torch.isfinite(colors.grad).all()
    assert colors.grad[0, 0, 0] < 0 and torch.count_nonzero(colors.grad[0, 2]) == 0


@pytest.mark.parametrize("empty", [False, True])
def test_multiscale_forward_assignment_and_backward_with_negative_images(empty):
    model = heads()
    loss = criterion(log_pose_diagnostics=True, assigner_multiply_by_pose_oks=True)
    outputs = model(features())
    assert outputs[0][1].shape == (2, 84, 13)
    assert outputs[1][0].shape == (2, 84, 7)
    assert outputs[1][-1].shape == (2, 84, 2)
    seen_scores = []
    handle = loss.assigner.register_forward_pre_hook(lambda module, args, kwargs: seen_scores.append(kwargs["pred_scores"]), with_kwargs=True)
    total, logged = loss(outputs, targets(empty))
    handle.remove()
    torch.testing.assert_close(seen_scores[0], outputs[0][1].detach())
    assert torch.isfinite(total) and torch.isfinite(logged).all()
    assert len(logged) == len(loss.component_names)
    total.backward()
    color_grad = model.head1.color_pred.weight.grad
    assert color_grad is not None and torch.isfinite(color_grad).all()
    if empty:
        assert torch.count_nonzero(color_grad) == 0
    else:
        assert torch.count_nonzero(color_grad) > 0
    diagnostics = dict(zip(loss.component_names, logged))
    torch.testing.assert_close(diagnostics["loss_cls"], diagnostics["board_loss_cls"] + diagnostics["piece_loss_cls"])


@pytest.mark.parametrize("factorized", [False, True])
def test_checkpoint_roundtrip_and_export_preserve_decoded_output_contract(factorized):
    model = heads(factorized).eval()
    restored = heads(factorized).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    inputs = features()
    with torch.no_grad():
        decoded, raw = model(inputs)
        torch.testing.assert_close(restored(inputs)[0], decoded)
        assert len(raw) == (9 if factorized else 8)
        assert decoded[1].shape[-1] == 13
        model.inference_mode = True
        traced = torch.jit.trace(model, (inputs,), strict=False)
        torch.testing.assert_close(traced(inputs), decoded)


def test_legacy_checkpoint_is_not_silently_interpreted_as_factorized():
    with pytest.raises(RuntimeError):
        heads().load_state_dict(heads(False).state_dict(), strict=True)
    outputs = heads()(features())
    with pytest.raises(ValueError, match="same factorized_piece_color"):
        ChessYoloNASPoseLoss(num_classes=13, board_oks_sigmas=[0.1] * 9)(outputs, targets())
    with pytest.raises(ValueError, match="same factorized_piece_color"):
        criterion()(heads(False)(features()), targets())
    model = heads()
    model.replace_num_classes(13, lambda *args: pytest.fail("Must keep the seven-channel detector"))
    with pytest.raises(ValueError, match="num_classes=13"):
        model.replace_num_classes(12, lambda *args: None)


def test_full_model_training_and_torch_export():
    from super_gradients.training import models

    recipes = Path(__file__).resolve().parents[2] / "src/super_gradients/recipes"
    arch = OmegaConf.load(recipes / "arch_params/chess_yolo_nas_pose_s_arch_params.yaml")
    arch.heads.ChessYoloNASPoseNDFLHeads.piece_objectness = False
    model = models.get("yolo_nas_pose_s", arch_params=arch, num_classes=13, pretrained_weights=None)
    inputs = torch.randn(2, 3, 64, 64)
    total, _ = criterion()(model(inputs), targets())
    total.backward()
    assert torch.isfinite(total)
    for scale in (model.heads.head1, model.heads.head2, model.heads.head3):
        assert torch.isfinite(scale.color_pred.weight.grad).all()

    model.eval()
    with torch.no_grad():
        decoded = model(inputs)[0]
        model.prep_model_for_conversion(input_size=(64, 64))
        assert model.heads.inference_mode
        converted = model(inputs)
        assert len(converted) == 4 and converted[1].shape[-1] == 13
        torch.testing.assert_close(converted, decoded, rtol=1e-4, atol=1e-4)
        if hasattr(torch, "export"):
            exported = torch.export.export(model, (inputs,))
            torch.testing.assert_close(exported.module()(inputs), converted)


@pytest.mark.parametrize("size", ["n", "s", "l"])
def test_chess_recipes_enable_matching_head_and_loss(size):
    recipes = Path(__file__).resolve().parents[2] / "src/super_gradients/recipes"
    arch = OmegaConf.load(recipes / f"arch_params/chess_yolo_nas_pose_{size}_arch_params.yaml")
    train = OmegaConf.load(recipes / "training_hyperparams/chess_coco2017_yolo_nas_pose_train_params.yaml")
    cfg = OmegaConf.create({"arch_params": arch, "training_hyperparams": train, "dataset_params": {"num_classes": 13}})
    assert cfg.arch_params.heads.ChessYoloNASPoseNDFLHeads.factorized_piece_color
    assert cfg.training_hyperparams.criterion_params.factorized_piece_color
    assert cfg.training_hyperparams.criterion_params.piece_color_loss_weight == 1
    cfg.arch_params.heads.ChessYoloNASPoseNDFLHeads.factorized_piece_color = False
    assert not cfg.training_hyperparams.criterion_params.factorized_piece_color


@pytest.mark.parametrize("kwargs", [{"num_classes": 12}, {"board_class_id": 0}])
def test_factorization_rejects_incompatible_label_order(kwargs):
    settings = dict(num_classes=13, board_oks_sigmas=[0.1] * 9, factorized_piece_color=True)
    settings.update(kwargs)
    with pytest.raises(ValueError, match="board_class_id=12"):
        ChessYoloNASPoseLoss(**settings)


@pytest.mark.parametrize("weight", [-1, float("inf"), float("nan")])
def test_color_weight_validation(weight):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        criterion(piece_color_loss_weight=weight)
