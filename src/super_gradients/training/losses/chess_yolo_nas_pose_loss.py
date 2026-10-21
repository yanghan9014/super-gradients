import dataclasses
import math
from typing import Mapping, Tuple, Union, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.common.environment.ddp_utils import get_world_size, is_distributed

from super_gradients.training.utils.bbox_utils import batch_distance2bbox
from super_gradients.training.utils.chess_classification import chess_joint_scores
from .functional import bbox_ciou_loss

from .ppyolo_loss import GIoULoss, batch_iou_similarity, check_points_inside_bboxes, gather_topk_anchors, compute_max_iou_anchor



@dataclasses.dataclass
class YoloNASPoseBoxesAssignmentResult:
    """
    This dataclass stores result of assignment of predicted boxes to ground truth boxes for YoloNASPose model.
    It produced by YoloNASPoseTaskAlignedAssigner and is used by YoloNASPoseLoss to compute the loss.

    For all fields, first dimension is batch dimension, second dimension is number of anchors.

    :param assigned_labels: Tensor of shape (B, L) - Assigned gt labels for each anchor location
    :param assigned_bboxes: Tensor of shape (B, L, 4) - Assigned groundtruth boxes in XYXY format for each anchor location
    :param assigned_scores: Tensor of shape (B, L, C) - Assigned scores for each anchor location
    :param assigned_poses: Tensor of shape (B, L, Num Joints, 3) - Assigned groundtruth poses for each anchor location
    :param assigned_gt_index: Tensor of shape (B, L) - Index of assigned groundtruth box for each anchor location
    :param assigned_crowd: Tensor of shape (B, L) - Whether the assigned groundtruth box is crowd
    """

    assigned_labels: Tensor
    assigned_bboxes: Tensor
    assigned_poses: Tensor
    assigned_scores: Tensor
    assigned_gt_index: Tensor
    assigned_crowd: Tensor
    assigned_rotation_eligible: Optional[Tensor] = None


# Board keypoint order is (a1, a8, h1, h8, center, a45, h45, 1de, 8de). Rotating the board 180
# degrees relabels a1<->h8, a8<->h1, a45<->h45 and 1de<->8de, leaving the center where it is. This
# is the only relabelling that keeps square colours consistent, since a1 must sit on a dark square.
BOARD_180_PERMUTATION = [3, 2, 1, 0, 4, 6, 5, 8, 7]


@torch.no_grad()
def board_rotation_choice(predicted_coords: Tensor, target_coords: Tensor, target_visibility: Tensor) -> Tensor:
    """One complete board permutation, shared by assignment and regression.

    Leading dimensions broadcast. Masked MSE avoids arbitrary ties when both OKS
    values saturate near zero early in training. Ties retain canonical orientation.
    """
    permutation = torch.as_tensor(BOARD_180_PERMUTATION, device=target_coords.device)

    def distance(coords, visibility):
        mask = (visibility > 0).float()
        squared = ((predicted_coords.float() - coords.float()) ** 2).sum(-1, keepdim=True)
        return (squared * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp(min=1)

    return distance(target_coords[..., permutation, :], target_visibility[..., permutation, :]) < distance(target_coords, target_visibility)


def batch_pose_oks(
    gt_keypoints: torch.Tensor,
    pred_keypoints: torch.Tensor,
    gt_bboxes_xyxy: torch.Tensor,
    sigmas: torch.Tensor,
    eps: float = 1e-9,
    gt_labels: Optional[torch.Tensor] = None,
    board_class_id: int = 12,
    piece_sigma: Optional[torch.Tensor] = None,
    rotation_eligible: Optional[Tensor] = None,
) -> Tensor:
    """
    Calculate batched OKS (Object Keypoint Similarity) between two sets of keypoints.

    :param gt_keypoints:   Joints with the shape [N, M1, Num Joints, 3]
    :param gt_bboxes_xyxy: Array of bboxes with the shape [N, M1, 4] in XYXY format
    :param pred_keypoints: Joints with the shape [N, M1, Num Joints, 3]
    :param sigmas:         Sigmas with the shape [Num Joints]
    :param eps (float):    Small constant for numerical stability
    :return iou:           OKS between gt_keypoints and pred_keypoints with the shape [N, M1, M2]
    """

    joints1_xy = gt_keypoints[:, :, :, 0:2].unsqueeze(2)  # [N, M1, 1, Num Joints, 2]
    joints2_xy = pred_keypoints[:, :, :, 0:2].unsqueeze(1)  # [N, 1, M2, Num Joints, 2]

    # Infer pose area from bbox area * 0.53 (COCO heuristic)
    area = (gt_bboxes_xyxy[:, :, 2] - gt_bboxes_xyxy[:, :, 0]) * (gt_bboxes_xyxy[:, :, 3] - gt_bboxes_xyxy[:, :, 1]) * 0.53  # [N, M1]
    area = area[:, :, None, None]  # [N, M1, 1, 1]

    if gt_labels is not None and piece_sigma is not None:
        e = torch.zeros(joints1_xy.shape[0], joints1_xy.shape[1], joints2_xy.shape[2], joints1_xy.shape[3], device=joints1_xy.device)
        board_mask = (gt_labels.squeeze(-1) == board_class_id)
        
        if board_mask.any():
            j1 = joints1_xy[board_mask]
            j2 = joints2_xy.expand(-1, board_mask.shape[1], -1, -1, -1)[board_mask]
            d = ((j1 - j2) ** 2).sum(dim=-1)
            a = area.expand(-1, -1, joints2_xy.shape[2], -1)[board_mask].squeeze(-1).unsqueeze(-1)
            sig = sigmas.reshape(1, 1, -1)
            e[board_mask] = d / (2 * sig) ** 2 / (a + eps) / 2
            
        piece_mask = ~board_mask
        if piece_mask.any():
            j1 = joints1_xy[piece_mask][:, :, 0:1, :]
            j2 = joints2_xy.expand(-1, piece_mask.shape[1], -1, -1, -1)[piece_mask][:, :, 0:1, :]
            d = ((j1 - j2) ** 2).sum(dim=-1)
            a = area.expand(-1, -1, joints2_xy.shape[2], -1)[piece_mask].squeeze(-1).unsqueeze(-1)
            sig = piece_sigma.reshape(1, 1, 1)
            e[piece_mask, :, 0:1] = d / (2 * sig) ** 2 / (a + eps) / 2
    else:
        d = ((joints1_xy - joints2_xy) ** 2).sum(dim=-1, keepdim=False)  # [N, M1, M2, Num Joints]
        sigmas = sigmas.reshape([1, 1, 1, -1])  # [1, 1, 1, Num Keypoints]
        e = d / (2 * sigmas) ** 2 / (area + eps) / 2

    oks = torch.exp(-e)  # [N, M1, M2, Num Keypoints]

    joints1_visiblity = gt_keypoints[:, :, :, 2].gt(0).float().unsqueeze(2)  # [N, M1, 1, Num Keypoints]
    num_visible_joints = joints1_visiblity.sum(dim=-1, keepdim=False)  # [N, M1, M2]
    mean_oks = (oks * joints1_visiblity).sum(dim=-1, keepdim=False) / (num_visible_joints + eps)  # [N, M1, M2]

    if rotation_eligible is not None and rotation_eligible.any():
        eligible = rotation_eligible.squeeze(-1).bool()
        if gt_labels is not None:
            eligible = eligible & (gt_labels.squeeze(-1) == board_class_id)
        batch_ids = eligible.nonzero(as_tuple=True)[0]
        known = gt_keypoints[eligible]
        pred = pred_keypoints[batch_ids]
        rotated = board_rotation_choice(pred, known[:, None, :, :2], known[:, None, :, 2:3])
        perm = torch.as_tensor(BOARD_180_PERMUTATION, device=known.device)
        chosen = torch.where(rotated[..., None, None], known[:, None, perm], known[:, None])
        distance = ((pred.float() - chosen[..., :2].float()) ** 2).sum(-1)
        selected_area = area[eligible].reshape(-1, 1, 1)
        exponent = distance / (2 * sigmas.reshape(1, 1, -1)) ** 2 / (selected_area + eps) / 2
        mask = (chosen[..., 2] > 0).float()
        mean_oks[eligible] = (torch.exp(-exponent) * mask).sum(-1) / (mask.sum(-1) + eps)
    return mean_oks


class YoloNASPoseTaskAlignedAssigner(nn.Module):
    """
    Task-aligned assigner repurposed from YoloNAS for pose estimation task

    This class is almost identical to TaskAlignedAssigner, but it also assigns poses and unlike in
    object detection where assigned scores are product of IoU and class confidence, in pose estimation
    final assignment score is product of pose OKS and bbox IoU. This was empirically found to provide
    superior performance that the original approach.
    """

    def __init__(self, sigmas: Tensor, topk: int = 13, alpha: float = 1.0, beta=6.0, eps=1e-9, multiply_by_pose_oks: bool = False, piece_sigma: Optional[Tensor] = None, board_class_id: int = 12):
        """

        :param sigmas:               Sigmas for OKS calculation
        :param topk:                 Maximum number of anchors that is selected for each gt box
        :param alpha:                Power factor for class probabilities of predicted boxes (Used compute alignment metric)
        :param beta:                 Power factor for IoU score of predicted boxes (Used compute alignment metric)
        :param eps:                  Small constant for numerical stability
        :param multiply_by_pose_oks: Whether to multiply alignment metric by pose OKS
        """
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.sigmas = sigmas
        self.multiply_by_pose_oks = multiply_by_pose_oks
        self.piece_sigma = piece_sigma
        self.board_class_id = board_class_id

    @torch.no_grad()
    def forward(
        self,
        pred_scores: Tensor,
        pred_bboxes: Tensor,
        pred_pose_coords: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_bboxes: Tensor,
        gt_poses: Tensor,
        gt_crowd: Tensor,
        pad_gt_mask: Tensor,
        bg_index: int,
        rotation_eligible: Optional[Tensor] = None,
    ) -> YoloNASPoseBoxesAssignmentResult:
        """
        This code is based on https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.

        :param pred_scores: Tensor (float32): predicted class probability, shape(B, L, C)
        :param pred_bboxes: Tensor (float32): predicted bounding boxes, shape(B, L, 4)
        :param pred_pose_coords: Tensor (float32): predicted poses, shape(B, L, Num Keypoints, 2)
        :param anchor_points: Tensor (float32): pre-defined anchors, shape(L, 2), xy format
        :param gt_labels: Tensor (int64|int32): Label of gt_bboxes, shape(B, n, 1)
        :param gt_bboxes: Tensor (float32): Ground truth bboxes, shape(B, n, 4)
        :param gt_poses: Tensor (float32): Ground truth poses, shape(B, n, Num Keypoints, 3)
        :param gt_crowd: Tensor (int): Whether the gt is crowd, shape(B, n, 1)
        :param pad_gt_mask: Tensor (float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        :param bg_index: int ( background index
        :param gt_scores: Tensor (one, float32) Score of gt_bboxes, shape(B, n, 1)
        :return:
            - assigned_labels, Tensor of shape (B, L)
            - assigned_bboxes, Tensor of shape (B, L, 4)
            - assigned_scores, Tensor of shape (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, _, num_keypoints, _ = pred_pose_coords.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full([batch_size, num_anchors], bg_index, dtype=torch.long, device=gt_labels.device)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4], device=gt_labels.device)
            assigned_poses = torch.zeros([batch_size, num_anchors, num_keypoints, 3], device=gt_labels.device)
            assigned_scores = torch.zeros([batch_size, num_anchors, num_classes], device=gt_labels.device)
            assigned_gt_index = torch.zeros([batch_size, num_anchors], dtype=torch.long, device=gt_labels.device)
            assigned_crowd = torch.zeros([batch_size, num_anchors], dtype=torch.bool, device=gt_labels.device)

            return YoloNASPoseBoxesAssignmentResult(
                assigned_labels=assigned_labels,
                assigned_bboxes=assigned_bboxes,
                assigned_scores=assigned_scores,
                assigned_gt_index=assigned_gt_index,
                assigned_poses=assigned_poses,
                assigned_crowd=assigned_crowd,
            )

        # compute iou between gt and pred bbox, [B, n, L]
        ious = batch_iou_similarity(gt_bboxes, pred_bboxes)

        if self.multiply_by_pose_oks:
            pose_oks = batch_pose_oks(gt_poses, pred_pose_coords, gt_bboxes, self.sigmas.to(pred_pose_coords.device), eps=self.eps, gt_labels=gt_labels, board_class_id=self.board_class_id, piece_sigma=self.piece_sigma.to(pred_pose_coords.device) if self.piece_sigma is not None else None, rotation_eligible=rotation_eligible)
            ious = ious * pose_oks

        # # gather pred bboxes class score
        # pred_scores = torch.permute(pred_scores, [0, 2, 1])
        batch_ind = torch.arange(end=batch_size, dtype=gt_labels.dtype, device=gt_labels.device).unsqueeze(-1)
        # gt_labels_ind = torch.stack([batch_ind.tile([1, num_max_boxes]), gt_labels.squeeze(-1)], dim=-1)
        # bbox_cls_scores = pred_scores[gt_labels_ind[..., 0], gt_labels_ind[..., 1]]

        B, L, C = pred_scores.shape
        gt_labels_squeezed = gt_labels.squeeze(-1).long()   # [B, n]
        # one-hot over classes: [B, n, C]
        gt_one_hot = F.one_hot(gt_labels_squeezed, num_classes=C).float()
        # pred_scores: [B, L, C] -> [B, 1, L, C]
        # gt_one_hot : [B, n, C] -> [B, n, 1, C]
        # broadcast → [B, n, L, C], then sum over C
        bbox_cls_scores = (pred_scores.unsqueeze(1) * gt_one_hot.unsqueeze(2)).sum(-1)  # [B, n, L]


        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(alignment_metrics * is_in_gts, self.topk, topk_mask=pad_gt_mask)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).tile([1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(dim=-2)
        assigned_gt_index = mask_positive.argmax(dim=-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = torch.gather(gt_labels.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(mask_positive_sum > 0, assigned_labels, torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gt_bboxes.reshape([-1, 4])[assigned_gt_index.flatten(), :]
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_poses = gt_poses.reshape([-1, num_keypoints, 3])[assigned_gt_index.flatten(), :]
        assigned_poses = assigned_poses.reshape([batch_size, num_anchors, num_keypoints, 3])

        assigned_scores = torch.nn.functional.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, index=torch.tensor(ind, device=assigned_scores.device, dtype=torch.long), dim=-1)
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance = alignment_metrics.max(dim=-1, keepdim=True).values
        max_ious_per_instance = (ious * mask_positive).max(dim=-1, keepdim=True).values
        alignment_metrics = alignment_metrics / (max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics = alignment_metrics.max(dim=-2).values.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        # respect crowd
        assigned_crowd = torch.gather(gt_crowd.flatten(), index=assigned_gt_index.flatten(), dim=0)
        assigned_crowd = assigned_crowd.reshape([batch_size, num_anchors])
        assigned_scores = assigned_scores * assigned_crowd.eq(0).unsqueeze(-1)

        return YoloNASPoseBoxesAssignmentResult(
            assigned_labels=assigned_labels,
            assigned_bboxes=assigned_bboxes,
            assigned_scores=assigned_scores,
            assigned_poses=assigned_poses,
            assigned_gt_index=assigned_gt_index,
            assigned_crowd=assigned_crowd,
            assigned_rotation_eligible=(rotation_eligible.reshape(-1)[assigned_gt_index].bool() if rotation_eligible is not None else None),
        )


class CIoULoss(nn.Module):
    """
    Complete IoU loss
    """

    def __init__(self, eps: float = 1e-10, reduction: str = "none"):
        """
        :param eps:         epsilon to avoid divide by zero, default as 1e-10
        :param reduction:   Options are "none", "mean" and "sum". default as none
        """

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"reduction must be one of 'none', 'mean', 'sum', but got {reduction}")
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, predictions: Tensor, targets: Tensor, loc_weights: Optional[Tensor] = None) -> Tensor:
        """
        :param predictions: Predicted boxes in xyxy format of [D0, D1,...Di, 4] shape
        :param targets:     Target boxes in xyxy format of [D0, D1,...Di, 4] shape
        :param loc_weights: Optional tensor of [D0, D1,...Di] shape with weights for each prediction
        :return:            CIOU loss
        """
        loss = bbox_ciou_loss(predictions, targets, eps=self.eps)
        if loc_weights is not None:
            loss = loss * loc_weights
        if self.reduction == "sum":
            loss = torch.sum(loss)
        elif self.reduction == "mean":
            loss = torch.mean(loss)
        return loss


@register_loss(Losses.CHESS_YOLONAS_POSE_LOSS)
class ChessYoloNASPoseLoss(nn.Module):
    """
    Loss for training YoloNASPose model
    """

    def __init__(
        self,
        oks_sigmas: Optional[Union[List[float], np.ndarray, Tensor]] = None,
        classification_loss_type: str = "focal",
        regression_iou_loss_type: str = "ciou",
        classification_loss_weight: float = 1.0,
        iou_loss_weight: float = 2.5,
        dfl_loss_weight: float = 0.5,
        pose_cls_loss_weight: float = 1.0,
        pose_reg_loss_weight: float = 1.0,
        pose_classification_loss_type: str = "bce",
        num_classes: int = 1,
        bbox_assigner_topk: int = 13,
        bbox_assigned_alpha: float = 1.0,
        bbox_assigned_beta: float = 4.0,
        assigner_multiply_by_pose_oks: bool = False,
        rescale_pose_loss_with_assigned_score: bool = False,
        average_losses_in_ddp: bool = False,
        board_oks_sigmas: Optional[Union[List[float], np.ndarray, Tensor]] = None,
        piece_oks_sigma: Optional[Union[List[float], np.ndarray, Tensor]] = None,
        board_class_id: int = 12,
        board_localization_loss_multiplier: float = 1.0,
        rotation_invariant_start_epoch: Optional[int] = None,
        ignore_unannotated_board_keypoints: bool = False,
        rotation_invariant_max_pieces: Optional[int] = None,
        log_pose_diagnostics: bool = False,
        factorized_piece_color: bool = False,
        piece_objectness: bool = False,
        piece_color_loss_weight: float = 1.0,
    ):
        """
        :param oks_sigmas:                 OKS sigmas for pose estimation. Array of [Num Keypoints].
        :param classification_loss_type:   Classification loss type. One of "focal" or "bce"
        :param regression_iou_loss_type:   Regression IoU loss type. One of "giou" or "ciou"
        :param classification_loss_weight: Classification loss weight
        :param iou_loss_weight:            IoU loss weight
        :param dfl_loss_weight:            DFL loss weight
        :param pose_cls_loss_weight:       Pose classification loss weight
        :param pose_reg_loss_weight:       Pose regression loss weight
        :param rotation_invariant_start_epoch: Epoch from which either board labelling is accepted; None disables.
                                           See _resolve_board_orientation.
        :param ignore_unannotated_board_keypoints: Ignore missing board landmarks in confidence loss, rather than
                                                   treating absent annotations as negative targets. Enabled by the chess recipe.
        :param rotation_invariant_max_pieces: Inclusive original on-board annotation count; unknown counts stay strict.
                                             None accepts every board once the start epoch is reached.
        :param log_pose_diagnostics: Log weighted board/piece contributions and rotation fractions without changing metrics.
        :param factorized_piece_color: Match the head's factorized mode. Type/board detection keeps
               focal/BCE loss; color uses cross-entropy only on assigned pieces, weighted by the
               same assignment quality and normalized by the same assigned score sum.
        :param piece_color_loss_weight: Color term multiplier inside classification loss (default 1).
        :param average_losses_in_ddp:      Whether to average losses in DDP mode. In theory, enabling this option
                                           should have the positive impact on model accuracy since it would smooth out
                                           influence of batches with small number of objects.
                                           However, it needs to be proven empirically.
        """
        super().__init__()
        if factorized_piece_color and (num_classes != 13 or board_class_id != 12):
            raise ValueError("Factorized chess classification requires num_classes=13 and board_class_id=12")
        if not math.isfinite(piece_color_loss_weight) or piece_color_loss_weight < 0:
            raise ValueError("piece_color_loss_weight must be finite and nonnegative")
        self.factorized_piece_color = factorized_piece_color
        if piece_objectness and not factorized_piece_color:
            raise ValueError("piece_objectness requires factorized_piece_color")
        self.piece_objectness = piece_objectness
        self.piece_color_loss_weight = piece_color_loss_weight
        
        if board_oks_sigmas is None and oks_sigmas is not None:
            board_oks_sigmas = oks_sigmas
            
        self.classification_loss_type = classification_loss_type
        self.classification_loss_weight = classification_loss_weight
        self.dfl_loss_weight = dfl_loss_weight
        self.iou_loss_weight = iou_loss_weight

        self.iou_loss = {"giou": GIoULoss, "ciou": CIoULoss}[regression_iou_loss_type]()
        self.num_keypoints = len(board_oks_sigmas)
        self.num_classes = num_classes
        self.oks_sigmas = torch.tensor(board_oks_sigmas)
        self.piece_oks_sigma = torch.tensor(piece_oks_sigma) if piece_oks_sigma is not None else None
        self.board_class_id = board_class_id
        self.board_localization_loss_multiplier = board_localization_loss_multiplier
        self.pose_cls_loss_weight = pose_cls_loss_weight
        self.pose_reg_loss_weight = pose_reg_loss_weight
        self.assigner = YoloNASPoseTaskAlignedAssigner(
            sigmas=self.oks_sigmas,
            topk=bbox_assigner_topk,
            alpha=bbox_assigned_alpha,
            beta=bbox_assigned_beta,
            multiply_by_pose_oks=assigner_multiply_by_pose_oks,
            piece_sigma=self.piece_oks_sigma,
            board_class_id=self.board_class_id,
        )
        self.pose_classification_loss_type = pose_classification_loss_type
        self.rescale_pose_loss_with_assigned_score = rescale_pose_loss_with_assigned_score
        self.average_losses_in_ddp = average_losses_in_ddp
        self.rotation_invariant_start_epoch = rotation_invariant_start_epoch
        self.ignore_unannotated_board_keypoints = ignore_unannotated_board_keypoints
        # Kept up to date by ChessBoardRotationInvarianceCallback.
        self.current_epoch = 0
        if rotation_invariant_max_pieces is not None and (
            isinstance(rotation_invariant_max_pieces, bool) or not isinstance(rotation_invariant_max_pieces, int) or rotation_invariant_max_pieces < 0
        ):
            raise ValueError("rotation_invariant_max_pieces must be a nonnegative integer or None")
        if rotation_invariant_max_pieces is not None and self.num_keypoints != 9:
            raise ValueError("Conditional board rotation requires nine board keypoints")
        self.rotation_invariant_max_pieces = rotation_invariant_max_pieces
        self.log_pose_diagnostics = log_pose_diagnostics
        self._pose_diagnostics = {}

    @torch.no_grad()
    def _unpack_flat_targets(self, targets: Tuple[Tensor, ...], batch_size: int) -> Mapping[str, torch.Tensor]:
        """
        Convert targets to PPYoloE-compatible format since it's the easiest (not the cleanest) way to
        have PP Yolo training & metrics computed

        :param targets: Tuple (boxes, joints, class_labels[, board_piece_counts])
                        - boxes: [N, 5] (batch_index, x1, y1, x2, y2)
                        - joints: [N, num_joints, 4] (batch_index, x, y, visibility)
                        - class_labels: [N, 2] (batch_index, class_id)
                        - board_piece_counts: optional [N, 2] (batch_index, original count or -1 if unknown)
        :return:        (Dictionary [str,Tensor]) with keys:
                        - gt_class: (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
                        - gt_bbox: (Tensor, float32): Ground truth bboxes, shape(B, n, 4) in XYXY format
                        - pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
        """
        target_boxes, target_joints, target_class_labels = targets[:3]

        image_index = target_boxes[:, 0].long()
        # A single device sync for the padded size, instead of several per image.
        counts = torch.bincount(image_index, minlength=batch_size)
        max_boxes = int(counts.max())
        # Each target's slot within its image, in collated order.
        order = torch.argsort(image_index, stable=True)
        image_index = image_index[order]
        slot = torch.arange(len(order), device=order.device) - (torch.cumsum(counts, dim=0) - counts)[image_index]

        def padded(values: Tensor, dtype: torch.dtype) -> Tensor:
            result = values.new_zeros((batch_size, max_boxes) + values.shape[1:], dtype=dtype)
            result[image_index, slot] = values[order].to(dtype)
            return result

        gt_bbox = padded(target_boxes[:, 1:5], target_boxes.dtype)
        result = {
            "gt_class": padded(target_class_labels[:, 1:], torch.long),
            "gt_bbox": gt_bbox,
            "pad_gt_mask": gt_bbox.sum(dim=-1, keepdim=True) > 0,
            "gt_poses": padded(target_joints[..., 1:], target_joints.dtype),
        }
        if len(targets) > 3:
            result["board_piece_counts"] = padded(targets[3][:, 1:], torch.long)
        return result

    def forward(
        self,
        outputs: Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
        targets: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tensor]:
        """
        :param outputs: Tuple of model decoded predictions and raw tensors.
        :param targets: A tuple of (boxes, joints, class_labels) tensors where
                        - boxes: [N, 5] (batch_index, x1, y1, x2, y2)
                        - joints: [N, num_joints, 4] (batch_index, x, y, visibility)
                        - class_labels: [N, 2] (batch_index, class_id)
        :return:        Tuple of two tensors where first element is main loss for backward and
                        second element is stacked tensor of all individual losses
        """
        _, predictions = outputs
        expected_tensors = 10 if self.piece_objectness else (9 if self.factorized_piece_color else 8)
        if len(predictions) != expected_tensors:
            raise ValueError("Head and loss must use the same factorized_piece_color and piece_objectness settings")
        color_logits = predictions[8] if self.factorized_piece_color else None
        objectness_logits = predictions[9] if self.piece_objectness else None

        (
            pred_scores,
            pred_distri,
            pred_pose_coords,  # [B, Anchors, C, 2]
            pred_pose_logits,  # [B, Anchors, C]
            anchors,
            anchor_points,
            num_anchors_list,
            stride_tensor,
        ) = predictions[:8]
        expected_channels = 7 if self.factorized_piece_color else self.num_classes
        if pred_scores.shape[-1] != expected_channels:
            raise ValueError("Unexpected classification channels; check factorized_piece_color on head and loss")

        _, target_joints, target_class_labels = targets[:3]
        targets = self._unpack_flat_targets(targets, batch_size=pred_scores.size(0))

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes, reg_max = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = targets["gt_class"]
        gt_bboxes = targets["gt_bbox"]
        gt_poses = targets["gt_poses"]
        pad_gt_mask = targets["pad_gt_mask"]
        gt_crowd = torch.zeros_like(pad_gt_mask, dtype=torch.bool)

        rotation_eligible = self._rotation_eligibility(targets)
        board_gt = (gt_labels == self.board_class_id) & pad_gt_mask.bool()
        self._pose_diagnostics = {
            "rotation_eligible_fraction": rotation_eligible.float().sum() / board_gt.sum().clamp(min=1),
        }

        # label assignment
        assign_result = self.assigner(
            pred_scores=(chess_joint_scores(pred_scores.detach(), color_logits.detach(),
                                            objectness_logits.detach() if objectness_logits is not None else None)
                         if self.factorized_piece_color else pred_scores.detach().sigmoid()),
            pred_bboxes=pred_bboxes.detach() * stride_tensor,
            pred_pose_coords=pred_pose_coords.detach(),
            anchor_points=anchor_points,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            gt_poses=gt_poses,
            gt_crowd=gt_crowd,
            pad_gt_mask=pad_gt_mask,
            bg_index=self.num_classes,
            rotation_eligible=rotation_eligible,
        )

        assigned_scores = assign_result.assigned_scores

        assigned_scores_sum = assigned_scores.sum()
        if self.average_losses_in_ddp and is_distributed():
            torch.distributed.all_reduce(assigned_scores_sum, op=torch.distributed.ReduceOp.SUM)
            assigned_scores_sum /= get_world_size()
        assigned_scores_sum = torch.clip(assigned_scores_sum, min=1.0)
        loss_cls = self._classification_loss(pred_scores, assigned_scores, assigned_scores_sum, color_logits,
                                             objectness_logits, assign_result.assigned_labels, assign_result.assigned_crowd)

        loss_iou, loss_dfl, loss_pose_cls, loss_pose_reg = self._bbox_loss(
            pred_distri,
            pred_bboxes,
            pred_pose_coords=pred_pose_coords,
            pred_pose_logits=pred_pose_logits,
            stride_tensor=stride_tensor,
            anchor_points=anchor_points_s,
            assign_result=assign_result,
            assigned_scores_sum=assigned_scores_sum,
            reg_max=reg_max,
        )

        loss_cls = loss_cls * self.classification_loss_weight
        loss_iou = loss_iou * self.iou_loss_weight
        loss_dfl = loss_dfl * self.dfl_loss_weight
        loss_pose_cls = loss_pose_cls * self.pose_cls_loss_weight
        loss_pose_reg = loss_pose_reg * self.pose_reg_loss_weight

        loss = loss_cls + loss_iou + loss_dfl + loss_pose_cls + loss_pose_reg
        log_losses = torch.stack(
            [
                loss_cls.detach(),
                loss_iou.detach(),
                loss_dfl.detach(),
                loss_pose_cls.detach(),
                loss_pose_reg.detach(),
                loss.detach(),
            ]
        )

        if self.log_pose_diagnostics:
            extra = [self._pose_diagnostics.get(name, loss.detach().new_zeros(())) for name in self.component_names[6:]]
            log_losses = torch.cat([log_losses, torch.stack(extra).detach()])
        return loss, log_losses

    @property
    def component_names(self):
        names = ["loss_cls", "loss_iou", "loss_dfl", "loss_pose_cls", "loss_pose_reg", "loss"]
        if self.log_pose_diagnostics:
            names += [f"{group}_loss_{term}" for group in ("board", "piece") for term in ("cls", "iou", "dfl", "pose_cls", "pose_reg")]
            names += ["rotation_eligible_fraction", "rotation_flipped_assigned_fraction"]
            if self.factorized_piece_color:
                names += ["piece_loss_type", "piece_loss_color"]
                if self.piece_objectness:
                    names += ["piece_loss_objectness"]
        return names

    def _classification_loss(
        self, pred_scores: Tensor, assigned_scores: Tensor, assigned_scores_sum: Tensor, color_logits: Optional[Tensor] = None,
        objectness_logits: Optional[Tensor] = None, assigned_labels: Optional[Tensor] = None, assigned_crowd: Optional[Tensor] = None
    ) -> Tensor:
        """Factor labels after assignment; retain all 13 joint labels for geometry and matching.

        White/black versions of a type share one detection target. Background and
        board anchors have zero color target, hence exactly zero color gradient.
        Soft assignment scores retain their quality weighting in both terms.
        """
        if self.piece_objectness:
            return self._piece_classification_loss(pred_scores, color_logits, objectness_logits, assigned_scores,
                                                   assigned_scores_sum, assigned_labels, assigned_crowd)
        color_loss = pred_scores.new_zeros(())
        detection_targets = assigned_scores
        if self.factorized_piece_color:
            detection_targets = torch.cat(
                (assigned_scores[..., :6] + assigned_scores[..., 6:12], assigned_scores[..., 12:13]), dim=-1
            )
            color_targets = torch.stack(
                (assigned_scores[..., :6].sum(dim=-1), assigned_scores[..., 6:12].sum(dim=-1)), dim=-1
            )
            # Float32 log-softmax is stable under mixed precision, including confident mistakes.
            color_loss = -(color_targets * F.log_softmax(color_logits.float(), dim=-1)).sum()
            color_loss = color_loss / assigned_scores_sum * self.piece_color_loss_weight

        if self.classification_loss_type == "focal":
            cls_unreduced = self._focal_loss(pred_scores, detection_targets, alpha=-1, reduction="none")
        elif self.classification_loss_type == "bce":
            cls_unreduced = F.binary_cross_entropy_with_logits(pred_scores, detection_targets, reduction="none")
        else:
            raise ValueError(f"Unknown classification loss type: {self.classification_loss_type}")

        if self.log_pose_diagnostics:
            by_class = cls_unreduced.sum(dim=(0, 1)) / assigned_scores_sum * self.classification_loss_weight
            board_index = 6 if self.factorized_piece_color else self.board_class_id
            weighted_color = color_loss * self.classification_loss_weight
            self._pose_diagnostics["board_loss_cls"] = by_class[board_index]
            self._pose_diagnostics["piece_loss_cls"] = by_class.sum() - by_class[board_index] + weighted_color
            if self.factorized_piece_color:
                self._pose_diagnostics["piece_loss_type"] = by_class[:6].sum()
                self._pose_diagnostics["piece_loss_color"] = weighted_color
        return cls_unreduced.sum() / assigned_scores_sum + color_loss

    def _piece_classification_loss(self, logits, colors, objectness, scores, score_sum, labels, crowd):
        # Use assignment labels, not soft quality: a poorly localized assigned piece
        # is still a positive object. Board/background never supervise type or color.
        positive = (labels < self.board_class_id) & ~crowd.bool()
        count = positive.sum().float()
        if self.average_losses_in_ddp and is_distributed():
            torch.distributed.all_reduce(count)
            count /= get_world_size()
        count = count.clamp(min=1)
        # Honor the same focal/BCE choice as board detection. Keep this calculation
        # in float32 so easy-background focal weights do not underflow prematurely.
        if self.classification_loss_type == "focal":
            obj = self._focal_loss(objectness[..., 0].float(), positive.float(), alpha=-1, reduction="none")
        elif self.classification_loss_type == "bce":
            obj = F.binary_cross_entropy_with_logits(objectness[..., 0].float(), positive.float(), reduction="none")
        else:
            raise ValueError(f"Unknown classification loss type: {self.classification_loss_type}")
        obj = (obj * ~crowd.bool()).sum() / count
        type_targets = F.one_hot(labels.clamp(max=11).remainder(6).long(), 6).to(logits.dtype)
        color_targets = F.one_hot(labels.clamp(max=11).div(6, rounding_mode="floor").long(), 2).to(logits.dtype)
        typ = -(F.log_softmax(logits[..., :6].float(), -1) * type_targets).sum(-1)
        col = -(F.log_softmax(colors.float(), -1) * color_targets).sum(-1)
        typ = (typ * positive).sum() / count
        col = (col * positive).sum() / count * self.piece_color_loss_weight
        # Keep the board's original targets, loss, normalization and weight.
        if self.classification_loss_type == "focal":
            board = self._focal_loss(logits[..., 6:7], scores[..., 12:13], alpha=-1, reduction="sum")
        elif self.classification_loss_type == "bce":
            board = F.binary_cross_entropy_with_logits(logits[..., 6:7], scores[..., 12:13], reduction="sum")
        else:
            raise ValueError(f"Unknown classification loss type: {self.classification_loss_type}")
        board = board / score_sum
        if self.log_pose_diagnostics:
            for name, value in {"board_loss_cls": board, "piece_loss_cls": obj + typ + col,
                                "piece_loss_objectness": obj, "piece_loss_type": typ, "piece_loss_color": col}.items():
                self._pose_diagnostics[name] = value * self.classification_loss_weight
        return board + obj + typ + col

    def _rotation_eligibility(self, targets):
        boards = (targets["gt_class"] == self.board_class_id) & targets["pad_gt_mask"].bool()
        if self.rotation_invariant_start_epoch is None or self.current_epoch < self.rotation_invariant_start_epoch:
            return torch.zeros_like(boards)
        if self.rotation_invariant_max_pieces is None:
            return boards
        counts = targets.get("board_piece_counts")
        if counts is None:
            raise ValueError("Conditional rotation requires original board piece counts from ChessYoloNASPoseCollateFN")
        return boards & (counts >= 0) & (counts <= self.rotation_invariant_max_pieces)

    def _df_loss(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        target_left = target.long()
        target_right = target_left + 1
        weight_left = target_right.float() - target
        weight_right = 1 - weight_left

        # [B,L,C] -> [B,C,L] to make compatible with torch.nn.functional.cross_entropy
        # which expects channel dim to be at index 1
        pred_dist = torch.moveaxis(pred_dist, -1, 1)

        loss_left = torch.nn.functional.cross_entropy(pred_dist, target_left, reduction="none") * weight_left
        loss_right = torch.nn.functional.cross_entropy(pred_dist, target_right, reduction="none") * weight_right
        return (loss_left + loss_right).mean(dim=-1, keepdim=True)

    def _resolve_board_orientation(self, predicted_coords: Tensor, target_coords: Tensor, target_visibility: Tensor) -> Tuple[Tensor, Tensor]:
        """Score each board against whichever of the two valid corner labellings it is closer to.

        A board with few or no pieces is symmetric under a 180 degree rotation, so nothing in the
        pixels says which end is a1. Scoring against one fixed labelling punishes a coin flip the
        model cannot win, and it responds by hedging: the predicted corners drift in towards the
        center, which costs us the homography. Accepting either labelling lets it commit to sharp
        corners, and the app resolves the flip downstream.

        The choice is made once per board and applied to all nine keypoints, so a single board
        cannot be scored half under one labelling and half under the other.

        :param predicted_coords:  [Num Instances, Num Joints, 2] - (x, y)
        :param target_coords:     [Num Instances, Num Joints, 2] - (x, y)
        :param target_visibility: [Num Instances, Num Joints, 1] - Visibility of each joint
        :return:                  (target_coords, target_visibility) under the closer labelling
        """
        if target_coords.shape[1] != len(BOARD_180_PERMUTATION):
            return target_coords, target_visibility

        permutation = torch.as_tensor(BOARD_180_PERMUTATION, device=target_coords.device)
        use_rotated = board_rotation_choice(predicted_coords, target_coords, target_visibility)[..., None, None]
        return (
            torch.where(use_rotated, target_coords[:, permutation], target_coords),
            torch.where(use_rotated, target_visibility[:, permutation], target_visibility),
        )

    def _keypoint_loss(
        self,
        predicted_coords: Tensor,
        target_coords: Tensor,
        predicted_logits: Tensor,
        target_visibility: Tensor,
        area: Tensor,
        sigmas: Tensor,
        assigned_scores: Optional[Tensor] = None,
        assigned_scores_sum: Optional[Tensor] = None,
        ignore_unannotated_keypoints: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """

        :param predicted_coords:  [Num Instances, Num Joints, 2] - (x, y)
        :param target_coords:     [Num Instances, Num Joints, 2] - (x, y)
        :param predicted_logits:  [Num Instances, Num Joints, 1] - Logits for each joint
        :param target_visibility: [Num Instances, Num Joints, 1] - Visibility of each joint
        :param sigmas:            [Num Joints] - Sigma for each joint
        :param area:              [Num Instances, 1] - Area of the corresponding bounding box
        :param ignore_unannotated_keypoints: If True, average confidence loss over annotated joints only.
                                           Missing joints contribute zero to both pose losses.
        :return:                  Tuple of (regression loss, classification loss)
                                  - regression loss [Num Instances, 1]
                                  - classification loss [Num Instances, 1]
        """
        sigmas = sigmas.reshape([1, -1, 1])
        area = area.reshape([-1, 1, 1])

        visible_targets_mask: Tensor = (target_visibility > 0).float()  # [Num Instances, Num Joints, 1]

        d = ((predicted_coords - target_coords) ** 2).sum(dim=-1, keepdim=True)  # [[Num Instances, Num Joints, 1]
        e = d / (2 * sigmas) ** 2 / (area + 1e-9) / 2  # [Num Instances, Num Joints, 1]
        regression_loss_unreduced = 1 - torch.exp(-e)  # [Num Instances, Num Joints, 1]

        regression_loss_reduced = (regression_loss_unreduced * visible_targets_mask).sum(dim=1, keepdim=False) / (
            visible_targets_mask.sum(dim=1, keepdim=False) + 1e-9
        )  # [Num Instances, 1]

        if self.pose_classification_loss_type == "bce":
            classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted_logits, visible_targets_mask, reduction="none")
        elif self.pose_classification_loss_type == "focal":
            classification_loss = self._focal_loss(predicted_logits, visible_targets_mask, alpha=0.25, gamma=2.0, reduction="none")
        else:
            raise ValueError(f"Unsupported pose classification loss type {self.pose_classification_loss_type}")

        if ignore_unannotated_keypoints:
            # Board v=0 means unknown, not a negative visibility example. Match
            # regression's per-instance mean over known landmarks. Empty masks
            # produce exactly zero loss and zero confidence gradient.
            classification_loss = (classification_loss * visible_targets_mask).sum(dim=1) / visible_targets_mask.sum(dim=1).clamp(min=1)
        else:
            classification_loss = classification_loss.mean(dim=1)

        if assigned_scores is None:
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss_reduced.mean()
        else:
            classification_loss = (classification_loss * assigned_scores).sum() / assigned_scores_sum
            regression_loss = (regression_loss_reduced * assigned_scores).sum() / assigned_scores_sum

        return regression_loss, classification_loss

    def _xyxy_box_area(self, boxes):
        """
        :param boxes: [..., 4] (x1, y1, x2, y2)
        :return: [...,1]
        """
        area = (boxes[..., 2:4] - boxes[..., 0:2]).prod(dim=-1, keepdim=True)
        return area

    def _bbox_loss(
        self,
        pred_dist,
        pred_bboxes,
        pred_pose_coords,
        pred_pose_logits,
        stride_tensor,
        anchor_points,
        assign_result: YoloNASPoseBoxesAssignmentResult,
        assigned_scores_sum,
        reg_max: int,
    ):
        # select positive samples mask that are not crowd and not background
        # loss ALWAYS respect the crowd targets by excluding them from contributing to the loss
        # if you want to train WITH crowd targets, mark them as non-crowd on dataset level
        mask_positive = (assign_result.assigned_labels != self.num_classes) * assign_result.assigned_crowd.eq(0)
        num_pos = mask_positive.sum()
        assigned_bboxes_divided_by_stride = assign_result.assigned_bboxes / stride_tensor

        # pos/neg loss
        if num_pos > 0:
            assigned_labels_pos = assign_result.assigned_labels[mask_positive]
            localization_multiplier = torch.ones_like(assigned_labels_pos, dtype=pred_bboxes.dtype).unsqueeze(-1)
            if self.board_localization_loss_multiplier != 1.0:
                localization_multiplier[assigned_labels_pos == self.board_class_id] = self.board_localization_loss_multiplier

            bbox_mask = mask_positive.unsqueeze(-1).tile([1, 1, 4])
            bbox_weight = torch.masked_select(assign_result.assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            # CIoU loss — skip compute when weight is zero
            if self.iou_loss_weight > 0:
                pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
                assigned_bboxes_pos = torch.masked_select(assigned_bboxes_divided_by_stride, bbox_mask).reshape([-1, 4])
                iou_loss_raw = self.iou_loss(pred_bboxes_pos, assigned_bboxes_pos)
                if iou_loss_raw.dim() == 1 and bbox_weight.dim() == 2:
                    iou_loss_raw = iou_loss_raw.unsqueeze(-1)
                iou_terms = iou_loss_raw * bbox_weight * localization_multiplier / assigned_scores_sum
                loss_iou = iou_terms.sum()
                if self.log_pose_diagnostics:
                    board = assigned_labels_pos == self.board_class_id
                    self._pose_diagnostics["board_loss_iou"] = iou_terms[board].sum() * self.iou_loss_weight
                    self._pose_diagnostics["piece_loss_iou"] = iou_terms[~board].sum() * self.iou_loss_weight
            else:
                loss_iou = torch.zeros([], device=pred_bboxes.device)

            # DFL loss — skip compute when weight is zero
            if self.dfl_loss_weight > 0:
                dist_mask = mask_positive.unsqueeze(-1).tile([1, 1, (reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, reg_max + 1])
                assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes_divided_by_stride, reg_max)
                assigned_ltrb_pos = torch.masked_select(assigned_ltrb, bbox_mask).reshape([-1, 4])
                dfl_loss_raw = self._df_loss(pred_dist_pos, assigned_ltrb_pos)
                if dfl_loss_raw.dim() == 1 and bbox_weight.dim() == 2:
                    dfl_loss_raw = dfl_loss_raw.unsqueeze(-1)
                dfl_terms = dfl_loss_raw * bbox_weight * localization_multiplier / assigned_scores_sum
                loss_dfl = dfl_terms.sum()
                if self.log_pose_diagnostics:
                    board = assigned_labels_pos == self.board_class_id
                    self._pose_diagnostics["board_loss_dfl"] = dfl_terms[board].sum() * self.dfl_loss_weight
                    self._pose_diagnostics["piece_loss_dfl"] = dfl_terms[~board].sum() * self.dfl_loss_weight
            else:
                loss_dfl = torch.zeros([], device=pred_bboxes.device)

            # Pose losses (keypoint regression + classification)
            # Do not divide poses by stride since this would skew the loss and make sigmas incorrect
            pred_pose_coords = pred_pose_coords[mask_positive]
            pred_pose_logits = pred_pose_logits[mask_positive].unsqueeze(-1)  # To make [Num Instances, Num Joints, 1]

            gt_pose_coords = assign_result.assigned_poses[..., 0:2][mask_positive]
            gt_pose_visibility = assign_result.assigned_poses[mask_positive][:, :, 2:3]

            assigned_bboxes_pos_image_coord = torch.masked_select(assign_result.assigned_bboxes, bbox_mask).reshape([-1, 4])
            area = self._xyxy_box_area(assigned_bboxes_pos_image_coord).reshape([-1, 1]) * 0.53
            assigned_labels_pos = assign_result.assigned_labels[mask_positive]

            if self.piece_oks_sigma is not None:
                mask_board = (assigned_labels_pos == self.board_class_id)
                mask_pieces = ~mask_board
                
                loss_pose_reg = torch.zeros([], device=pred_bboxes.device)
                loss_pose_cls = torch.zeros([], device=pred_bboxes.device)
                
                reg_b, cls_b, reg_p, cls_p = 0.0, 0.0, 0.0, 0.0

                if mask_board.any():
                    board_target_coords = gt_pose_coords[mask_board]
                    board_target_visibility = gt_pose_visibility[mask_board]
                    eligible = assign_result.assigned_rotation_eligible
                    if eligible is None:
                        active = self.rotation_invariant_start_epoch is not None and self.current_epoch >= self.rotation_invariant_start_epoch
                        board_eligible = torch.full_like(mask_board[mask_board], active and self.rotation_invariant_max_pieces is None)
                    else:
                        board_eligible = eligible[mask_positive][mask_board]
                    flipped = torch.zeros_like(board_eligible)
                    if board_eligible.any():
                        flipped = board_eligible & board_rotation_choice(pred_pose_coords[mask_board], board_target_coords, board_target_visibility)
                        perm = torch.as_tensor(BOARD_180_PERMUTATION, device=board_target_coords.device)
                        board_target_coords = torch.where(flipped[:, None, None], board_target_coords[:, perm], board_target_coords)
                        board_target_visibility = torch.where(flipped[:, None, None], board_target_visibility[:, perm], board_target_visibility)
                    self._pose_diagnostics["rotation_flipped_assigned_fraction"] = flipped.float().sum() / board_eligible.sum().clamp(min=1)

                    reg_b, cls_b = self._keypoint_loss(
                        predicted_coords=pred_pose_coords[mask_board],
                        target_coords=board_target_coords,
                        predicted_logits=pred_pose_logits[mask_board],
                        target_visibility=board_target_visibility,
                        ignore_unannotated_keypoints=self.ignore_unannotated_board_keypoints,
                        assigned_scores=bbox_weight[mask_board] if self.rescale_pose_loss_with_assigned_score else None,
                        assigned_scores_sum=assigned_scores_sum if self.rescale_pose_loss_with_assigned_score else None,
                        area=area[mask_board],
                        sigmas=self.oks_sigmas.to(pred_pose_logits.device),
                    )
                    loss_pose_reg += reg_b * self.board_localization_loss_multiplier
                    loss_pose_cls += cls_b

                if mask_pieces.any():
                    reg_p, cls_p = self._keypoint_loss(
                        predicted_coords=pred_pose_coords[mask_pieces][:, 0:1, :],
                        target_coords=gt_pose_coords[mask_pieces][:, 0:1, :],
                        predicted_logits=pred_pose_logits[mask_pieces][:, 0:1, :],
                        target_visibility=gt_pose_visibility[mask_pieces][:, 0:1, :],
                        assigned_scores=bbox_weight[mask_pieces] if self.rescale_pose_loss_with_assigned_score else None,
                        assigned_scores_sum=assigned_scores_sum if self.rescale_pose_loss_with_assigned_score else None,
                        area=area[mask_pieces],
                        sigmas=self.piece_oks_sigma.to(pred_pose_logits.device),
                    )
                    loss_pose_reg += reg_p
                    loss_pose_cls += cls_p
                    
                if not self.rescale_pose_loss_with_assigned_score:
                    total_instances = len(assigned_labels_pos)
                    loss_pose_reg = (reg_b * mask_board.sum() * self.board_localization_loss_multiplier + reg_p * mask_pieces.sum()) / total_instances
                    loss_pose_cls = (cls_b * mask_board.sum() + cls_p * mask_pieces.sum()) / total_instances

                if self.log_pose_diagnostics:
                    for group, reg, cls, mask, multiplier in (
                        ("board", reg_b, cls_b, mask_board, self.board_localization_loss_multiplier),
                        ("piece", reg_p, cls_p, mask_pieces, 1.0),
                    ):
                        fraction = 1.0 if self.rescale_pose_loss_with_assigned_score else mask.sum() / len(assigned_labels_pos)
                        self._pose_diagnostics[group + "_loss_pose_reg"] = torch.as_tensor(reg, device=pred_bboxes.device) * fraction * multiplier * self.pose_reg_loss_weight
                        self._pose_diagnostics[group + "_loss_pose_cls"] = torch.as_tensor(cls, device=pred_bboxes.device) * fraction * self.pose_cls_loss_weight

            else:
                loss_pose_reg, loss_pose_cls = self._keypoint_loss(
                    predicted_coords=pred_pose_coords,
                    target_coords=gt_pose_coords,
                    predicted_logits=pred_pose_logits,
                    target_visibility=gt_pose_visibility,
                    assigned_scores=bbox_weight if self.rescale_pose_loss_with_assigned_score else None,
                    assigned_scores_sum=assigned_scores_sum if self.rescale_pose_loss_with_assigned_score else None,
                    area=area,
                    sigmas=self.oks_sigmas.to(pred_pose_logits.device),
                )
        else:
            loss_iou = torch.zeros([], device=pred_bboxes.device)
            loss_dfl = torch.zeros([], device=pred_bboxes.device)
            loss_pose_cls = torch.zeros([], device=pred_bboxes.device)
            loss_pose_reg = torch.zeros([], device=pred_bboxes.device)

        return loss_iou, loss_dfl, loss_pose_cls, loss_pose_reg

    def _bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor) -> Tuple[Tensor, int]:
        """
        Decode predicted bounding boxes using anchor points and predicted distribution
        :param anchor_points: Anchor locations (center for each point) of [B, L, 2] shape
        :param pred_dist:     Predicted offset distributions of [B, L, 4 * (reg_max + 1)] shape
        :return:              Decoded bounding boxes (XYXY format) of [B, L, 4] shape and reg_max
        """
        b, l, *_ = pred_dist.size()
        pred_dist = torch.softmax(pred_dist.reshape([b, l, 4, -1]), dim=-1)

        reg_max = pred_dist.size(-1) - 1
        proj_conv = torch.linspace(0, reg_max, reg_max + 1, device=pred_dist.device).reshape([1, reg_max + 1, 1, 1])

        pred_dist = torch.nn.functional.conv2d(pred_dist.permute(0, 3, 1, 2), proj_conv).squeeze(1)
        return batch_distance2bbox(anchor_points, pred_dist), reg_max

    def _bbox2distance(self, points, bbox, reg_max):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], dim=-1).clip(0, reg_max - 0.01)

    @staticmethod
    def _focal_loss(pred_logits: Tensor, label: Tensor, alpha=0.25, gamma=2.0, reduction="sum") -> Tensor:
        pred_score = pred_logits.sigmoid()
        weight = torch.abs(pred_score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t
        # This is same, but binary_cross_entropy_with_logits is faster
        # loss = -weight * (label * torch.nn.functional.logsigmoid(pred_logits) + (1 - label) * torch.nn.functional.logsigmoid(-pred_logits))
        loss = weight * torch.nn.functional.binary_cross_entropy_with_logits(pred_logits, label, reduction="none")

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Unsupported reduction type {reduction}")
        return loss
