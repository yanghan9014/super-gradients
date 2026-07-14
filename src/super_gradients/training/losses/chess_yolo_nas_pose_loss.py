"""Chess-specific pose assignment and losses without bounding-box supervision."""

import dataclasses
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss
from super_gradients.training.datasets.pose_estimation_datasets.yolo_nas_pose_collate_fn import (
    undo_flat_collate_tensors_with_batch_index,
)


# Canonical coordinates for [a1, a8, h1, h8, center, a45, h45, 1de, 8de].
CANONICAL_BOARD_KEYPOINTS = (
    (0.0, 8.0),
    (0.0, 0.0),
    (8.0, 8.0),
    (8.0, 0.0),
    (4.0, 4.0),
    (0.0, 4.0),
    (8.0, 4.0),
    (4.0, 8.0),
    (4.0, 0.0),
)

BOARD_AREA_UNAVAILABLE = 0
BOARD_AREA_HOMOGRAPHY = 1
BOARD_AREA_AABB = 2


@dataclasses.dataclass
class BoardGeometry:
    """Per-image board geometry used by assignment and area-normalized losses."""

    areas: Tensor  # [B]
    quadrilaterals: Tensor  # [B, 4, 2], ordered a8 -> h8 -> h1 -> a1
    aabbs: Tensor  # [B, 4], xyxy around visible annotated points
    area_valid: Tensor  # [B]
    homography_valid: Tensor  # [B]
    area_source: Tensor  # [B], one of BOARD_AREA_*


@dataclasses.dataclass
class YoloNASPoseAssignmentResult:
    """Pose-based dense-anchor assignment result."""

    assigned_labels: Tensor  # [B, A], background is num_classes
    assigned_poses: Tensor  # [B, A, J, 3]
    assigned_gt_index: Tensor  # [B, A]
    assigned_quality: Tensor  # [B, A], detached g for the selected GT
    positive_mask: Tensor  # [B, A]


def _shoelace_area(points: Tensor) -> Tensor:
    x = points[..., 0]
    y = points[..., 1]
    return 0.5 * torch.abs((x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1)).sum(dim=-1))


def _fit_projective_transform(
    canonical_points: Tensor,
    image_points: Tensor,
    eps: float,
    max_condition_number: float,
) -> Optional[Tensor]:
    """Fit canonical-to-image homography with h[2,2] fixed to one."""

    if canonical_points.shape[0] < 4:
        return None

    canonical_points = canonical_points.float()
    image_points = image_points.float()
    x = canonical_points[:, 0]
    y = canonical_points[:, 1]
    u = image_points[:, 0]
    v = image_points[:, 1]
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    row_u = torch.stack((x, y, ones, zeros, zeros, zeros, -u * x, -u * y), dim=-1)
    row_v = torch.stack((zeros, zeros, zeros, x, y, ones, -v * x, -v * y), dim=-1)
    matrix = torch.stack((row_u, row_v), dim=1).reshape(-1, 8)
    target = torch.stack((u, v), dim=1).reshape(-1, 1)

    singular_values = torch.linalg.svdvals(matrix)
    if singular_values.numel() < 8 or not torch.isfinite(singular_values).all():
        return None
    smallest = singular_values[-1]
    condition_number = singular_values[0] / smallest.clamp_min(eps)
    if smallest <= eps or condition_number > max_condition_number:
        return None

    solution = torch.linalg.pinv(matrix) @ target
    homography = torch.cat((solution.flatten(), solution.new_ones(1))).reshape(3, 3)
    if not torch.isfinite(homography).all():
        return None
    return homography


def _project_points(homography: Tensor, points: Tensor, eps: float) -> Optional[Tensor]:
    homogeneous = torch.cat((points.float(), points.new_ones((points.shape[0], 1), dtype=torch.float32)), dim=-1)
    projected = homogeneous @ homography.transpose(0, 1)
    denominator = projected[:, 2:3]
    if (denominator.abs() <= eps).any():
        return None
    projected = projected[:, :2] / denominator
    if not torch.isfinite(projected).all():
        return None
    return projected


@torch.no_grad()
def estimate_board_geometry(
    gt_labels: Tensor,
    gt_poses: Tensor,
    pad_gt_mask: Tensor,
    board_class_id: int = 12,
    min_board_area: float = 16.0,
    max_condition_number: float = 1e8,
    eps: float = 1e-6,
) -> BoardGeometry:
    """Estimate full board area, falling back to the visible-point AABB.

    A projective transform is fitted from visible board landmarks to their known
    canonical coordinates. If that fit is unavailable or unstable, the fallback
    is the tight axis-aligned box containing the visible board landmarks. When
    those are unavailable too, all visible annotated points in the image are
    used. Degenerate AABBs remain invalid so area-normalized supervision can be
    skipped for that sample.
    """

    device = gt_poses.device
    batch_size = gt_poses.shape[0]
    canonical = torch.tensor(CANONICAL_BOARD_KEYPOINTS, device=device, dtype=torch.float32)
    canonical_corners = canonical[[1, 3, 2, 0]]  # a8, h8, h1, a1

    areas = torch.zeros(batch_size, device=device, dtype=torch.float32)
    quadrilaterals = torch.zeros(batch_size, 4, 2, device=device, dtype=torch.float32)
    aabbs = torch.zeros(batch_size, 4, device=device, dtype=torch.float32)
    area_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    homography_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    area_source = torch.zeros(batch_size, device=device, dtype=torch.long)

    labels = gt_labels.squeeze(-1).long()
    valid_gt = pad_gt_mask.squeeze(-1).bool()

    for batch_index in range(batch_size):
        board_indices = torch.nonzero(valid_gt[batch_index] & labels[batch_index].eq(board_class_id), as_tuple=False).flatten()
        fallback_points = gt_poses.new_zeros((0, 2), dtype=torch.float32)

        if board_indices.numel() > 0:
            visible_counts = (gt_poses[batch_index, board_indices, :, 2] > 0).sum(dim=-1)
            board_index = board_indices[visible_counts.argmax()]
            board_pose = gt_poses[batch_index, board_index]
            visible = board_pose[:, 2] > 0
            fallback_points = board_pose[visible, :2].float()

            if visible.sum() >= 4:
                homography = _fit_projective_transform(
                    canonical[visible],
                    board_pose[visible, :2],
                    eps=eps,
                    max_condition_number=max_condition_number,
                )
                if homography is not None:
                    projected_corners = _project_points(homography, canonical_corners, eps=eps)
                    if projected_corners is not None:
                        full_area = _shoelace_area(projected_corners)
                        if torch.isfinite(full_area) and full_area >= min_board_area:
                            quadrilaterals[batch_index] = projected_corners
                            areas[batch_index] = full_area
                            area_valid[batch_index] = True
                            homography_valid[batch_index] = True
                            area_source[batch_index] = BOARD_AREA_HOMOGRAPHY
                            minimum = projected_corners.min(dim=0).values
                            maximum = projected_corners.max(dim=0).values
                            aabbs[batch_index] = torch.cat((minimum, maximum))

        if area_valid[batch_index]:
            continue

        if fallback_points.shape[0] == 0:
            visible_all = (gt_poses[batch_index, :, :, 2] > 0) & valid_gt[batch_index, :, None]
            fallback_points = gt_poses[batch_index, :, :, :2][visible_all].float()

        if fallback_points.shape[0] > 0:
            minimum = fallback_points.min(dim=0).values
            maximum = fallback_points.max(dim=0).values
            aabb_area = (maximum - minimum).prod()
            aabbs[batch_index] = torch.cat((minimum, maximum))
            if torch.isfinite(aabb_area) and aabb_area >= min_board_area:
                areas[batch_index] = aabb_area
                area_valid[batch_index] = True
                area_source[batch_index] = BOARD_AREA_AABB

    return BoardGeometry(
        areas=areas,
        quadrilaterals=quadrilaterals,
        aabbs=aabbs,
        area_valid=area_valid,
        homography_valid=homography_valid,
        area_source=area_source,
    )


def _points_inside_convex_quad(points: Tensor, quadrilateral: Tensor, eps: float) -> Tensor:
    edges = torch.roll(quadrilateral, shifts=-1, dims=0) - quadrilateral  # [4, 2]
    relative = points[:, None, :] - quadrilateral[None, :, :]  # [A, 4, 2]
    cross = edges[None, :, 0] * relative[:, :, 1] - edges[None, :, 1] * relative[:, :, 0]  # [A, 4]
    return (cross >= -eps).all(dim=-1) | (cross <= eps).all(dim=-1)  # [A]


def batch_localization_quality(
    gt_labels: Tensor,
    gt_poses: Tensor,
    pred_pose_coords: Tensor,
    board_areas: Tensor,
    pad_gt_mask: Tensor,
    piece_sigma_q: float,
    board_sigma_q: float,
    board_class_id: int = 12,
    eps: float = 1e-9,
) -> Tensor:
    """Compute detached pairwise localization quality g for every GT/anchor."""

    gt_xy = gt_poses[..., :2].float().unsqueeze(2)  # [B, N, 1, J, 2]
    pred_xy = pred_pose_coords.detach().float().unsqueeze(1)  # [B, 1, A, J, 2]
    squared_distance = ((pred_xy - gt_xy) ** 2).sum(dim=-1)  # [B, N, A, J]
    visible = gt_poses[..., 2].gt(0).float().unsqueeze(2)  # [B, N, 1, J]

    area = board_areas.float().clamp_min(eps)[:, None, None]  # [B, 1, 1]
    piece_error = squared_distance[..., 0] / area  # [B, N, A]
    board_error = (squared_distance * visible).sum(dim=-1) / (area * visible.sum(dim=-1).clamp_min(1.0))  # [B, N, A]

    labels = gt_labels.squeeze(-1).long()  # [B, N]
    is_board = labels.eq(board_class_id)  # [B, N]
    normalized_error = torch.where(is_board.unsqueeze(-1), board_error, piece_error)  # [B, N, A]
    sigma = torch.where(
        is_board,
        normalized_error.new_full(is_board.shape, board_sigma_q),
        normalized_error.new_full(is_board.shape, piece_sigma_q),
    )  # [B, N]
    quality = torch.exp(-normalized_error / (2.0 * sigma.unsqueeze(-1).square().clamp_min(eps)))  # [B, N, A]

    piece_visible = gt_poses[..., 0, 2].gt(0)  # [B, N]
    board_visible = gt_poses[..., 2].gt(0).any(dim=-1)  # [B, N]
    valid_pose = torch.where(is_board, board_visible, piece_visible)  # [B, N]
    valid = pad_gt_mask.squeeze(-1).bool() & valid_pose & board_areas.gt(eps).unsqueeze(-1)  # [B, N]
    return quality * valid.unsqueeze(-1)  # [B, N, A]


class YoloNASPoseTaskAlignedAssigner(nn.Module):
    """Assign anchors using class compatibility and image-space pose quality.

    Shape notation used below: B=batch, N=padded GT instances, A=anchors,
    C=classes, J=keypoints, and M=the safety-cap size.
    """

    def __init__(
        self,
        alignment_support_delta: float = 0.5,
        max_positives_per_gt: int = 32,
        piece_candidate_radius: float = 0.10,
        piece_sigma_q: float = 0.04,
        board_sigma_q: float = 0.025,
        board_class_id: int = 12,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.alignment_support_delta = alignment_support_delta
        self.max_positives_per_gt = max_positives_per_gt
        self.piece_candidate_radius = piece_candidate_radius
        self.piece_sigma_q = piece_sigma_q
        self.board_sigma_q = board_sigma_q
        self.board_class_id = board_class_id
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pred_class_probabilities: Tensor,
        pred_pose_coords: Tensor,
        anchor_points: Tensor,
        gt_labels: Tensor,
        gt_poses: Tensor,
        pad_gt_mask: Tensor,
        board_geometry: BoardGeometry,
        bg_index: int,
        class_alignment_progress: float = 1.0,
    ) -> YoloNASPoseAssignmentResult:
        """Build one GT assignment for each anchor without box or IoU supervision.

        Inputs are ``pred_class_probabilities`` [B,A,C], ``pred_pose_coords``
        [B,A,J,2], ``anchor_points`` [A,2], ``gt_labels`` [B,N,1],
        ``gt_poses`` [B,N,J,3], and ``pad_gt_mask`` [B,N,1].
        """
        batch_size, num_anchors, num_classes = pred_class_probabilities.shape  # [B, A, C]
        num_gt = gt_labels.shape[1]  # N
        num_keypoints = pred_pose_coords.shape[2]  # J

        if num_gt == 0:
            return YoloNASPoseAssignmentResult(
                assigned_labels=torch.full((batch_size, num_anchors), bg_index, dtype=torch.long, device=gt_labels.device),  # [B, A]
                assigned_poses=pred_pose_coords.new_zeros((batch_size, num_anchors, num_keypoints, 3)),  # [B, A, J, 3]
                assigned_gt_index=torch.zeros((batch_size, num_anchors), dtype=torch.long, device=gt_labels.device),  # [B, A]
                assigned_quality=pred_pose_coords.new_zeros((batch_size, num_anchors)),  # [B, A]
                positive_mask=torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=gt_labels.device),  # [B, A]
            )

        labels = gt_labels.squeeze(-1).long()  # [B, N]
        valid_gt = pad_gt_mask.squeeze(-1).bool()  # [B, N]
        is_board = labels.eq(self.board_class_id)  # [B, N]
        board_scale = board_geometry.areas.clamp_min(self.eps).sqrt()  # [B]

        # Piece candidates are local to the annotated center.
        piece_points = gt_poses[..., 0, :2].float()  # [B, N, 2]
        distances = torch.linalg.vector_norm(piece_points.unsqueeze(2) - anchor_points.float()[None, None, :, :], dim=-1)  # [B, N, A]
        piece_candidates = distances <= (self.piece_candidate_radius * board_scale)[:, None, None]  # [B, N, A]
        piece_candidates &= gt_poses[..., 0, 2].gt(0).unsqueeze(-1)  # [B, N, A]

        # Board candidates use the reconstructed full quadrilateral, or the
        # visible-point AABB when the homography fit was unavailable.
        board_region = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=anchor_points.device)  # [B, A]
        for batch_index in range(batch_size):
            if board_geometry.homography_valid[batch_index]:
                board_region[batch_index] = _points_inside_convex_quad(
                    anchor_points.float(), board_geometry.quadrilaterals[batch_index], self.eps
                )  # [A]
            elif board_geometry.area_valid[batch_index]:
                x1, y1, x2, y2 = board_geometry.aabbs[batch_index]
                board_region[batch_index] = (
                    (anchor_points[:, 0] >= x1)
                    & (anchor_points[:, 0] <= x2)
                    & (anchor_points[:, 1] >= y1)
                    & (anchor_points[:, 1] <= y2)
                )  # [A]

        candidate_mask = torch.where(is_board.unsqueeze(-1), board_region.unsqueeze(1), piece_candidates)  # [B, N, A]
        candidate_mask &= valid_gt.unsqueeze(-1) & board_geometry.area_valid[:, None, None]  # [B, N, A]

        quality = batch_localization_quality(
            gt_labels=gt_labels,
            gt_poses=gt_poses,
            pred_pose_coords=pred_pose_coords,
            board_areas=board_geometry.areas,
            pad_gt_mask=pad_gt_mask,
            piece_sigma_q=self.piece_sigma_q,
            board_sigma_q=self.board_sigma_q,
            board_class_id=self.board_class_id,
            eps=self.eps,
        )  # [B, N, A]

        safe_labels = labels.clamp(min=0, max=num_classes - 1)  # [B, N]
        class_probabilities = pred_class_probabilities.detach().float().unsqueeze(1).expand(-1, num_gt, -1, -1)  # [B, N, A, C]
        class_probabilities = class_probabilities.gather(
            dim=-1,
            index=safe_labels[:, :, None, None].expand(-1, -1, num_anchors, 1),  # [B, N, A, 1]
        ).squeeze(-1)  # [B, N, A]

        # Ranking is done in log space. During warmup the class exponent starts
        # at zero, so assignment is geometry-only; d is deliberately never used.
        class_exponent = float(class_alignment_progress)  # scalar tau in [0, 1]
        log_alignment = class_exponent * class_probabilities.clamp_min(self.eps).log()  # [B, N, A]
        log_alignment += quality.clamp_min(self.eps).log()  # [B, N, A]
        log_alignment = log_alignment.masked_fill(~candidate_mask, -torch.inf)  # [B, N, A]

        # Retain the likelihood support within delta of each GT's best anchor.
        # top-k is only a safety cap; it does not define the normal support size.
        best_log_alignment = log_alignment.max(dim=-1, keepdim=True).values  # [B, N, 1]
        support_threshold = best_log_alignment - self.alignment_support_delta  # [B, N, 1]
        safety_cap = min(self.max_positives_per_gt, num_anchors)  # M
        top_values, top_indices = torch.topk(log_alignment, k=safety_cap, dim=-1)  # each [B, N, M]
        within_support = torch.isfinite(top_values) & (top_values >= support_threshold)  # [B, N, M]
        selected = torch.zeros_like(candidate_mask)  # [B, N, A]
        selected.scatter_(dim=-1, index=top_indices, src=within_support)  # [B, N, A]

        positive_mask = selected.any(dim=1)  # [B, A]
        # A shared anchor goes to the GT whose current pose is the best match.
        conflict_quality = quality.masked_fill(~selected, -1.0)  # [B, N, A]
        assigned_gt_index = conflict_quality.argmax(dim=1)  # [B, A]

        assigned_labels = labels.gather(1, assigned_gt_index)  # [B, A]
        assigned_labels = torch.where(positive_mask, assigned_labels, torch.full_like(assigned_labels, bg_index))  # [B, A]
        assigned_quality = quality.gather(1, assigned_gt_index.unsqueeze(1)).squeeze(1)  # [B, A]
        assigned_quality = torch.where(positive_mask, assigned_quality, torch.zeros_like(assigned_quality))  # [B, A]

        pose_index = assigned_gt_index[:, :, None, None].expand(-1, -1, num_keypoints, 3)  # [B, A, J, 3]
        assigned_poses = gt_poses.gather(1, pose_index)  # [B, A, J, 3]

        return YoloNASPoseAssignmentResult(
            assigned_labels=assigned_labels,
            assigned_poses=assigned_poses,
            assigned_gt_index=assigned_gt_index,
            assigned_quality=assigned_quality,
            positive_mask=positive_mask,
        )


@register_loss(Losses.CHESS_YOLONAS_POSE_LOSS)
class ChessYoloNASPoseLoss(nn.Module):
    """Separate class, detection-quality, keypoint-score, and pose losses."""

    def __init__(
        self,
        board_oks_sigmas: Union[List[float], np.ndarray, Tensor],
        piece_oks_sigma: float,
        num_classes: int = 13,
        board_class_id: int = 12,
        classification_loss_weight: float = 1.0,
        quality_loss_weight: float = 1.0,
        pose_cls_loss_weight: float = 1.0,
        pose_reg_loss_weight: float = 1.0,
        board_localization_loss_multiplier: float = 1.0,
        alignment_support_delta: float = 0.5,
        max_positives_per_gt: int = 32,
        piece_candidate_radius: float = 0.10,
        piece_quality_sigma: float = 0.04,
        board_quality_sigma: float = 0.025,
        alignment_warmup_iterations: int = 1000,
        min_board_area: float = 16.0,
        homography_max_condition_number: float = 1e8,
    ):
        super().__init__()
        board_oks_sigmas = torch.as_tensor(board_oks_sigmas, dtype=torch.float32)
        if board_oks_sigmas.ndim != 1 or board_oks_sigmas.numel() == 0 or not board_oks_sigmas.gt(0).all():
            raise ValueError("board_oks_sigmas must be a non-empty one-dimensional sequence of positive values")
        if piece_oks_sigma <= 0:
            raise ValueError("piece_oks_sigma must be positive")
        if piece_quality_sigma <= 0 or board_quality_sigma <= 0:
            raise ValueError("Quality sigmas must be positive")
        if alignment_support_delta < 0:
            raise ValueError("alignment_support_delta must be non-negative")
        if max_positives_per_gt <= 0:
            raise ValueError("max_positives_per_gt must be positive")
        if piece_candidate_radius <= 0:
            raise ValueError("piece_candidate_radius must be positive")
        if alignment_warmup_iterations < 0:
            raise ValueError("alignment_warmup_iterations must be non-negative")
        if min_board_area <= 0:
            raise ValueError("min_board_area must be positive")
        if homography_max_condition_number <= 1:
            raise ValueError("homography_max_condition_number must be greater than one")

        self.num_classes = num_classes
        self.board_class_id = board_class_id
        self.classification_loss_weight = classification_loss_weight
        self.quality_loss_weight = quality_loss_weight
        self.pose_cls_loss_weight = pose_cls_loss_weight
        self.pose_reg_loss_weight = pose_reg_loss_weight
        self.board_localization_loss_multiplier = board_localization_loss_multiplier
        self.alignment_warmup_iterations = int(alignment_warmup_iterations)
        self.min_board_area = min_board_area
        self.homography_max_condition_number = homography_max_condition_number

        self.register_buffer("board_oks_sigmas", board_oks_sigmas, persistent=False)
        self.register_buffer("piece_oks_sigma", torch.tensor(piece_oks_sigma, dtype=torch.float32), persistent=False)
        self.register_buffer("_training_iteration", torch.zeros((), dtype=torch.long), persistent=False)

        self.assigner = YoloNASPoseTaskAlignedAssigner(
            alignment_support_delta=alignment_support_delta,
            max_positives_per_gt=max_positives_per_gt,
            piece_candidate_radius=piece_candidate_radius,
            piece_sigma_q=piece_quality_sigma,
            board_sigma_q=board_quality_sigma,
            board_class_id=board_class_id,
        )

    @torch.no_grad()
    def _unpack_flat_targets(self, targets: Tuple[Tensor, ...], batch_size: int) -> Mapping[str, Tensor]:
        """Unpack joints and labels; synthetic target boxes are intentionally ignored."""

        _, target_joints, target_class_labels = targets[:3]
        per_image_poses = undo_flat_collate_tensors_with_batch_index(target_joints, batch_size)
        per_image_labels = undo_flat_collate_tensors_with_batch_index(target_class_labels, batch_size)
        max_instances = max((len(labels) for labels in per_image_labels), default=0)

        padded_poses = []
        padded_labels = []
        padded_masks = []
        for poses, labels in zip(per_image_poses, per_image_labels):
            count = len(labels)
            pad_count = max_instances - count
            padded_poses.append(F.pad(poses, (0, 0, 0, 0, 0, pad_count), mode="constant", value=0))
            padded_labels.append(F.pad(labels.long(), (0, 0, 0, pad_count), mode="constant", value=0))
            mask = torch.zeros((max_instances, 1), dtype=torch.bool, device=labels.device)
            mask[:count] = True
            padded_masks.append(mask)

        return {
            "gt_class": torch.stack(padded_labels, dim=0),
            "gt_poses": torch.stack(padded_poses, dim=0),
            "pad_gt_mask": torch.stack(padded_masks, dim=0),
        }

    def _warmup_progress(self) -> float:
        if not self.training or self.alignment_warmup_iterations == 0:
            return 1.0
        progress = min(float(self._training_iteration.item()) / float(self.alignment_warmup_iterations), 1.0)
        self._training_iteration.add_(1)
        return progress

    @staticmethod
    def _positive_gt_groups(assign_result: YoloNASPoseAssignmentResult) -> Tensor:
        """Return a group index for each positive anchor's (batch, GT) pair."""

        positive_mask = assign_result.positive_mask  # [B, A]
        batch_indices = torch.arange(positive_mask.shape[0], device=positive_mask.device)[:, None].expand_as(positive_mask)  # [B, A]
        gt_pairs = torch.stack(
            (batch_indices[positive_mask], assign_result.assigned_gt_index[positive_mask]), dim=-1
        )  # [P, 2]
        _, group_indices = torch.unique(gt_pairs, dim=0, return_inverse=True)  # [G, 2], [P]
        return group_indices

    @staticmethod
    def _mean_per_gt(values: Tensor, group_indices: Tensor) -> Tensor:
        """Average anchors within each GT, then give every GT equal weight."""

        num_groups = int(group_indices.max().item()) + 1
        sums = values.new_zeros(num_groups).scatter_add_(0, group_indices, values)  # [G]
        counts = values.new_zeros(num_groups).scatter_add_(0, group_indices, torch.ones_like(values))  # [G]
        return (sums / counts.clamp_min(1.0)).mean()

    def forward(
        self,
        outputs: Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
        targets: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tensor]:
        _, predictions = outputs
        (
            pred_class_logits,
            pred_quality_logits,
            pred_pose_coords,
            pred_pose_logits,
            anchor_points,
            _num_anchors_list,
            _stride_tensor,
        ) = predictions

        unpacked = self._unpack_flat_targets(targets, batch_size=pred_class_logits.shape[0])
        gt_labels = unpacked["gt_class"]
        gt_poses = unpacked["gt_poses"]
        pad_gt_mask = unpacked["pad_gt_mask"]

        board_geometry = estimate_board_geometry(
            gt_labels=gt_labels,
            gt_poses=gt_poses,
            pad_gt_mask=pad_gt_mask,
            board_class_id=self.board_class_id,
            min_board_area=self.min_board_area,
            max_condition_number=self.homography_max_condition_number,
        )
        warmup_progress = self._warmup_progress()

        assign_result = self.assigner(
            pred_class_probabilities=pred_class_logits.detach().softmax(dim=-1),
            pred_pose_coords=pred_pose_coords.detach(),
            anchor_points=anchor_points,
            gt_labels=gt_labels,
            gt_poses=gt_poses,
            pad_gt_mask=pad_gt_mask,
            board_geometry=board_geometry,
            bg_index=self.num_classes,
            class_alignment_progress=warmup_progress,
        )

        positive_mask = assign_result.positive_mask
        if positive_mask.any():
            class_loss_per_anchor = F.cross_entropy(
                pred_class_logits[positive_mask],
                assign_result.assigned_labels[positive_mask],
                reduction="none",
            )  # [P]
            loss_cls = self._mean_per_gt(class_loss_per_anchor, self._positive_gt_groups(assign_result))
        else:
            loss_cls = pred_class_logits.sum() * 0.0

        # Early positives first learn foregroundness, then anneal to the detached
        # localization quality target. Background remains exactly zero.
        quality_target = torch.zeros_like(pred_quality_logits.squeeze(-1))
        positive_quality = (1.0 - warmup_progress) + warmup_progress * assign_result.assigned_quality
        quality_target = torch.where(positive_mask, positive_quality, quality_target)
        has_gt = pad_gt_mask.squeeze(-1).any(dim=-1)
        quality_valid_image = board_geometry.area_valid | ~has_gt
        quality_valid = quality_valid_image[:, None].expand_as(quality_target)
        if quality_valid.any():
            loss_quality = F.binary_cross_entropy_with_logits(
                pred_quality_logits.squeeze(-1)[quality_valid],
                quality_target[quality_valid],
                reduction="mean",
            )
        else:
            loss_quality = pred_quality_logits.sum() * 0.0

        loss_pose_reg, loss_pose_cls = self._pose_losses(
            pred_pose_coords=pred_pose_coords,
            pred_pose_logits=pred_pose_logits,
            assign_result=assign_result,
            board_areas=board_geometry.areas,
        )

        loss_cls = loss_cls * self.classification_loss_weight
        loss_quality = loss_quality * self.quality_loss_weight
        loss_pose_cls = loss_pose_cls * self.pose_cls_loss_weight
        loss_pose_reg = loss_pose_reg * self.pose_reg_loss_weight
        loss = loss_cls + loss_quality + loss_pose_cls + loss_pose_reg
        log_losses = torch.stack(
            (loss_cls.detach(), loss_quality.detach(), loss_pose_cls.detach(), loss_pose_reg.detach(), loss.detach())
        )
        return loss, log_losses

    @property
    def component_names(self) -> List[str]:
        return ["loss_cls", "loss_quality", "loss_pose_cls", "loss_pose_reg", "loss"]

    def _pose_losses(
        self,
        pred_pose_coords: Tensor,
        pred_pose_logits: Tensor,
        assign_result: YoloNASPoseAssignmentResult,
        board_areas: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        positive_mask = assign_result.positive_mask
        if not positive_mask.any():
            zero = pred_pose_coords.sum() * 0.0
            return zero, pred_pose_logits.sum() * 0.0

        predicted_coords = pred_pose_coords[positive_mask].float()
        predicted_logits = pred_pose_logits[positive_mask]
        target_pose = assign_result.assigned_poses[positive_mask].float()
        labels = assign_result.assigned_labels[positive_mask]
        instance_areas = board_areas[:, None].expand_as(positive_mask)[positive_mask].float().clamp_min(1e-9)
        is_board = labels.eq(self.board_class_id)
        num_instances, num_keypoints = predicted_logits.shape

        # Board landmarks are all applicable; piece records only use keypoint 0.
        applicable = torch.zeros((num_instances, num_keypoints), dtype=torch.bool, device=predicted_logits.device)
        applicable[is_board] = True
        applicable[~is_board, 0] = True
        visible = target_pose[..., 2].gt(0)

        pose_cls_per_keypoint = F.binary_cross_entropy_with_logits(predicted_logits, visible.float(), reduction="none")
        pose_cls_per_instance = (pose_cls_per_keypoint * applicable).sum(dim=-1) / applicable.sum(dim=-1).clamp_min(1)
        positive_gt_groups = self._positive_gt_groups(assign_result)
        loss_pose_cls = self._mean_per_gt(pose_cls_per_instance, positive_gt_groups)

        squared_distance = ((predicted_coords - target_pose[..., :2]) ** 2).sum(dim=-1)
        sigmas = self.board_oks_sigmas.to(predicted_coords.device).expand(num_instances, -1).clone()
        sigmas[~is_board] = self.piece_oks_sigma.to(predicted_coords.device)
        exponent = squared_distance / ((2.0 * sigmas).square() * instance_areas[:, None] * 2.0 + 1e-9)
        regression_per_keypoint = 1.0 - torch.exp(-exponent)
        regression_mask = visible & applicable
        regression_per_instance = (regression_per_keypoint * regression_mask).sum(dim=-1) / regression_mask.sum(dim=-1).clamp_min(1)

        if self.board_localization_loss_multiplier != 1.0:
            regression_per_instance = torch.where(
                is_board,
                regression_per_instance * self.board_localization_loss_multiplier,
                regression_per_instance,
            )
        # g is not a loss weight. Anchors are averaged within each GT so an
        # adaptive support size does not give easy objects more total weight.
        loss_pose_reg = self._mean_per_gt(regression_per_instance, positive_gt_groups)
        return loss_pose_reg, loss_pose_cls
