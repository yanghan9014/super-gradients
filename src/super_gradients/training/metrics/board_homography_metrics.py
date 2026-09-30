"""Board matching by displacement in rectified chess-square coordinates."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from super_gradients.training.metrics.pose_estimation_utils import ImageKeypointMatchingResult


# a1, a8, h1, h8, center, a45, h45, 1de, 8de. One unit is one square.
BOARD_REFERENCE_POINTS = np.array([(0, 8), (0, 0), (8, 8), (8, 0), (4, 4), (0, 4), (8, 4), (4, 8), (4, 0)], dtype=np.float64)
# Match ChessVision's inference homography point order (corners, edges, center).
_REFERENCE_ORDER = [0, 1, 2, 3, 5, 6, 7, 8, 4]


def _fit_homography(points: np.ndarray, usable: np.ndarray, method: int) -> Optional[np.ndarray]:
    indices = [i for i in _REFERENCE_ORDER if usable[i] and np.isfinite(points[i]).all()]
    if len(indices) < 4:
        return None
    try:
        # Prediction fitting mirrors ChessVision homography_from_results: RANSAC
        # with OpenCV's default 3-board-unit reprojection threshold. Annotations
        # use all available reference points without outlier rejection (method=0).
        homography, _ = cv2.findHomography(points[indices], BOARD_REFERENCE_POINTS[indices], method=method)
        if homography is None or not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
            return None
        return homography
    except (cv2.error, np.linalg.LinAlgError):
        return None


def _project(points: np.ndarray, homography: np.ndarray) -> Optional[np.ndarray]:
    projected = np.column_stack((points, np.ones(len(points)))) @ homography.T
    scale = np.max(np.abs(projected), axis=1)
    if not np.isfinite(projected).all() or np.any(np.abs(projected[:, 2]) <= np.finfo(np.float64).eps * scale):
        return None
    result = projected[:, :2] / projected[:, 2:3]
    return result if np.isfinite(result).all() else None


def _annotation_reference(points: np.ndarray, visibility: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    usable = (visibility > 0) & np.isfinite(points).all(axis=1)
    homography = _fit_homography(points, usable, method=0)
    if homography is None:
        return None
    try:
        reference_points = _project(BOARD_REFERENCE_POINTS, np.linalg.inv(homography))
    except np.linalg.LinAlgError:
        return None
    if reference_points is None:
        return None
    # Keep the actual annotations. Only missing/out-of-frame points are inferred
    # from the annotation homography, so cropping cannot hide an inaccurate edge.
    reference_points[usable] = points[usable]
    reference_coords = _project(reference_points, homography)
    if reference_coords is None:
        return None
    return reference_points, reference_coords


def _match_errors(errors: np.ndarray, scores: Tensor, tolerance: float) -> ImageKeypointMatchingResult:
    """Greedy one-to-one matching, with predictions already in confidence order."""
    matched = torch.zeros((len(scores), 1), dtype=torch.bool)
    targets_matched = np.zeros(errors.shape[1], dtype=bool)
    for pred_index, distances in enumerate(errors):
        for target_index in np.argsort(distances, kind="stable"):
            # Allow only numerical fitting noise at the inclusive boundary.
            if distances[target_index] > tolerance + 1e-6:
                break
            if not targets_matched[target_index]:
                targets_matched[target_index] = True
                matched[pred_index, 0] = True
                break
    return ImageKeypointMatchingResult(
        preds_matched=matched,
        preds_to_ignore=torch.zeros_like(matched),
        preds_scores=scores,
        num_targets=errors.shape[1],
    )


def compute_board_homography_matching(
    predicted_poses: Tensor,
    predicted_scores: Tensor,
    targets: Tensor,
    targets_visibilities: Tensor,
    tolerance: float,
    top_k: int,
    predicted_pose_scores: Optional[Tensor] = None,
    keypoint_confidence_threshold: float = 0.7,
) -> Tuple[ImageKeypointMatchingResult, ImageKeypointMatchingResult]:
    """Return strict and 180-degree-relaxed board matches at one geometric cutoff.

    Fit image-to-board homographies, then transform the SAME annotation reference
    points with both. A pair passes only when every absolute x and y difference is
    <= tolerance in square units. The relaxed result also permits (x,y)->(8-x,8-y)
    for the entire predicted board; individual points cannot choose their own flip.

    Missing annotation landmarks are projected from its fitted homography. Fewer
    than four usable points, degenerate homographies, and non-finite projections
    cannot produce a match. Such targets remain in the recall denominator.
    Predicted keypoint scores select fitting points as in ChessVision inference;
    when omitted, all finite predicted points are used.
    """
    indices = torch.argsort(predicted_scores, descending=True, stable=True)[:top_k]
    scores = predicted_scores[indices].detach().cpu()
    poses = predicted_poses[indices, :, :2].detach().cpu().numpy().astype(np.float64)
    targets_np = targets.detach().cpu().numpy().astype(np.float64)
    visibility = targets_visibilities.detach().cpu().numpy()
    usable_predictions = np.ones(poses.shape[:2], dtype=bool)
    if predicted_pose_scores is not None:
        point_scores = predicted_pose_scores[indices].detach().cpu().numpy()
        usable_predictions = np.isfinite(point_scores) & (point_scores >= keypoint_confidence_threshold)

    homographies = [_fit_homography(points, usable, cv2.RANSAC) for points, usable in zip(poses, usable_predictions)]
    strict_errors = np.full((len(poses), len(targets_np)), np.inf)
    relaxed_errors = np.full_like(strict_errors, np.inf)
    for target_index, (points, visible) in enumerate(zip(targets_np, visibility)):
        reference = _annotation_reference(points, visible)
        if reference is None:
            continue
        reference_points, reference_coords = reference
        for pred_index, homography in enumerate(homographies):
            if homography is None:
                continue
            predicted_coords = _project(reference_points, homography)
            if predicted_coords is None:
                continue
            strict_error = np.abs(predicted_coords - reference_coords).max()
            flipped_error = np.abs((8.0 - predicted_coords) - reference_coords).max()
            strict_errors[pred_index, target_index] = strict_error
            relaxed_errors[pred_index, target_index] = min(strict_error, flipped_error)

    return _match_errors(strict_errors, scores, tolerance), _match_errors(relaxed_errors, scores, tolerance)
