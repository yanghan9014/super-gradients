"""Post-processing for the box-free chess YOLO-NAS-Pose output contract."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from super_gradients.module_interfaces.pose_estimation_post_prediction_callback import ChessPoseEstimationPredictions


BOARD_CLASS_ID = 12
BOARD_KEYPOINT_NAMES = ["a1", "a8", "h1", "h8", "center", "a45", "h45", "1de", "8de"]


class ChessYoloNASPosePostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """Threshold box-free predictions and retain task-specific candidates.

    Pieces are only top-k filtered here. Duplicate pieces are resolved after
    homography projection by keeping the highest fused score per board square.
    The single board candidate is selected with fused detection score and its
    strongest keypoint-visibility scores.
    """

    def __init__(
        self,
        pose_confidence_threshold: float,
        nms_iou_threshold: float = 0.7,
        pre_nms_max_predictions: int = 300,
        post_nms_max_predictions: int = 50,
        piece_kp_threshold: float = 0.0,
        board_score_top_k: int = 5,
    ):
        if post_nms_max_predictions > pre_nms_max_predictions:
            raise ValueError("post_nms_max_predictions must be less than pre_nms_max_predictions")
        super().__init__()
        self.pose_confidence_threshold = pose_confidence_threshold
        # Kept as compatibility-only configuration; no box NMS is performed.
        self.nms_iou_threshold = nms_iou_threshold
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions
        self.piece_kp_threshold = piece_kp_threshold
        self.board_score_top_k = board_score_top_k

    @torch.no_grad()
    def __call__(self, outputs: Tuple[Tuple[Tensor, ...], ...]) -> List[ChessPoseEstimationPredictions]:
        predictions = outputs[0]
        decoded_predictions: List[ChessPoseEstimationPredictions] = []

        for pred_values in zip(*predictions):
            fused_scores, class_probabilities, quality_scores, pose_coords, keypoint_scores = pred_values

            piece_confidence, piece_labels = fused_scores[:, :BOARD_CLASS_ID].max(dim=1)
            piece_mask = piece_confidence >= self.pose_confidence_threshold
            if self.piece_kp_threshold > 0.0:
                piece_mask &= keypoint_scores[:, 0] >= self.piece_kp_threshold

            piece_result = self._process_pieces(
                confidence=piece_confidence[piece_mask],
                labels=piece_labels[piece_mask],
                fused_scores=fused_scores[piece_mask],
                class_probabilities=class_probabilities[piece_mask],
                quality_scores=quality_scores[piece_mask],
                pose_coords=pose_coords[piece_mask],
                keypoint_scores=keypoint_scores[piece_mask],
            )

            board_confidence = fused_scores[:, BOARD_CLASS_ID]
            board_mask = board_confidence >= self.pose_confidence_threshold
            board_result = self._process_board(
                confidence=board_confidence[board_mask],
                fused_scores=fused_scores[board_mask],
                class_probabilities=class_probabilities[board_mask],
                quality_scores=quality_scores[board_mask],
                pose_coords=pose_coords[board_mask],
                keypoint_scores=keypoint_scores[board_mask],
            )

            parts: List[Dict[str, Tensor]] = []
            if board_result is not None:
                parts.append(board_result)
            if piece_result is not None:
                parts.append(piece_result)

            if parts:
                final_poses = torch.cat([part["poses"] for part in parts], dim=0)
                final_keypoint_scores = torch.cat([part["keypoint_scores"] for part in parts], dim=0)
                final_fused_scores = torch.cat([part["fused_scores"] for part in parts], dim=0)
                final_class_probabilities = torch.cat([part["class_probabilities"] for part in parts], dim=0)
                final_quality_scores = torch.cat([part["quality_scores"] for part in parts], dim=0)
                final_labels = torch.cat([part["labels"] for part in parts], dim=0)
            else:
                num_joints = pose_coords.shape[1]
                final_poses = pose_coords.new_zeros((0, num_joints, 2))
                final_keypoint_scores = keypoint_scores.new_zeros((0, num_joints))
                final_fused_scores = fused_scores.new_zeros((0, fused_scores.shape[-1]))
                final_class_probabilities = class_probabilities.new_zeros((0, class_probabilities.shape[-1]))
                final_quality_scores = quality_scores.new_zeros((0, 1))
                final_labels = torch.zeros((0,), dtype=torch.long, device=fused_scores.device)

            limit = self.post_nms_max_predictions
            decoded_predictions.append(
                ChessPoseEstimationPredictions(
                    poses=final_poses[:limit],
                    pose_scores=final_keypoint_scores[:limit],
                    scores=final_fused_scores[:limit],
                    labels=final_labels[:limit],
                    bboxes_xyxy=None,
                    class_probabilities=final_class_probabilities[:limit],
                    quality_scores=final_quality_scores[:limit],
                )
            )

        return decoded_predictions

    def _process_pieces(
        self,
        confidence: Tensor,
        labels: Tensor,
        fused_scores: Tensor,
        class_probabilities: Tensor,
        quality_scores: Tensor,
        pose_coords: Tensor,
        keypoint_scores: Tensor,
    ) -> Optional[Dict[str, Tensor]]:
        if confidence.numel() == 0:
            return None

        limit = min(confidence.shape[0], self.pre_nms_max_predictions)
        indices = torch.topk(confidence, k=limit, largest=True, sorted=True).indices
        return {
            "poses": pose_coords[indices],
            "keypoint_scores": keypoint_scores[indices],
            "fused_scores": fused_scores[indices],
            "class_probabilities": class_probabilities[indices],
            "quality_scores": quality_scores[indices],
            "labels": labels[indices],
        }

    def _process_board(
        self,
        confidence: Tensor,
        fused_scores: Tensor,
        class_probabilities: Tensor,
        quality_scores: Tensor,
        pose_coords: Tensor,
        keypoint_scores: Tensor,
    ) -> Optional[Dict[str, Tensor]]:
        if confidence.numel() == 0:
            return None

        k = min(keypoint_scores.shape[1], self.board_score_top_k)
        if k > 0:
            strongest = torch.topk(keypoint_scores, k=k, dim=1).values.clamp_min(1e-8)
            usable_keypoint_score = strongest.log().mean(dim=1).exp()
        else:
            usable_keypoint_score = confidence.new_ones(confidence.shape)

        best = torch.argmax(confidence * usable_keypoint_score)
        return {
            "poses": pose_coords[best].unsqueeze(0),
            "keypoint_scores": keypoint_scores[best].unsqueeze(0),
            "fused_scores": fused_scores[best].unsqueeze(0),
            "class_probabilities": class_probabilities[best].unsqueeze(0),
            "quality_scores": quality_scores[best].unsqueeze(0),
            "labels": torch.full((1,), BOARD_CLASS_ID, dtype=torch.long, device=confidence.device),
        }
