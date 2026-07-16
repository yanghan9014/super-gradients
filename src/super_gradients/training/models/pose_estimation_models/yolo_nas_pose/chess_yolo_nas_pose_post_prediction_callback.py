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

    Piece classes are selected from the conditional class distribution, and
    pieces are objectness-thresholded and top-k filtered without using their class
    probability as a gate. Duplicate pieces are resolved after homography
    projection by keeping the highest fused score per board square.
    Boards are not objectness-thresholded here.
    The single board candidate is selected by its fused detection score.
    """

    def __init__(
        self,
        pose_confidence_threshold: float,
        pre_nms_max_predictions: int = 300,
        post_nms_max_predictions: int = 50,
    ):
        if post_nms_max_predictions > pre_nms_max_predictions:
            raise ValueError("post_nms_max_predictions must be less than pre_nms_max_predictions")
        super().__init__()
        # Keep the public SuperGradients argument name for compatibility, but
        # interpret it as the minimum scalar objectness for piece anchors only.
        self.piece_objectness_threshold = pose_confidence_threshold
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions

    @torch.no_grad()
    def __call__(self, outputs: Tuple[Tuple[Tensor, ...], ...]) -> List[ChessPoseEstimationPredictions]:
        predictions = outputs[0]
        decoded_predictions: List[ChessPoseEstimationPredictions] = []

        for pred_values in zip(*predictions):
            fused_scores, class_probabilities, objectness_scores, pose_coords, keypoint_scores = pred_values

            objectness = objectness_scores.squeeze(-1)
            predicted_labels = class_probabilities.argmax(dim=1)
            piece_mask = (predicted_labels < BOARD_CLASS_ID) & (objectness >= self.piece_objectness_threshold)

            piece_result = self._process_pieces(
                confidence=objectness[piece_mask],
                labels=predicted_labels[piece_mask],
                fused_scores=fused_scores[piece_mask],
                class_probabilities=class_probabilities[piece_mask],
                objectness_scores=objectness_scores[piece_mask],
                pose_coords=pose_coords[piece_mask],
                keypoint_scores=keypoint_scores[piece_mask],
            )

            board_mask = predicted_labels == BOARD_CLASS_ID
            board_confidence = fused_scores[board_mask, BOARD_CLASS_ID]
            board_result = self._process_board(
                confidence=board_confidence,
                fused_scores=fused_scores[board_mask],
                class_probabilities=class_probabilities[board_mask],
                objectness_scores=objectness_scores[board_mask],
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
                final_objectness_scores = torch.cat([part["objectness_scores"] for part in parts], dim=0)
                final_labels = torch.cat([part["labels"] for part in parts], dim=0)
            else:
                num_joints = pose_coords.shape[1]
                final_poses = pose_coords.new_zeros((0, num_joints, 2))
                final_keypoint_scores = keypoint_scores.new_zeros((0, num_joints))
                final_fused_scores = fused_scores.new_zeros((0, fused_scores.shape[-1]))
                final_class_probabilities = class_probabilities.new_zeros((0, class_probabilities.shape[-1]))
                final_objectness_scores = objectness_scores.new_zeros((0, 1))
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
                    objectness_scores=final_objectness_scores[:limit],
                )
            )

        return decoded_predictions

    def _process_pieces(
        self,
        confidence: Tensor,
        labels: Tensor,
        fused_scores: Tensor,
        class_probabilities: Tensor,
        objectness_scores: Tensor,
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
            "objectness_scores": objectness_scores[indices],
            "labels": labels[indices],
        }

    def _process_board(
        self,
        confidence: Tensor,
        fused_scores: Tensor,
        class_probabilities: Tensor,
        objectness_scores: Tensor,
        pose_coords: Tensor,
        keypoint_scores: Tensor,
    ) -> Optional[Dict[str, Tensor]]:
        if confidence.numel() == 0:
            return None

        best = torch.argmax(confidence)
        return {
            "poses": pose_coords[best].unsqueeze(0),
            "keypoint_scores": keypoint_scores[best].unsqueeze(0),
            "fused_scores": fused_scores[best].unsqueeze(0),
            "class_probabilities": class_probabilities[best].unsqueeze(0),
            "objectness_scores": objectness_scores[best].unsqueeze(0),
            "labels": torch.full((1,), BOARD_CLASS_ID, dtype=torch.long, device=confidence.device),
        }
