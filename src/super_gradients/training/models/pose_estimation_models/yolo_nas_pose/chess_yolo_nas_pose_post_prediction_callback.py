from typing import List, Tuple

import torch
import torchvision
from torch import Tensor

from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from super_gradients.module_interfaces.pose_estimation_post_prediction_callback import ChessPoseEstimationPredictions

# Board class ID in the new 13-class schema (0-11 pieces, 12 board)
BOARD_CLASS_ID = 12

# Board keypoint names (indices 0-8 in the 9-keypoint layout)
BOARD_KEYPOINT_NAMES = ["a1", "a8", "h1", "h8", "center", "a45", "h45", "1de", "8de"]


class ChessYoloNASPosePostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """
    Post-prediction callback for Chess YoloNASPose model.

    Handles two types of detections:
    - Pieces (classes 0-11): Use keypoint[0] only, NMS on bboxes
    - Board (class 12): Single detection with 9 keypoints, no NMS needed (take highest-confidence board)
    """

    def __init__(
        self,
        pose_confidence_threshold: float,
        nms_iou_threshold: float,
        pre_nms_max_predictions: int,
        post_nms_max_predictions: int,
        piece_kp_threshold: float = 0.0,
        board_score_top_k: int = 5,
    ):
        """
        :param pose_confidence_threshold: Detection confidence threshold
        :param nms_iou_threshold:         IoU threshold for NMS step (pieces only)
        :param pre_nms_max_predictions:   Max predictions entering NMS
        :param post_nms_max_predictions:  Max predictions after NMS
        :param piece_kp_threshold:        Min confidence score for piece keypoint 0
        :param board_score_top_k:         Number of top keypoints to average for board scoring
        """
        if post_nms_max_predictions > pre_nms_max_predictions:
            raise ValueError("post_nms_max_predictions must be less than pre_nms_max_predictions")

        super().__init__()
        self.pose_confidence_threshold = pose_confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions
        self.piece_kp_threshold = piece_kp_threshold
        self.board_score_top_k = board_score_top_k

    @torch.no_grad()
    def __call__(self, outputs: Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], ...]) -> List[ChessPoseEstimationPredictions]:
        """
        Decode YoloNASPose predictions into ChessPoseEstimationPredictions.

        :param outputs: Output of the model's forward() method
        :return:        List of decoded predictions for each image in the batch
        """
        predictions = outputs[0]

        decoded_predictions: List[ChessPoseEstimationPredictions] = []
        for pred_values in zip(*predictions):
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores = pred_values
            # pred_bboxes_xyxy  [Anchors, 4] in XYXY format
            # pred_bboxes_conf  [Anchors, NumClasses] confidence scores [0..1]
            # pred_pose_coords  [Anchors, NumJoints, 2] in (x,y) format
            # pred_pose_scores  [Anchors, NumJoints] keypoint confidence [0..1]

            # Piece predictions: max over classes 0 to 11
            piece_conf, piece_label = torch.max(pred_bboxes_conf[:, :BOARD_CLASS_ID], dim=1)
            piece_mask = piece_conf >= self.pose_confidence_threshold
            if self.piece_kp_threshold > 0.0:
                piece_mask = piece_mask & (pred_pose_scores[:, 0] >= self.piece_kp_threshold)

            # Board predictions: class 12
            board_conf = pred_bboxes_conf[:, BOARD_CLASS_ID]
            board_mask = board_conf >= self.pose_confidence_threshold

            # ---- Process pieces (classes 0-11): standard NMS ----
            piece_final = self._process_pieces(
                piece_conf[piece_mask],
                piece_label[piece_mask],
                pred_bboxes_conf[piece_mask],
                pred_bboxes_xyxy[piece_mask],
                pred_pose_coords[piece_mask],
                pred_pose_scores[piece_mask],
            )

            # ---- Process board (class 12): take the best board by combined score ----
            board_final = self._process_board(
                board_conf[board_mask],
                pred_bboxes_conf.new_full((board_mask.sum(),), BOARD_CLASS_ID, dtype=torch.long),
                pred_bboxes_conf[board_mask],
                pred_bboxes_xyxy[board_mask],
                pred_pose_coords[board_mask],
                pred_pose_scores[board_mask],
            )

            # ---- Combine: board first, then pieces ----
            parts_poses, parts_pose_scores, parts_scores, parts_labels, parts_bboxes = [], [], [], [], []

            if board_final is not None:
                parts_poses.append(board_final["poses"])
                parts_pose_scores.append(board_final["pose_scores"])
                parts_scores.append(board_final["scores"])
                parts_labels.append(board_final["labels"])
                parts_bboxes.append(board_final["bboxes"])

            if piece_final is not None:
                parts_poses.append(piece_final["poses"])
                parts_pose_scores.append(piece_final["pose_scores"])
                parts_scores.append(piece_final["scores"])
                parts_labels.append(piece_final["labels"])
                parts_bboxes.append(piece_final["bboxes"])

            if parts_poses:
                num_joints = pred_pose_coords.shape[1]
                final_poses = torch.cat(parts_poses, dim=0)
                final_pose_scores = torch.cat(parts_pose_scores, dim=0)
                final_scores = torch.cat(parts_scores, dim=0)
                final_labels = torch.cat(parts_labels, dim=0)
                final_bboxes = torch.cat(parts_bboxes, dim=0)
            else:
                num_joints = pred_pose_coords.shape[1] if pred_pose_coords.dim() == 3 else 9
                final_poses = pred_pose_coords.new_zeros((0, num_joints, 2))
                final_pose_scores = pred_pose_scores.new_zeros((0, num_joints))
                final_scores = pred_bboxes_conf.new_zeros((0,))
                final_labels = pred_bboxes_conf.new_zeros((0,), dtype=torch.long)
                final_bboxes = pred_bboxes_xyxy.new_zeros((0, 4))

            k = self.post_nms_max_predictions
            decoded_predictions.append(
                ChessPoseEstimationPredictions(
                    poses=final_poses[:k],
                    pose_scores=final_pose_scores[:k],
                    scores=final_scores[:k],
                    labels=final_labels[:k],
                    bboxes_xyxy=final_bboxes[:k],
                )
            )

        return decoded_predictions

    def _process_pieces(self, cls_conf, cls_label, bboxes_conf, bboxes_xyxy, pose_coords, pose_scores):
        """Apply top-K + NMS to piece detections. Returns dict or None."""
        if cls_conf.numel() == 0:
            return None

        # Top-K filtering
        if cls_conf.size(0) > self.pre_nms_max_predictions:
            topk = torch.topk(cls_conf, k=self.pre_nms_max_predictions, largest=True, sorted=True)
            idx = topk.indices
            cls_conf = cls_conf[idx]
            cls_label = cls_label[idx]
            bboxes_conf = bboxes_conf[idx]
            bboxes_xyxy = bboxes_xyxy[idx]
            pose_coords = pose_coords[idx]
            pose_scores = pose_scores[idx]

        # Per-class NMS
        nms_idx = torchvision.ops.boxes.batched_nms(
            boxes=bboxes_xyxy, scores=cls_conf, idxs=cls_label, iou_threshold=self.nms_iou_threshold
        )

        return {
            "poses": pose_coords[nms_idx],         # [M, J, 2]
            "pose_scores": pose_scores[nms_idx],    # [M, J]
            "scores": bboxes_conf[nms_idx],         # [M, C]
            "labels": cls_label[nms_idx],           # [M]
            "bboxes": bboxes_xyxy[nms_idx],         # [M, 4]
        }

    def _process_board(self, cls_conf, cls_label, bboxes_conf, bboxes_xyxy, pose_coords, pose_scores):
        """Take the best board detection using a combined confidence and keypoint score. Returns dict or None."""
        if cls_conf.numel() == 0:
            return None

        # Take the single best board detection weighted by top-K keypoint score quality (geometric mean)
        k = min(pose_scores.shape[1], self.board_score_top_k)
        if k > 0:
            topk_scores = torch.topk(pose_scores, k=k, dim=1).values.clamp(min=1e-8)
            board_kpt_score = torch.exp(torch.log(topk_scores).mean(dim=1))
        else:
            board_kpt_score = pose_scores.new_ones(pose_scores.shape[0])
            
        board_score = cls_conf * board_kpt_score
        best_idx = torch.argmax(board_score)

        return {
            "poses": pose_coords[best_idx].unsqueeze(0),         # [1, J, 2]
            "pose_scores": pose_scores[best_idx].unsqueeze(0),    # [1, J]
            "scores": bboxes_conf[best_idx].unsqueeze(0),         # [1, C]
            "labels": cls_label[best_idx].unsqueeze(0),           # [1]
            "bboxes": bboxes_xyxy[best_idx].unsqueeze(0),         # [1, 4]
        }
