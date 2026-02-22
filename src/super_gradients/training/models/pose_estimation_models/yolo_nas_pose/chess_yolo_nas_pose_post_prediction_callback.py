from typing import List, Tuple, Dict

import torch
import torchvision
from torch import Tensor

from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from super_gradients.module_interfaces.pose_estimation_post_prediction_callback import ChessPoseEstimationPredictions

# Board is the last class (index 12 in a 13-class model: 12 pieces + 1 board)
BOARD_CLASS_ID = 12

# Number of board keypoints (must match model)
NUM_BOARD_KEYPOINTS = 9


class ChessYoloNASPosePostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """
    A post-prediction callback for the Chess YoloNASPose model.
    Performs confidence thresholding, Top-K and NMS for pieces,
    and selects the top-1 board detection with 9 structured keypoints.

    All output poses have shape [N, 9, 2] to match the metrics' num_joints=9:
      - Pieces: only keypoint 0 is populated, rest are zeros
      - Board:  all 9 keypoints are populated
    """

    def __init__(
        self,
        pose_confidence_threshold: float,
        nms_iou_threshold: float,
        pre_nms_max_predictions: int,
        post_nms_max_predictions: int,
    ):
        if post_nms_max_predictions > pre_nms_max_predictions:
            raise ValueError("post_nms_max_predictions must be less than pre_nms_max_predictions")

        super().__init__()
        self.pose_confidence_threshold = pose_confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.pre_nms_max_predictions = pre_nms_max_predictions
        self.post_nms_max_predictions = post_nms_max_predictions
        self.class_agnostic_nms = False

    @torch.no_grad()
    def __call__(self, outputs: Tuple[Tuple[Tensor, ...], ...]) -> List[ChessPoseEstimationPredictions]:
        # First element is model decoded predictions, second is raw logits for loss
        predictions = outputs[0]

        decoded_predictions: List[ChessPoseEstimationPredictions] = []
        for pred_values in zip(*predictions):
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores, pred_board_coords, pred_board_scores = pred_values
            # pred_bboxes_xyxy  [Anchors, 4]
            # pred_bboxes_conf  [Anchors, C]  (C=13)
            # pred_pose_coords  [Anchors, 1, 2]
            # pred_pose_scores  [Anchors, 1]
            # pred_board_coords [Anchors, 9, 2]
            # pred_board_scores [Anchors, 9]

            pred_cls_conf, pred_cls_label = torch.max(pred_bboxes_conf, dim=1)
            conf_mask = pred_cls_conf >= self.pose_confidence_threshold

            pred_cls_conf = pred_cls_conf[conf_mask].float()
            pred_cls_label = pred_cls_label[conf_mask]
            pred_bboxes_conf = pred_bboxes_conf[conf_mask].float()
            pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask].float()
            pred_pose_coords = pred_pose_coords[conf_mask].float()
            pred_pose_scores = pred_pose_scores[conf_mask].float()
            pred_board_coords = pred_board_coords[conf_mask].float()
            pred_board_scores = pred_board_scores[conf_mask].float()

            piece_mask = pred_cls_label != BOARD_CLASS_ID
            board_mask = pred_cls_label == BOARD_CLASS_ID

            # ---- Piece predictions ----
            piece_cls_conf = pred_cls_conf[piece_mask]
            piece_cls_label = pred_cls_label[piece_mask]
            piece_bboxes_conf = pred_bboxes_conf[piece_mask]
            piece_bboxes_xyxy = pred_bboxes_xyxy[piece_mask]
            piece_pose_coords = pred_pose_coords[piece_mask]  # [N_piece, 1, 2]

            piece_final_poses = piece_final_scores = piece_final_labels = piece_final_bboxes = None

            if piece_bboxes_conf.size(0) > 0:
                # Top-K filtering
                if piece_bboxes_conf.size(0) > self.pre_nms_max_predictions:
                    topk_candidates = torch.topk(piece_cls_conf, k=self.pre_nms_max_predictions, dim=0, largest=True, sorted=True)
                    topk_idx = topk_candidates.indices
                    piece_cls_conf = piece_cls_conf[topk_idx].float()
                    piece_cls_label = piece_cls_label[topk_idx]
                    piece_bboxes_conf = piece_bboxes_conf[topk_idx]
                    piece_bboxes_xyxy = piece_bboxes_xyxy[topk_idx]
                    piece_pose_coords = piece_pose_coords[topk_idx]

                # NMS
                if self.class_agnostic_nms:
                    idx_to_keep = torchvision.ops.boxes.nms(piece_bboxes_xyxy.float(), piece_cls_conf.float(), iou_threshold=self.nms_iou_threshold)
                else:
                    idx_to_keep = torchvision.ops.boxes.batched_nms(
                        boxes=piece_bboxes_xyxy.float(), scores=piece_cls_conf.float(), idxs=piece_cls_label, iou_threshold=self.nms_iou_threshold
                    )

                kept_coords = piece_pose_coords[idx_to_keep]  # [K, 1, 2]
                n_kept = kept_coords.shape[0]

                # Pad to [K, 9, 2]: keypoint 0 gets the real coords, rest are zeros
                piece_final_poses = torch.zeros(n_kept, NUM_BOARD_KEYPOINTS, 2, device=kept_coords.device, dtype=kept_coords.dtype)
                piece_final_poses[:, 0, :] = kept_coords[:, 0, :]

                piece_final_scores = piece_bboxes_conf[idx_to_keep]
                piece_final_labels = piece_cls_label[idx_to_keep]
                piece_final_bboxes = piece_bboxes_xyxy[idx_to_keep]

            # ---- Board prediction (top-1 board detection → single entry with 9 keypoints) ----
            board_final_poses = board_final_scores = board_final_labels = board_final_bboxes = None

            board_cls_conf = pred_cls_conf[board_mask]
            board_bboxes_conf = pred_bboxes_conf[board_mask]
            board_bboxes_xyxy_filtered = pred_bboxes_xyxy[board_mask]
            board_kpts = pred_board_coords[board_mask]  # [N_board, 9, 2]

            if board_cls_conf.numel() > 0:
                best_board_idx = torch.argmax(board_cls_conf)
                best_board_kpts = board_kpts[best_board_idx]  # [9, 2]
                best_board_bbox = board_bboxes_xyxy_filtered[best_board_idx]  # [4]
                best_board_conf = board_bboxes_conf[best_board_idx]  # [C]

                # Single board prediction with 9 keypoints: [1, 9, 2]
                board_final_poses = best_board_kpts.unsqueeze(0)  # [1, 9, 2]
                board_final_labels = torch.tensor([BOARD_CLASS_ID], dtype=torch.long, device=pred_cls_label.device)
                board_final_bboxes = best_board_bbox.unsqueeze(0)  # [1, 4]
                board_final_scores = best_board_conf.unsqueeze(0)  # [1, C]

            # ---- Combine ----
            final_poses_parts = []
            final_scores_parts = []
            final_labels_parts = []
            final_bboxes_parts = []

            # Board first so it survives truncation
            if board_final_poses is not None:
                final_poses_parts.append(board_final_poses)
                final_scores_parts.append(board_final_scores)
                final_labels_parts.append(board_final_labels)
                final_bboxes_parts.append(board_final_bboxes)

            if piece_final_poses is not None:
                final_poses_parts.append(piece_final_poses)
                final_scores_parts.append(piece_final_scores)
                final_labels_parts.append(piece_final_labels)
                final_bboxes_parts.append(piece_final_bboxes)

            if final_poses_parts:
                final_poses = torch.cat(final_poses_parts, dim=0)    # [N_total, 9, 2]
                final_scores = torch.cat(final_scores_parts, dim=0)  # [N_total, C] or [N_total]
                final_labels = torch.cat(final_labels_parts, dim=0)  # [N_total]
                final_bboxes = torch.cat(final_bboxes_parts, dim=0)  # [N_total, 4]
            else:
                final_poses = pred_pose_coords.new_zeros((0, NUM_BOARD_KEYPOINTS, 2))
                final_scores = pred_bboxes_conf.new_zeros((0,))
                final_labels = pred_cls_label.new_zeros((0,), dtype=torch.long)
                final_bboxes = pred_bboxes_xyxy.new_zeros((0, 4))

            decoded_predictions.append(
                ChessPoseEstimationPredictions(
                    poses=final_poses[: self.post_nms_max_predictions],
                    scores=final_scores[: self.post_nms_max_predictions],
                    labels=final_labels[: self.post_nms_max_predictions],
                    bboxes_xyxy=final_bboxes[: self.post_nms_max_predictions],
                )
            )

        return decoded_predictions
