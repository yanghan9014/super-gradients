from typing import List, Tuple

import torch
import torchvision
from torch import Tensor

from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from super_gradients.module_interfaces.pose_estimation_post_prediction_callback import ChessPoseEstimationPredictions

# Board is the last class (index 12 in a 13-class model: 12 pieces + 1 board)
BOARD_CLASS_ID = 12

# Ordered names of the 9 board keypoints (matches training data order)
BOARD_KEYPOINT_NAMES = ["a1", "a8", "h1", "h8", "center", "a45", "h45", "1de", "8de"]


class ChessYoloNASPosePostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """
    A post-prediction callback for the Chess YoloNASPose model.

    Outputs:
      - Pieces: poses=[N, 2], scores=[N, C], labels=[N] (labels 0-11)
      - Board:  board_keypoints=[9, 2], board_confidence=float (from top-1 detection)
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
            pred_board_coords = pred_board_coords[conf_mask].float()

            piece_mask = pred_cls_label != BOARD_CLASS_ID
            board_mask = pred_cls_label == BOARD_CLASS_ID

            # ---- Piece predictions ----
            piece_cls_conf = pred_cls_conf[piece_mask]
            piece_cls_label = pred_cls_label[piece_mask]
            piece_bboxes_conf = pred_bboxes_conf[piece_mask]
            piece_bboxes_xyxy = pred_bboxes_xyxy[piece_mask]
            piece_pose_coords = pred_pose_coords[piece_mask]  # [N_piece, 1, 2]

            if piece_bboxes_conf.size(0) > 0:
                # Top-K filtering
                if piece_bboxes_conf.size(0) > self.pre_nms_max_predictions:
                    topk_idx = torch.topk(piece_cls_conf, k=self.pre_nms_max_predictions, dim=0, largest=True, sorted=True).indices
                    piece_cls_conf = piece_cls_conf[topk_idx]
                    piece_cls_label = piece_cls_label[topk_idx]
                    piece_bboxes_conf = piece_bboxes_conf[topk_idx]
                    piece_bboxes_xyxy = piece_bboxes_xyxy[topk_idx]
                    piece_pose_coords = piece_pose_coords[topk_idx]

                # NMS
                if self.class_agnostic_nms:
                    idx_to_keep = torchvision.ops.boxes.nms(piece_bboxes_xyxy, piece_cls_conf, iou_threshold=self.nms_iou_threshold)
                else:
                    idx_to_keep = torchvision.ops.boxes.batched_nms(
                        boxes=piece_bboxes_xyxy, scores=piece_cls_conf, idxs=piece_cls_label, iou_threshold=self.nms_iou_threshold
                    )

                idx_to_keep = idx_to_keep[: self.post_nms_max_predictions]

                final_poses = piece_pose_coords[idx_to_keep, 0, :]  # [K, 2]
                final_scores = piece_bboxes_conf[idx_to_keep]
                final_labels = piece_cls_label[idx_to_keep]
                final_bboxes = piece_bboxes_xyxy[idx_to_keep]
            else:
                final_poses = pred_pose_coords.new_zeros((0, 2))
                final_scores = pred_bboxes_conf.new_zeros((0, pred_bboxes_conf.shape[-1] if pred_bboxes_conf.dim() > 1 else 13))
                final_labels = pred_cls_label.new_zeros((0,), dtype=torch.long)
                final_bboxes = pred_bboxes_xyxy.new_zeros((0, 4))

            # ---- Board prediction (top-1 → structured fields) ----
            board_cls_conf = pred_cls_conf[board_mask]
            board_kpts = pred_board_coords[board_mask]  # [N_board, 9, 2]

            board_keypoints = None
            board_confidence = None

            if board_cls_conf.numel() > 0:
                best_idx = torch.argmax(board_cls_conf)
                board_keypoints = board_kpts[best_idx]  # [9, 2]
                board_confidence = float(board_cls_conf[best_idx])

            decoded_predictions.append(
                ChessPoseEstimationPredictions(
                    poses=final_poses,
                    scores=final_scores,
                    labels=final_labels,
                    bboxes_xyxy=final_bboxes,
                    board_keypoints=board_keypoints,
                    board_confidence=board_confidence,
                )
            )

        return decoded_predictions
