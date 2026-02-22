from typing import List, Tuple, Dict

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
    Performs confidence thresholding, Top-K and NMS for pieces,
    and selects the top-1 board detection with 9 structured keypoints.
    """

    def __init__(
        self,
        pose_confidence_threshold: float,
        nms_iou_threshold: float,
        pre_nms_max_predictions: int,
        post_nms_max_predictions: int,
    ):
        """
        :param pose_confidence_threshold: Pose detection confidence threshold
        :param nms_iou_threshold:         IoU threshold for NMS step.
        :param pre_nms_max_predictions:   Number of predictions participating in NMS step
        :param post_nms_max_predictions:  Maximum number of boxes to return after NMS step
        """
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
        """
        Take YoloNASPose's predictions and decode them into usable pose predictions.

        :param outputs: Output of the model's forward() method
        :return:        List of decoded predictions for each image in the batch.
        """
        # First element is model decoded predictions, second is raw logits for loss
        predictions = outputs[0]

        decoded_predictions: List[ChessPoseEstimationPredictions] = []
        for pred_values in zip(*predictions):
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores, pred_board_coords, pred_board_scores = pred_values
            # pred_bboxes_xyxy  [Anchors, 4] in XYXY format
            # pred_bboxes_conf  [Anchors, C] confidence scores [0..1]  (C=13: 12 pieces + 1 board)
            # pred_pose_coords  [Anchors, 1, 2] piece keypoint in (x,y)
            # pred_pose_scores  [Anchors, 1] piece keypoint confidence
            # pred_board_coords [Anchors, 9, 2] board keypoints in (x,y)
            # pred_board_scores [Anchors, 9] board keypoint confidences

            pred_cls_conf, pred_cls_label = torch.max(pred_bboxes_conf, dim=1)
            conf_mask = pred_cls_conf >= self.pose_confidence_threshold  # [Anchors]

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

            # ---- Piece predictions (same as before) ----
            piece_cls_conf = pred_cls_conf[piece_mask]
            piece_cls_label = pred_cls_label[piece_mask]
            piece_bboxes_conf = pred_bboxes_conf[piece_mask]
            piece_bboxes_xyxy = pred_bboxes_xyxy[piece_mask]
            piece_pose_coords = pred_pose_coords[piece_mask]

            piece_final_coords = piece_final_scores = piece_final_labels = piece_final_bboxes = None

            if piece_bboxes_conf.size(0) > 0:
                # Filter piece predictions by top-k
                if piece_bboxes_conf.size(0) > self.pre_nms_max_predictions:
                    topk_candidates = torch.topk(piece_cls_conf, k=self.pre_nms_max_predictions, dim=0, largest=True, sorted=True)
                    topk_idx = topk_candidates.indices
                    piece_cls_conf = piece_cls_conf[topk_idx].float()
                    piece_cls_label = piece_cls_label[topk_idx]
                    piece_bboxes_conf = piece_bboxes_conf[topk_idx]
                    piece_bboxes_xyxy = piece_bboxes_xyxy[topk_idx]
                    piece_pose_coords = piece_pose_coords[topk_idx]

                # NMS for piece predictions
                if self.class_agnostic_nms:
                    idx_to_keep = torchvision.ops.boxes.nms(piece_bboxes_xyxy.float(), piece_cls_conf.float(), iou_threshold=self.nms_iou_threshold)
                else:
                    idx_to_keep = torchvision.ops.boxes.batched_nms(
                        boxes=piece_bboxes_xyxy.float(), scores=piece_cls_conf.float(), idxs=piece_cls_label, iou_threshold=self.nms_iou_threshold
                    )

                piece_final_coords = piece_pose_coords[idx_to_keep]
                if piece_final_coords.dim() == 3:
                    piece_final_coords = piece_final_coords[:, 0, :]
                piece_final_scores = piece_bboxes_conf[idx_to_keep]
                piece_final_labels = piece_cls_label[idx_to_keep]
                piece_final_bboxes = piece_bboxes_xyxy[idx_to_keep]

            # ---- Board prediction (top-1 board detection with 9 structured keypoints) ----
            board_final_coords = board_final_scores = board_final_labels = board_final_bboxes = None

            board_cls_conf = pred_cls_conf[board_mask]
            board_bboxes_conf = pred_bboxes_conf[board_mask]
            board_bboxes_xyxy_filtered = pred_bboxes_xyxy[board_mask]
            board_kpts = pred_board_coords[board_mask]  # [N_board, 9, 2]
            board_kpt_scores = pred_board_scores[board_mask]  # [N_board, 9]

            if board_cls_conf.numel() > 0:
                # Take the single best board detection
                best_board_idx = torch.argmax(board_cls_conf)
                best_board_kpts = board_kpts[best_board_idx]  # [9, 2]
                best_board_kpt_scores = board_kpt_scores[best_board_idx]  # [9]
                best_board_bbox = board_bboxes_xyxy_filtered[best_board_idx]  # [4]
                best_board_conf = board_bboxes_conf[best_board_idx]  # [C]

                board_final_coords = best_board_kpts  # [9, 2]
                board_final_labels = torch.full((9,), BOARD_CLASS_ID, dtype=torch.long, device=pred_cls_label.device)
                board_final_bboxes = best_board_bbox.unsqueeze(0).expand(9, -1)  # [9, 4]
                board_final_scores = best_board_conf.unsqueeze(0).expand(9, -1)  # [9, C]

            # ---- Combine ----
            final_poses_parts = []
            final_scores_parts = []
            final_labels_parts = []
            final_bboxes_parts = []

            # Board predictions first so they survive truncation
            if board_final_coords is not None:
                final_poses_parts.append(board_final_coords)
                final_scores_parts.append(board_final_scores)
                final_labels_parts.append(board_final_labels)
                final_bboxes_parts.append(board_final_bboxes)

            if piece_final_coords is not None:
                final_poses_parts.append(piece_final_coords)
                final_scores_parts.append(piece_final_scores)
                final_labels_parts.append(piece_final_labels)
                final_bboxes_parts.append(piece_final_bboxes)

            if final_poses_parts:
                final_poses = torch.cat(final_poses_parts, dim=0)
                final_scores = torch.cat(final_scores_parts, dim=0)
                final_labels = torch.cat(final_labels_parts, dim=0)
                final_bboxes = torch.cat(final_bboxes_parts, dim=0)
            else:
                final_poses = pred_pose_coords.new_zeros((0, 2))
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
