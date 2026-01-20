from typing import List, Tuple, Dict

import math
import torch
import torchvision
from torch import Tensor

from super_gradients.module_interfaces import AbstractPoseEstimationPostPredictionCallback
from super_gradients.module_interfaces.pose_estimation_post_prediction_callback import ChessPoseEstimationPredictions

BOARD_CORNER_CLASS_IDS: Dict[str, int] = {
    "a1": 12,
    "a8": 13,
    "h1": 14,
    "h8": 15,
    "center": 16,
    "a45": 17,
    "h45": 18,
    "1de": 19,
    "8de": 20,
}

CLOCK_WISE_ORDER = ["a1", "a45", "a8", "8de", "h8", "h45", "h1", "1de"]
CLOCK_WISE_ORDER_CLASS_IDS = [BOARD_CORNER_CLASS_IDS[key] for key in CLOCK_WISE_ORDER]

CLOCK_WISE_CLASS_ID_TO_IDX = {cid: idx for idx, cid in enumerate(CLOCK_WISE_ORDER_CLASS_IDS)}
NUM_SLOTS = len(CLOCK_WISE_ORDER_CLASS_IDS)

# BOARD_CORNER_CLASS_IDS_INV: Dict[int, str] = {v: k for k, v in BOARD_CORNER_CLASS_IDS.items()}

class ChessYoloNASPosePostPredictionCallback(AbstractPoseEstimationPostPredictionCallback):
    """
    A post-prediction callback for YoloNASPose model.
    Performs confidence thresholding, Top-K and NMS steps.
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
    def __call__(self, outputs: Tuple[Tuple[Tensor, Tensor, Tensor, Tensor], ...]) -> List[ChessPoseEstimationPredictions]:
        """
        Take YoloNASPose's predictions and decode them into usable pose predictions.

        :param outputs: Output of the model's forward() method
        :return:        List of decoded predictions for each image in the batch.
        """
        # First is model predictions, second element of tuple is logits for loss computation
        predictions = outputs[0]

        decoded_predictions: List[ChessPoseEstimationPredictions] = []
        for pred_values in zip(*predictions):
            pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores = pred_values
            # pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores, pred_board_coords = pred_values
            # pred_bboxes [Anchors, 4] in XYXY format
            # pred_scores [Anchors, 1] confidence scores [0..1]
            # pred_pose_coords [Anchors, Num Keypoints, 2] in (x,y) format
            # pred_pose_scores [Anchors, Num Keypoints] confidence scores [0..1]

            pred_cls_conf, pred_cls_label = torch.max(pred_bboxes_conf, dim=1)
            conf_mask = pred_cls_conf >= self.pose_confidence_threshold  # [Anchors]

            pred_cls_conf = pred_cls_conf[conf_mask].float()
            pred_cls_label = pred_cls_label[conf_mask]
            pred_bboxes_conf = pred_bboxes_conf[conf_mask].float()
            pred_bboxes_xyxy = pred_bboxes_xyxy[conf_mask].float()
            pred_pose_coords = pred_pose_coords[conf_mask].float()
            pred_pose_scores = pred_pose_scores[conf_mask].float()

            piece_mask = pred_cls_label < 12
            board_mask = pred_cls_label >= 12

            piece_cls_conf = pred_cls_conf[piece_mask]
            piece_cls_label = pred_cls_label[piece_mask]
            piece_bboxes_conf = pred_bboxes_conf[piece_mask]
            piece_bboxes_xyxy = pred_bboxes_xyxy[piece_mask]
            piece_pose_coords = pred_pose_coords[piece_mask]
            piece_pose_scores = pred_pose_scores[piece_mask]

            board_cls_conf = pred_cls_conf[board_mask]
            board_cls_label = pred_cls_label[board_mask]
            board_bboxes_conf = pred_bboxes_conf[board_mask]
            board_bboxes_xyxy = pred_bboxes_xyxy[board_mask]
            board_pose_coords = pred_pose_coords[board_mask]
            if board_pose_coords.dim() == 3:
                board_pose_coords = board_pose_coords[:, 0, :]

            piece_final_coords = piece_final_scores = piece_final_labels = piece_final_bboxes = None
            board_final_coords = board_final_scores = board_final_labels = board_final_bboxes = None

            if piece_bboxes_conf.size(0) > 0:
                # Filter piece predictions by self.nms_top_k
                if piece_bboxes_conf.size(0) > self.pre_nms_max_predictions:
                    topk_candidates = torch.topk(piece_cls_conf, k=self.pre_nms_max_predictions, dim=0, largest=True, sorted=True)
                    topk_idx = topk_candidates.indices
                    piece_cls_conf = piece_cls_conf[topk_idx].float()
                    piece_cls_label = piece_cls_label[topk_idx]
                    piece_bboxes_conf = piece_bboxes_conf[topk_idx]
                    piece_bboxes_xyxy = piece_bboxes_xyxy[topk_idx]
                    piece_pose_coords = piece_pose_coords[topk_idx]
                    piece_pose_scores = piece_pose_scores[topk_idx]

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

            center_mask = board_cls_label == BOARD_CORNER_CLASS_IDS['center']
            use_robust_post_processing = False
            if use_robust_post_processing and board_cls_conf.numel() > 0 and sum(center_mask) > 0 and sum(~center_mask) > 0:
                center_id = torch.argmax(center_mask * board_cls_conf)
                center_xy = board_pose_coords[center_id].squeeze(0)

                non_center_coords = board_pose_coords[~center_mask].squeeze(1)
                non_center_bboxes = board_bboxes_xyxy[~center_mask]
                non_center_max_scores = board_cls_conf[~center_mask]
                non_center_scores = board_bboxes_conf[~center_mask]
                non_center_labels = board_cls_label[~center_mask]
                
                angle_to_center = torch.atan2(non_center_coords[:, 1] - center_xy[1], non_center_coords[:, 0] - center_xy[0])
                polar_order = torch.argsort(angle_to_center)
                coords_sorted = non_center_coords[polar_order]
                bboxes_sorted = non_center_bboxes[polar_order]
                max_scores_sorted = non_center_max_scores[polar_order]
                scores_sorted = non_center_scores[polar_order]
                labels_sorted = non_center_labels[polar_order]
                # print([BOARD_CORNER_CLASS_IDS_INV[labels_sorted[i].item()] for i in range(len(labels_sorted))])
                
                slot_idx = [CLOCK_WISE_CLASS_ID_TO_IDX[l.item()] for l in labels_sorted]
                log_scores = [s.item() for s in torch.log(max_scores_sorted.clamp_min(1e-6))]       # [N]
                N = len(labels_sorted)

                eps = 0.05
                log_eps = math.log(eps)

                neg_inf = -1e30
                max_missing_in_jump = 3
                dp = [neg_inf] * N
                dp[0] = log_scores[0]
                last_slot_idx = {}
                chosen_idx = [{k: None for k in range(NUM_SLOTS)} for _ in range(N)]
                for i in range(N):
                    updated = 0
                    s = slot_idx[i]
                    log_c = log_scores[i]
                    if s in last_slot_idx:
                        last_i = last_slot_idx[s]
                        diff = log_c - log_scores[last_i]
                        dp[i] = dp[last_i]
                        chosen_idx[i] = chosen_idx[last_i].copy()
                        if diff > 0:
                            dp[i] += diff
                            chosen_idx[i][s] = i
                            last_slot_idx[s] = i
                        updated = 1
                    
                    for step in range(1, max_missing_in_jump + 1):
                        s_prev = (NUM_SLOTS + s - step) % NUM_SLOTS
                        if s_prev in last_slot_idx:
                            last_j = last_slot_idx[s_prev]
                            delta = step - 1                  # how many missing slots between s_prev and s
                            new_val = dp[last_j] + log_c + delta * log_eps
                            if new_val > dp[i]:
                                dp[i] = new_val
                                chosen_idx[i] = chosen_idx[last_j].copy()
                                chosen_idx[i][s] = i
                                last_slot_idx[s] = i
                                updated = 1
                                break
                    if not updated:
                        last_slot_idx[s] = i
                        dp[i] = log_c
                        chosen_idx[i][s] = i
                cumulative_log_scores = torch.full((N,), neg_inf, device=scores_sorted.device, dtype=scores_sorted.dtype)
                for i in range(N):
                    cumulative_log_scores[i] = sum([(log_scores[v] if v is not None else log_eps) for v in chosen_idx[i].values()])
                
                chosen_indexes = [idx for idx in chosen_idx[torch.argmax(cumulative_log_scores)].values() if idx is not None]
                if chosen_indexes:
                    chosen_indexes = torch.tensor(chosen_indexes, device=coords_sorted.device, dtype=torch.long)
                    board_final_coords = coords_sorted[chosen_indexes]
                    board_final_bboxes = bboxes_sorted[chosen_indexes]
                    board_final_scores = scores_sorted[chosen_indexes]
                    board_final_labels = labels_sorted[chosen_indexes]
                    center_coord = center_xy.unsqueeze(0)
                    center_bbox = board_bboxes_xyxy[center_id].unsqueeze(0)
                    center_score = board_bboxes_conf[center_id].unsqueeze(0)
                    center_label = board_cls_label[center_id].unsqueeze(0)
                    board_final_coords = torch.cat([board_final_coords, center_coord], dim=0)
                    board_final_bboxes = torch.cat([board_final_bboxes, center_bbox], dim=0)
                    board_final_scores = torch.cat([board_final_scores, center_score], dim=0)
                    board_final_labels = torch.cat([board_final_labels, center_label], dim=0)
            else:
                unique_board_labels = torch.unique(board_cls_label)
                if unique_board_labels.numel() > 0:
                    keep_indices = []
                    for lbl in unique_board_labels:
                        label_indices = torch.nonzero(board_cls_label == lbl, as_tuple=True)[0]
                        best_local_idx = torch.argmax(board_cls_conf[label_indices])
                        keep_indices.append(label_indices[best_local_idx])

                    keep_tensor = torch.stack(keep_indices)
                    board_final_coords = board_pose_coords[keep_tensor]
                    board_final_bboxes = board_bboxes_xyxy[keep_tensor]
                    board_final_scores = board_bboxes_conf[keep_tensor]
                    board_final_labels = board_cls_label[keep_tensor]

            final_poses_parts = []
            final_scores_parts = []
            final_labels_parts = []
            final_bboxes_parts = []

            # Keep board predictions first so they survive truncation if needed.
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
