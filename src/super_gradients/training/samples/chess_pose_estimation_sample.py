import dataclasses
from typing import Optional

import numpy as np

from super_gradients.training.samples.pose_estimation_sample import PoseEstimationSample

__all__ = ["ChessPoseEstimationSample"]


@dataclasses.dataclass
class ChessPoseEstimationSample(PoseEstimationSample):
    """
    A pose estimation sample for chess piece detection.

    Carries original board piece counts through augmentation, and provides a
    distinct type for dataset-specific collate functions and metrics.
    """

    # Original annotation-space count, per instance; -1 means unknown/not a board.
    # Geometry and piece filtering must never change a board's eligibility.
    board_piece_counts: Optional[np.ndarray] = None

    def filter_by_mask(self, mask):
        if self.board_piece_counts is not None:
            self.board_piece_counts = self.board_piece_counts[mask]
        return super().filter_by_mask(mask)
