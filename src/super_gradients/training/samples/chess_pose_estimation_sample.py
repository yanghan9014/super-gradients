import dataclasses

from super_gradients.training.samples.pose_estimation_sample import PoseEstimationSample

__all__ = ["ChessPoseEstimationSample"]


@dataclasses.dataclass
class ChessPoseEstimationSample(PoseEstimationSample):
    """
    A pose estimation sample for chess piece detection.

    Identical to PoseEstimationSample but exists as a distinct type so that
    dataset-specific collate functions and metrics can identify chess samples
    via ``isinstance`` checks when needed.
    """

    pass
