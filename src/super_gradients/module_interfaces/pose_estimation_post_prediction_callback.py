import abc
import dataclasses
import numpy as np

from typing import Any, List
from typing import Union, Optional
from torch import Tensor

__all__ = ["PoseEstimationPredictions", "ChessPoseEstimationPredictions", "AbstractPoseEstimationPostPredictionCallback"]


@dataclasses.dataclass
class PoseEstimationPredictions:
    """
    A data class that encapsulates pose estimation predictions for a single image.

    :param poses:        Array of shape [N, K, 3] where N is number of poses and K is number of joints.
                         Last dimension is [x, y, score] where score the confidence score for the specific joint
                         with [0..1] range.
    :param scores:       Array of shape [N, J] with scores for each pose with [0..1] range.
    :param bboxes_xyxy:  Array of shape [N, 4] with bounding boxes for each pose in XYXY format.
                         Can be None if bounding boxes are not available (for instance, DEKR model does not output boxes).
    """

    poses: Union[Tensor, np.ndarray]
    scores: Union[Tensor, np.ndarray]
    bboxes_xyxy: Optional[Union[Tensor, np.ndarray]]

@dataclasses.dataclass
class ChessPoseEstimationPredictions:
    """
    A data class that encapsulates pose estimation predictions for a single image.

    :param poses:        Array of shape [N, Num Joints, 2] with (x, y) coordinates per keypoint.
                         Pieces use kp[0] only; board detections use kp[0..8].
    :param pose_scores:  Array of shape [N, Num Joints] with per-keypoint confidence [0..1].
    :param scores:       Array of shape [N, Num Classes] with class scores for each detection.
    :param labels:       Array of shape [N] with class labels (0-11: pieces, 12: board).
    :param bboxes_xyxy:  Optional bounding boxes. It is None for the box-free chess model.
    :param class_probabilities: Optional conditional softmax class probabilities [N, Num Classes].
    :param objectness_scores: Optional scalar foreground probabilities [N, 1].
    """

    poses: Union[Tensor, np.ndarray]
    pose_scores: Union[Tensor, np.ndarray]
    scores: Union[Tensor, np.ndarray]
    labels: Union[Tensor, np.ndarray]
    bboxes_xyxy: Optional[Union[Tensor, np.ndarray]]
    class_probabilities: Optional[Union[Tensor, np.ndarray]] = None
    objectness_scores: Optional[Union[Tensor, np.ndarray]] = None


class AbstractPoseEstimationPostPredictionCallback(abc.ABC):
    """
    A protocol interface of a post-prediction callback for pose estimation models.
    """

    @abc.abstractmethod
    def __call__(self, predictions: Any) -> List[ChessPoseEstimationPredictions]:
        ...
