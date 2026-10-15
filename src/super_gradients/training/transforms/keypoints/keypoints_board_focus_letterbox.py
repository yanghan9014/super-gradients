from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform

# The board outline a1 -> a8 -> h8 -> h1, as the app traces it for its setup check.
_OUTLINE = np.array([(0, 8), (0, 0), (8, 0), (8, 8)], dtype=np.float64)


def _project(points: np.ndarray, homography: np.ndarray) -> Optional[np.ndarray]:
    """Image points, or None if any lies at or beyond the board plane's horizon."""
    projected = np.column_stack([points, np.ones(len(points))]) @ homography.T
    centre_side = np.sign((np.array([4.0, 4.0, 1.0]) @ homography.T)[2])
    if centre_side == 0 or np.any(projected[:, 2] * centre_side <= 1e-9 * np.abs(projected).max()):
        return None
    result = projected[:, :2] / projected[:, 2:3]
    return result if np.isfinite(result).all() else None


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def _place(start: float, size: float, limit: int) -> Tuple[int, int]:
    """An integer [low, high) span of `size` at `start`, slid inside [0, limit); the whole axis if it doesn't fit."""
    if size >= limit:
        return 0, limit
    start = min(max(start, 0.0), limit - size)
    low, high = int(round(start)), int(round(start + size))
    return max(low, 0), min(max(high, low + 1), limit)


def board_focus_window(
    landmarks: Optional[np.ndarray],
    image_size: Tuple[int, int],
    output_size: Tuple[int, int] = (640, 480),
    min_board_fraction: float = 0.12,
    margin_squares: float = 2.0,
) -> Tuple[int, int, int, int]:
    """(left, top, right, bottom) of the source region to letterbox for validation.

    The whole image, unless the board would cover less than `min_board_fraction` of the letterboxed
    frame (the app's setup check: the a1-a8-h8-h1 outline's area). Then: the board plus `margin_squares`
    on every side, measured in board squares through the annotation homography, widened to the output
    aspect ratio around its centre and slid back inside the image. No padding is added for margin the
    image doesn't have; an axis the window can't fit within keeps the whole image. Deterministic.

    :param landmarks:  [9, 3] board landmarks (x, y, visibility) in source pixels: a1, a8, h1, h8, center, a45, h45, 1de, 8de.
    :param image_size: (width, height) of the source image.
    """
    from super_gradients.training.metrics.board_homography_metrics import BOARD_REFERENCE_POINTS

    width, height = image_size
    whole = (0, 0, width, height)
    if landmarks is None:
        return whole
    landmarks = np.asarray(landmarks, dtype=np.float64)
    usable = (landmarks[:, 2] > 0) & np.isfinite(landmarks[:, :2]).all(axis=1)
    if usable.sum() < 4:
        return whole
    to_board, _ = cv2.findHomography(landmarks[usable, :2], BOARD_REFERENCE_POINTS[usable], 0)
    if to_board is None or not np.isfinite(to_board).all():
        return whole
    try:
        to_image = np.linalg.inv(to_board)
    except np.linalg.LinAlgError:
        return whole
    outline = _project(_OUTLINE, to_image)
    if outline is None:
        return whole

    scale = min(output_size[0] / width, output_size[1] / height)
    if _polygon_area(outline) * scale**2 >= min_board_fraction * output_size[0] * output_size[1]:
        return whole

    m = margin_squares
    margin = _project(np.array([(-m, 8 + m), (-m, -m), (8 + m, -m), (8 + m, 8 + m)]), to_image)
    if margin is None:
        # Too near the horizon for a perspective margin: use the board's average square instead.
        pad = m * np.sqrt(_polygon_area(outline)) / 8
        (x0, y0), (x1, y1) = outline.min(axis=0) - pad, outline.max(axis=0) + pad
    else:
        (x0, y0), (x1, y1) = margin.min(axis=0), margin.max(axis=0)

    aspect = output_size[0] / output_size[1]
    window_width = max(x1 - x0, (y1 - y0) * aspect)
    window_height = window_width / aspect
    left, right = _place((x0 + x1 - window_width) / 2, window_width, width)
    top, bottom = _place((y0 + y1 - window_height) / 2, window_height, height)
    return left, top, right, bottom


def letterbox_image(
    image: np.ndarray, output_size: Tuple[int, int], pad_value: Union[int, Sequence[int]] = 0
) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
    """Fit the image inside output_size (width, height), centred on a padded canvas.

    Area averaging when shrinking, bilinear when enlarging. Returns the canvas and (scale_x, scale_y, left, top):
    a source point (x, y) lands at (x * scale_x + left, y * scale_y + top).
    """
    height, width = image.shape[:2]
    out_width, out_height = output_size
    scale = min(out_width / width, out_height / height)
    new_width, new_height = round(width * scale), round(height * scale)
    if (new_width, new_height) != (width, height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    left, top = (out_width - new_width) // 2, (out_height - new_height) // 2
    if image.ndim == 3 and not isinstance(pad_value, Iterable):
        pad_value = (pad_value,) * image.shape[2]
    canvas = cv2.copyMakeBorder(
        image, top, out_height - new_height - top, left, out_width - new_width - left, cv2.BORDER_CONSTANT, value=pad_value
    )
    return canvas, (new_width / width, new_height / height, left, top)


@register_transform()
class KeypointsBoardFocusLetterbox(AbstractKeypointTransform):
    """Deterministic validation framing: letterbox the whole photo, or crop around a board too small for the app.

    The app refuses to start until the board covers `min_board_fraction` of the frame, so photos with smaller
    boards are cropped to the board plus a margin (see board_focus_window) before letterboxing. Pieces outside
    the crop are dropped. Board landmarks keep their coordinates and remain if they land on the canvas.
    Must come first, on the source image: the window is chosen in source pixels.
    """

    def __init__(
        self,
        output_height: int = 480,
        output_width: int = 640,
        image_pad_value: Union[int, List[int]] = 0,
        mask_pad_value: float = 1,
        board_class_id: int = 12,
        min_board_fraction: float = 0.12,
        margin_squares: float = 2.0,
    ):
        super().__init__()
        self.output_size = (output_width, output_height)
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value
        self.board_class_id = board_class_id
        self.min_board_fraction = min_board_fraction
        self.margin_squares = margin_squares

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        height, width = sample.image.shape[:2]
        labels = None if sample.labels is None else np.asarray(sample.labels).reshape(-1)
        boards = [] if labels is None else np.flatnonzero(labels == self.board_class_id)
        landmarks = sample.joints[boards[0]] if len(boards) == 1 else None
        left, top, right, bottom = board_focus_window(landmarks, (width, height), self.output_size, self.min_board_fraction, self.margin_squares)

        if labels is not None and (left, top, right, bottom) != (0, 0, width, height):
            xy = sample.joints[:, 0, :2]
            inside = (xy >= [left, top]).all(axis=1) & (xy < [right, bottom]).all(axis=1)
            sample = sample.filter_by_mask((labels == self.board_class_id) | inside)

        sample.image, (scale_x, scale_y, pad_left, pad_top) = letterbox_image(
            sample.image[top:bottom, left:right], self.output_size, self.image_pad_value
        )
        mask, _ = letterbox_image(sample.mask[top:bottom, left:right], self.output_size, self.mask_pad_value)
        sample.mask = mask.astype(sample.mask.dtype, copy=False)

        joints = sample.joints.astype(np.float32, copy=True)
        joints[:, :, 0] = (joints[:, :, 0] - left) * scale_x + pad_left
        joints[:, :, 1] = (joints[:, :, 1] - top) * scale_y + pad_top
        sample.joints = joints
        if sample.bboxes_xywh is not None:
            boxes = sample.bboxes_xywh.astype(np.float32, copy=True)
            boxes[:, 0] = (boxes[:, 0] - left) * scale_x + pad_left
            boxes[:, 1] = (boxes[:, 1] - top) * scale_y + pad_top
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            sample.bboxes_xywh = boxes
        if sample.areas is not None:
            sample.areas = np.multiply(sample.areas, scale_x * scale_y, dtype=np.float32)
        return sample.sanitize_sample()

    def get_equivalent_preprocessing(self) -> List:
        """Inference has no annotations to crop by: the whole-image letterbox."""
        shape = (self.output_size[1], self.output_size[0])
        pad = list(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else [self.image_pad_value] * 3
        return [
            {Processings.KeypointsLongestMaxSizeRescale: {"output_shape": shape}},
            {Processings.KeypointsCenterPadding: {"output_shape": shape, "pad_value": pad}},
        ]
