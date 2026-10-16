import random
from typing import Iterable, List, Tuple, Union

import cv2
import numpy as np

from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsBoardFramedCrop(AbstractKeypointTransform):
    """Frame the board in a 4:3 window the way the iOS app frames it, then scale to the output size.

    The app centre-crops a fixed 4:3 window from the camera and scales it to 640x480; players
    frame the board inside that window. Here the window starts as the largest 4:3 window the
    image allows, placed around the annotated board. It is enlarged only as far as needed for
    the board to span at most `max_board_fraction` of its width and height, so the board is
    always whole with room around it; where the window extends past the image it is padded.

    Deterministic mode (validation) centres the window on the board, moved inside the image where
    the margin allows. Random mode (training) picks a random board fraction between
    `min_board_fraction` and `max_board_fraction` (never zooming in beyond the largest window)
    and a random position that keeps the board and its margin inside, so padding can land
    on any side. Images without a visible board get the app's centred crop.

    With `defer_crop`, the photo is only area-shrunk to the output scale and the window is recorded in
    `sample.pending_crop`; annotations are already in window coordinates. KeypointsBoardSafeGeometry then
    cuts it within its own warp, so a sharp full-resolution photo is resampled once rather than twice.
    It must follow before any other geometric transform.
    """

    def __init__(
        self,
        output_height: int = 480,
        output_width: int = 640,
        board_class_id: int = 12,
        max_board_fraction: float = 0.85,
        min_board_fraction: float = 0.5,
        randomize: bool = False,
        image_pad_value: Union[int, Iterable[int]] = 0,
        mask_pad_value: float = 1,
        interpolation: int = cv2.INTER_LINEAR,
        preserve_out_of_frame_board_annotations: bool = False,
        defer_crop: bool = False,
    ):
        super().__init__()
        if not 0 < min_board_fraction <= max_board_fraction <= 1:
            raise ValueError("Board fractions must satisfy 0 < min_board_fraction <= max_board_fraction <= 1")
        self.output_height = output_height
        self.output_width = output_width
        self.board_class_id = board_class_id
        self.max_board_fraction = max_board_fraction
        self.min_board_fraction = min_board_fraction
        self.randomize = randomize
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value
        self.interpolation = interpolation
        self.preserve_out_of_frame_board_annotations = preserve_out_of_frame_board_annotations
        self.defer_crop = defer_crop

    def board_extent(self, sample: PoseEstimationSample):
        """(x0, y0, x1, y1) of the visible board landmarks, clipped to the image; None without a board."""
        height, width = sample.image.shape[:2]
        if sample.labels is None:
            return None
        points = [
            joints[(joints[:, 2] > 0) & (joints[:, :2] >= 0).all(axis=1) & (joints[:, :2] < [width, height]).all(axis=1), :2]
            for joints, label in zip(sample.joints, sample.labels)
            if label == self.board_class_id
        ]
        points = np.concatenate(points) if points else np.zeros((0, 2))
        if len(points) == 0:
            return None
        x0, y0 = np.clip(points.min(axis=0), 0, (width, height))
        x1, y1 = np.clip(points.max(axis=0), 0, (width, height))
        return x0, y0, x1, y1

    def window(self, sample: PoseEstimationSample) -> Tuple[float, float, float]:
        """(left, top, width) of the 4:3 window in image pixels; its height is width * 3 / 4."""
        height, width = sample.image.shape[:2]
        aspect = self.output_width / self.output_height
        base_width = min(width, height * aspect)
        extent = self.board_extent(sample)
        if extent is None:
            return (width - base_width) / 2, (height - base_width / aspect) / 2, base_width
        x0, y0, x1, y1 = extent
        board_width, board_height = x1 - x0, y1 - y0
        # Window width at which the board spans exactly `fraction` of the limiting axis.
        needed = max(board_width, board_height * aspect)
        fraction = random.uniform(self.min_board_fraction, self.max_board_fraction) if self.randomize else self.max_board_fraction
        window_width = max(base_width, needed / fraction)
        window_height = window_width / aspect
        margin_x = window_width * (1 - self.max_board_fraction) / 2
        margin_y = window_height * (1 - self.max_board_fraction) / 2
        # Positions that keep the board and its margin inside the window.
        left_range = (x1 + margin_x - window_width, x0 - margin_x)
        top_range = (y1 + margin_y - window_height, y0 - margin_y)
        centre = None if self.randomize else ((x0 + x1 - window_width) / 2, (y0 + y1 - window_height) / 2)
        left = self._position(left_range, width, window_width, None if centre is None else centre[0])
        top = self._position(top_range, height, window_height, None if centre is None else centre[1])
        return left, top, window_width

    @staticmethod
    def _position(allowed, image_size, window_size, target=None):
        """A position from `allowed` (board and margin inside): random, or the closest to `target`.

        Where the window fits in the image and the margin allows, it stays inside, like the app's
        crop, so there is no needless padding; otherwise padding goes wherever the position puts it.
        """
        low, high = allowed
        if window_size <= image_size:
            inside = max(low, 0.0), min(high, image_size - window_size)
            if inside[0] <= inside[1]:
                low, high = inside
        if target is None:
            return random.uniform(low, high)
        return float(np.clip(target, low, high))

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        left, top, window_width = self.window(sample)
        scale = self.output_width / window_width
        if self.defer_crop:
            # One area-averaged shrink of the whole photo to the output scale; the cut comes later.
            height, width = sample.image.shape[:2]
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            sample.image = cv2.resize(sample.image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            sample.mask = cv2.resize(sample.mask, size, interpolation=cv2.INTER_NEAREST)
            scale_x, scale_y = size[0] / width, size[1] / height
            sample.pending_crop = (left * scale_x, top * scale_y, self.output_width, self.output_height)
        else:
            scale_x = scale_y = scale
            matrix = np.array([[scale, 0, -left * scale], [0, scale, -top * scale]], dtype=np.float32)
            size = (self.output_width, self.output_height)
            pad = tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else (self.image_pad_value,) * sample.image.shape[2]
            sample.image = cv2.warpAffine(sample.image, matrix, size, flags=self.interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=pad)
            dtype = sample.mask.dtype
            sample.mask = cv2.warpAffine(
                sample.mask.astype(np.uint8), matrix, size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.mask_pad_value
            ).astype(dtype)

        # float32 like KeypointsLongestMaxSize, which this replaces; the loss writes these into float32 tensors.
        joints = sample.joints.astype(np.float32, copy=True)
        joints[:, :, 0] = (joints[:, :, 0] - left) * scale_x
        joints[:, :, 1] = (joints[:, :, 1] - top) * scale_y
        outside = (joints[:, :, 0] < 0) | (joints[:, :, 1] < 0) | (joints[:, :, 0] >= self.output_width) | (joints[:, :, 1] >= self.output_height)
        if self.preserve_out_of_frame_board_annotations and sample.labels is not None:
            # Geometry can bring known board points into the final canvas. It performs
            # the final boundary check; piece annotations still require source evidence.
            outside &= (np.asarray(sample.labels).reshape(-1) != self.board_class_id)[:, None]
        joints[outside, 2] = 0
        sample.joints = joints
        if sample.bboxes_xywh is not None:
            boxes = sample.bboxes_xywh.astype(np.float32, copy=True)
            boxes[:, 0] = (boxes[:, 0] - left) * scale_x
            boxes[:, 1] = (boxes[:, 1] - top) * scale_y
            boxes[:, 2] *= scale_x
            boxes[:, 3] *= scale_y
            sample.bboxes_xywh = boxes
        if sample.areas is not None:
            sample.areas = np.multiply(sample.areas, scale_x * scale_y, dtype=np.float32)
        return sample

    def get_equivalent_preprocessing(self) -> List:
        """Inference has no annotations to frame by; app frames already arrive as the 640x480 crop.

        For such a frame this transform is the identity, as is resizing to fit and padding, which
        also keeps any other input the right size.
        """
        shape = (self.output_height, self.output_width)
        pad = list(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else [self.image_pad_value] * 3
        return [
            {Processings.KeypointsLongestMaxSizeRescale: {"output_shape": shape}},
            {Processings.KeypointsCenterPadding: {"output_shape": shape, "pad_value": pad}},
        ]
