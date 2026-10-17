import math
import random
from types import SimpleNamespace
from typing import Iterable

import cv2
import numpy as np

from super_gradients.common.registry.registry import register_transform
from .keypoints_random_perspective import KeypointsRandomPerspectiveTransform


@register_transform()
class KeypointsBoardSafeGeometry(KeypointsRandomPerspectiveTransform):
    """Rotate/shear and optionally distort a framed image, with bounded board clipping.

    Place after KeypointsBoardFramedCrop. Candidates are checked using coordinates only;
    the accepted affine/perspective matrix is rendered once by the parent transform.
    Each board retains at least ceil(min_retained_fraction * initially visible points),
    and at least min_visible_keypoints where the input contains that many. Pre-existing
    unannotated landmarks remain unannotated. If no candidate passes, use the identity.
    This clipping guard limits lost image context; the sample's annotation policy
    separately controls supervision of known landmarks outside the canvas.

    Rejection sampling deliberately biases rotations away from excessive clipping for
    the chosen framing. It does not shrink the board to accommodate those rotations.
    """

    def __init__(
        self,
        max_rotation=90.0,
        max_shear=5.0,
        distortion_scale=0.15,
        perspective_prob=0.5,
        min_retained_fraction=0.75,
        min_visible_keypoints=4,
        max_attempts=20,
        board_class_id=12,
        image_pad_value=0,
        mask_pad_value=1,
        interpolation_mode=cv2.INTER_LINEAR,
    ):
        if not 0 <= max_rotation <= 180 or not 0 <= max_shear < 45:
            raise ValueError("Rotation must be in [0, 180] and shear in [0, 45) degrees")
        if not 0 <= distortion_scale < 0.5 or not 0 <= perspective_prob <= 1:
            raise ValueError("Distortion must be in [0, 0.5) and perspective probability in [0, 1]")
        if not 0 < min_retained_fraction <= 1:
            raise ValueError("min_retained_fraction must be in (0, 1]")
        if not isinstance(max_attempts, int) or max_attempts < 1 or not isinstance(min_visible_keypoints, int) or min_visible_keypoints < 1:
            raise ValueError("max_attempts and min_visible_keypoints must be positive integers")
        super().__init__(
            distortion_scale=distortion_scale, image_pad_value=image_pad_value, mask_pad_value=mask_pad_value, interpolation_mode=interpolation_mode, prob=1.0
        )
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.perspective_prob = perspective_prob
        self.min_retained_fraction = min_retained_fraction
        self.min_visible_keypoints = min_visible_keypoints
        self.max_attempts = max_attempts
        self.board_class_id = board_class_id

    @staticmethod
    def _inside(points, width, height):
        return np.isfinite(points).all(axis=-1) & (points >= 0).all(axis=-1) & (points < [width, height]).all(axis=-1)

    def _sample_matrix(self, image, scales):
        height, width = image.shape[:2]
        cx, cy = (width - 1) / 2, (height - 1) / 2
        rotation = np.eye(3)
        rotation[:2] = cv2.getRotationMatrix2D((cx, cy), random.uniform(-self.max_rotation, self.max_rotation), 1.0)
        sx, sy = [math.tan(math.radians(random.uniform(-self.max_shear, self.max_shear))) for _ in range(2)]
        shear = np.array([[1, sx, -sx * cy], [sy, 1, -sy * cx], [0, 0, 1]])
        matrix = rotation @ shear
        if random.random() < self.perspective_prob:
            perspective, _, _ = super()._get_perspective_matrix(image, scales)
            matrix = perspective @ matrix
        return matrix

    def _get_sample_matrix(self, sample, distortion_scales):
        img = sample.image
        height, width = img.shape[:2]
        board_points = []
        if sample.labels is not None:
            for joints, label in zip(sample.joints, sample.labels):
                if label == self.board_class_id:
                    visible = (joints[:, 2] > 0) & self._inside(joints[:, :2], width, height)
                    if visible.any():
                        board_points.append(joints[visible, :2].astype(np.float32))
        corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        for _ in range(self.max_attempts):
            matrix = self._sample_matrix(img, distortion_scales)
            if all(self._retains_board(points, matrix, width, height) for points in board_points):
                return matrix, corners, cv2.perspectiveTransform(corners[None], matrix)[0]
        return np.eye(3), corners, corners.copy()

    def apply_to_sample(self, sample):
        """Warp as usual; after KeypointsBoardFramedCrop(defer_crop=True), also cut its window in the same warp.

        The geometry is sampled on the output canvas exactly as for an already-cut image, so annotations and random
        draws are unchanged. The image is resampled once instead of twice, and anything outside the window is padding,
        as if it had been cut first.
        """
        if sample.pending_crop is None:
            return super().apply_to_sample(sample)
        left, top, width, height = sample.pending_crop
        random.random()  # The parent's probability draw (prob is 1), so random sequences match the uncut path.
        canvas = SimpleNamespace(image=np.broadcast_to(np.uint8(0), (height, width)), joints=sample.joints, labels=sample.labels)
        matrix, _, corners = self._get_sample_matrix(canvas, self._distortion_scales())
        interpolation = random.choice(self.interpolation_mode)
        warp = matrix @ np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]])

        pad = tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else (self.image_pad_value,) * sample.image.shape[-1]
        image = cv2.warpPerspective(sample.image, warp, (width, height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=pad)
        mask = cv2.warpPerspective(sample.mask, warp, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.mask_pad_value)
        inside = np.zeros(sample.image.shape[:2], dtype=np.uint8)
        inside[max(round(top), 0) : round(top) + height, max(round(left), 0) : round(left) + width] = 1
        outside = cv2.warpPerspective(inside, warp, (width, height), flags=cv2.INTER_NEAREST) == 0
        image[outside] = pad
        mask[outside] = self.mask_pad_value

        sample.image, sample.mask, sample.pending_crop = image, mask, None
        self._apply_to_annotations(sample, matrix, width, height, corners)
        return sample.sanitize_sample()

    def _retains_board(self, points, matrix, width, height):
        projected = cv2.perspectiveTransform(points[None], matrix)[0]
        required = max(math.ceil(len(points) * self.min_retained_fraction), min(len(points), self.min_visible_keypoints))
        return self._inside(projected, width, height).sum() >= required
