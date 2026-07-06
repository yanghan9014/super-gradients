import random
from typing import List, Union, Iterable, Tuple, Optional

import cv2
import numpy as np

from super_gradients.common.object_names import Transforms
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xywh_to_xyxy, xyxy_to_xywh
from super_gradients.training.samples import PoseEstimationSample
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform(Transforms.KeypointsRandomPerspectiveTransform)
class KeypointsRandomPerspectiveTransform(AbstractKeypointTransform):
    """
    Apply random perspective transform to image, mask, and joints.
    """

    def __init__(
        self,
        distortion_scale: Union[float, Tuple[float, float], List[Union[float, Tuple[float, float]]]] = 0.2,
        image_pad_value: Union[int, float, List[int]] = 0,
        mask_pad_value: float = 1,
        interpolation_mode: Union[int, List[int]] = cv2.INTER_LINEAR,
        prob: float = 0.5,
    ):
        """
        :param distortion_scale:   Scale of perspective deviation. Can be:
                                     - A single float (applied as max fraction of size to shift all corners)
                                     - A tuple (sx, sy) representing max shift fraction for x and y axes for all corners
                                     - A list of 4 floats/tuples representing individual max shift for each corner:
                                       [top-left, top-right, bottom-right, bottom-left]
        :param image_pad_value:    Value to pad the image during perspective transform. Can be single scalar or list.
        :param mask_pad_value:     Value to pad the mask during perspective transform.
        :param interpolation_mode: A constant integer or list of integers, specifying the interpolation mode to use.
        :param prob:               Probability to apply the transform.
        """
        super().__init__()

        self.distortion_scale = distortion_scale
        self.image_pad_value = image_pad_value
        self.mask_pad_value = mask_pad_value
        self.prob = prob
        self.interpolation_mode = tuple(interpolation_mode) if isinstance(interpolation_mode, Iterable) else (interpolation_mode,)

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(distortion_scale={self.distortion_scale}, "
            f"image_pad_value={self.image_pad_value}, "
            f"mask_pad_value={self.mask_pad_value}, "
            f"prob={self.prob})"
        )

    def _get_perspective_matrix(self, img: np.ndarray, distortion_scales: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute a random perspective transform matrix.
        """
        height, width = img.shape[:2]

        # Source points are the 4 corners of the image
        src_pts = np.float32([
            [0, 0],                  # top-left
            [width - 1, 0],          # top-right
            [width - 1, height - 1],  # bottom-right
            [0, height - 1]          # bottom-left
        ])

        # Calculate random shifts for each corner
        # Corner 0: top-left (shift can go right and down)
        # Corner 1: top-right (shift can go left and down)
        # Corner 2: bottom-right (shift can go left and up)
        # Corner 3: bottom-left (shift can go right and up)
        dst_pts = src_pts.copy()

        # top-left
        dx0 = random.uniform(0, distortion_scales[0][0] * width)
        dy0 = random.uniform(0, distortion_scales[0][1] * height)
        dst_pts[0] += [dx0, dy0]

        # top-right
        dx1 = random.uniform(-distortion_scales[1][0] * width, 0)
        dy1 = random.uniform(0, distortion_scales[1][1] * height)
        dst_pts[1] += [dx1, dy1]

        # bottom-right
        dx2 = random.uniform(-distortion_scales[2][0] * width, 0)
        dy2 = random.uniform(-distortion_scales[2][1] * height, 0)
        dst_pts[2] += [dx2, dy2]

        # bottom-left
        dx3 = random.uniform(0, distortion_scales[3][0] * width)
        dy3 = random.uniform(-distortion_scales[3][1] * height, 0)
        dst_pts[3] += [dx3, dy3]

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix, src_pts, dst_pts

    def apply_to_sample(self, sample: PoseEstimationSample) -> PoseEstimationSample:
        """
        Apply perspective transform to given pose estimation sample.
        """
        if random.random() < self.prob:
            height, width = sample.image.shape[:2]

            # Parse distortion scales into 4 corner shifts
            scales = []
            if isinstance(self.distortion_scale, (int, float)):
                scales = [(float(self.distortion_scale), float(self.distortion_scale))] * 4
            elif isinstance(self.distortion_scale, tuple) and len(self.distortion_scale) == 2:
                scales = [(float(self.distortion_scale[0]), float(self.distortion_scale[1]))] * 4
            elif isinstance(self.distortion_scale, (list, tuple)) and len(self.distortion_scale) == 4:
                for s in self.distortion_scale:
                    if isinstance(s, (int, float)):
                        scales.append((float(s), float(s)))
                    elif isinstance(s, tuple) and len(s) == 2:
                        scales.append((float(s[0]), float(s[1])))
                    else:
                        raise ValueError(f"Invalid distortion_scale format: {self.distortion_scale}")
            else:
                raise ValueError(f"Invalid distortion_scale format: {self.distortion_scale}")

            M, src_pts, dst_pts = self._get_perspective_matrix(sample.image, scales)
            interpolation = random.choice(self.interpolation_mode)

            image_pad_value = (
                tuple(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else tuple([self.image_pad_value] * sample.image.shape[-1])
            )

            # Apply to image
            sample.image = cv2.warpPerspective(
                sample.image,
                M,
                dsize=(width, height),
                flags=interpolation,
                borderValue=image_pad_value,
                borderMode=cv2.BORDER_CONSTANT
            )

            # Apply to mask
            sample.mask = cv2.warpPerspective(
                sample.mask,
                M,
                dsize=(width, height),
                flags=cv2.INTER_NEAREST,
                borderValue=self.mask_pad_value,
                borderMode=cv2.BORDER_CONSTANT
            )

            # Apply to joints
            # joints shape: [N, K, 3] (where last dim is x, y, visibility)
            if sample.joints is not None and len(sample.joints) > 0:
                keypoints_with_visibility = sample.joints.copy()
                xy = keypoints_with_visibility[:, :, 0:2].reshape(-1, 1, 2).astype(np.float32)
                transformed_xy = cv2.perspectiveTransform(xy, M)
                transformed_xy = transformed_xy.reshape(keypoints_with_visibility.shape[0], keypoints_with_visibility.shape[1], 2)
                keypoints_with_visibility[:, :, 0:2] = transformed_xy

                # Update visibility status of joints that were moved outside visible area
                outside_left = keypoints_with_visibility[:, :, 0] < 0
                outside_top = keypoints_with_visibility[:, :, 1] < 0
                outside_right = keypoints_with_visibility[:, :, 0] >= width
                outside_bottom = keypoints_with_visibility[:, :, 1] >= height
                joints_outside_image = outside_left | outside_top | outside_right | outside_bottom
                keypoints_with_visibility[joints_outside_image, 2] = 0

                sample.joints = keypoints_with_visibility.astype(sample.joints.dtype, copy=False)

            # Apply to bboxes
            # bboxes_xywh shape: [N, 4]
            if sample.bboxes_xywh is not None and len(sample.bboxes_xywh) > 0:
                bboxes_xyxy = xywh_to_xyxy(sample.bboxes_xywh, image_shape=None)
                new_bboxes_xyxy = []
                for bbox in bboxes_xyxy:
                    x_min, y_min, x_max, y_max = bbox[:4]
                    corners = np.array([
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ], dtype=np.float32).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(corners, M).reshape(-1, 2)
                    x_coords = transformed_corners[:, 0]
                    y_coords = transformed_corners[:, 1]
                    new_bboxes_xyxy.append([min(x_coords), min(y_coords), max(x_coords), max(y_coords)])

                new_bboxes_xyxy = np.array(new_bboxes_xyxy, dtype=sample.bboxes_xywh.dtype)

                # Update areas if present
                if sample.areas is not None:
                    old_bbox_area = sample.bboxes_xywh[:, 2] * sample.bboxes_xywh[:, 3]
                    new_bbox_xywh = xyxy_to_xywh(new_bboxes_xyxy, image_shape=None)
                    new_bbox_area = new_bbox_xywh[:, 2] * new_bbox_xywh[:, 3]
                    scale_factor = new_bbox_area / np.clip(old_bbox_area, 1e-6, None)
                    sample.areas = (sample.areas * scale_factor).astype(sample.areas.dtype)

                sample.bboxes_xywh = xyxy_to_xywh(new_bboxes_xyxy, image_shape=None).astype(sample.bboxes_xywh.dtype)

            elif sample.areas is not None:
                # Fallback area calculation based on image-level distortion
                def polygon_area(pts):
                    x = pts[:, 0]
                    y = pts[:, 1]
                    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                old_img_area = width * height
                new_img_area = polygon_area(dst_pts)
                scale_factor = new_img_area / old_img_area
                sample.areas = (sample.areas * scale_factor).astype(sample.areas.dtype)

            sample = sample.sanitize_sample()

        return sample

    def get_equivalent_preprocessing(self):
        raise RuntimeError(f"{self.__class__.__name__} does not have equivalent preprocessing because it is non-deterministic.")
