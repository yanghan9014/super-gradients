from typing import Iterable

import cv2
import numpy as np

from super_gradients.common.object_names import Processings
from super_gradients.common.registry.registry import register_transform
from super_gradients.training.processing.processing import KeypointsCenterPadding, KeypointsLongestMaxSizeRescale
from .abstract_keypoints_transform import AbstractKeypointTransform


@register_transform()
class KeypointsLetterbox(AbstractKeypointTransform):
    """Fit the entire image, then center-pad on one axis. No annotation-dependent crop.

    Uses the inference processors directly so image pixels, rounding, and offsets match.
    Known off-image keypoints can enter the padded canvas; missing (v=0) points stay missing.
    With board annotation preservation enabled, known board points also remain supervised outside it.
    """

    def __init__(self, output_height=480, output_width=640, image_pad_value=0, mask_pad_value=1):
        super().__init__()
        if output_height <= 0 or output_width <= 0:
            raise ValueError("Output dimensions must be positive")
        self.output_height, self.output_width = output_height, output_width
        self.image_pad_value, self.mask_pad_value = image_pad_value, mask_pad_value
        self.resize = KeypointsLongestMaxSizeRescale((output_height, output_width))
        self.padding = KeypointsCenterPadding((output_height, output_width), image_pad_value)

    def apply_to_sample(self, sample):
        image, resize_meta = self.resize.preprocess_image(sample.image)
        height, width = image.shape[:2]
        sample.image, pad_meta = self.padding.preprocess_image(image)
        pad = pad_meta.padding_coordinates
        sample.mask = cv2.copyMakeBorder(
            cv2.resize(sample.mask, (width, height), interpolation=cv2.INTER_NEAREST),
            pad.top,
            pad.bottom,
            pad.left,
            pad.right,
            cv2.BORDER_CONSTANT,
            value=self.mask_pad_value,
        )
        scale = resize_meta.scale_factor_w
        sample.joints = sample.joints.astype(np.float32, copy=True)
        sample.joints[:, :, :2] *= scale
        sample.joints[:, :, :2] += [pad.left, pad.top]
        if sample.bboxes_xywh is not None:
            sample.bboxes_xywh = np.multiply(sample.bboxes_xywh, scale, dtype=np.float32)
            sample.bboxes_xywh[:, :2] += [pad.left, pad.top]
        if sample.areas is not None:
            sample.areas = np.multiply(sample.areas, scale**2, dtype=np.float32)
        return sample.sanitize_sample()

    def get_equivalent_preprocessing(self):
        pad = list(self.image_pad_value) if isinstance(self.image_pad_value, Iterable) else self.image_pad_value
        shape = (self.output_height, self.output_width)
        return [
            {Processings.KeypointsLongestMaxSizeRescale: {"output_shape": shape}},
            {Processings.KeypointsCenterPadding: {"output_shape": shape, "pad_value": pad}},
        ]
