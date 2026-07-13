from typing import Tuple, Union, List, Callable, Optional

import torch
from omegaconf import DictConfig
from torch import nn, Tensor

import super_gradients.common.factories.detection_modules_factory as det_factory
from super_gradients.common.registry import register_detection_module
from super_gradients.module_interfaces import SupportsReplaceNumClasses
from super_gradients.modules.base_modules import BaseDetectionModule
from super_gradients.training.utils import HpmStruct, torch_version_is_greater_or_equal
from super_gradients.training.utils.utils import infer_model_dtype, infer_model_device

# Declare type aliases for better readability
# We cannot use typing.TypeAlias since it is not supported in python 3.7
YoloNasPoseDecodedPredictions = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
YoloNasPoseRawOutputs = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[int], Tensor]


@register_detection_module()
class ChessYoloNASPoseNDFLHeads(BaseDetectionModule, SupportsReplaceNumClasses):
    def __init__(
        self,
        num_classes: int,
        in_channels: Tuple[int, int, int],
        heads_list: List[Union[HpmStruct, DictConfig]],
        grid_cell_scale: float = 5.0,
        grid_cell_offset: float = 0.5,
        reg_max: int = 16,
        inference_mode: bool = False,
        eval_size: Optional[Tuple[int, int]] = None,
        width_mult: float = 1.0,
        pose_offset_multiplier: float = 1.0,
        compensate_grid_cell_offset: bool = True,
    ):
        """
        Initializes the NDFLHeads module.

        :param num_classes: Number of detection classes
        :param in_channels: Number of channels for each feature map (See width_mult)
        :param grid_cell_scale: A scaling factor applied to the grid cell coordinates.
               This scaling factor is used to define anchor boxes (see generate_anchors_for_grid_cell).
        :param grid_cell_offset: A fixed offset that is added to the grid cell coordinates.
               This offset represents a 'center' of the cell and is 0.5 by default.
        :param reg_max: Number of bins in the regression head
        :param eval_size: (rows, cols) Size of the image for evaluation. Setting this value can be beneficial for inference speed,
               since anchors will not be regenerated for each forward call.
        :param width_mult: A scaling factor applied to in_channels.
        :param pose_offset_multiplier: A scaling factor applied to the pose regression offset. This multiplier is
               meant to reduce absolute magnitude of weights in pose regression layers.
               Default value is 1.0.
        :param compensate_grid_cell_offset: (bool) Controls whether to subtract anchor cell offset from the pose regression.
               If True, predicted pose coordinates decoded as (offsets + anchors - grid_cell_offset) * stride.
               If False, predicted pose coordinates decoded as (offsets + anchors) * stride.
               Default value is True.

        """
        in_channels = [max(round(c * width_mult), 1) for c in in_channels]
        super().__init__(in_channels)

        self.in_channels = tuple(in_channels)
        self.num_classes = num_classes
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max  # Unused legacy setting kept for config compatibility.
        self.eval_size = eval_size
        self.pose_offset_multiplier = pose_offset_multiplier
        self.compensate_grid_cell_offset = compensate_grid_cell_offset
        self.inference_mode = inference_mode

        self._init_weights()

        factory = det_factory.DetectionModulesFactory()
        heads_list = self._insert_heads_list_params(heads_list, factory, num_classes, reg_max)

        self.num_heads = len(heads_list)
        fpn_strides: List[int] = []
        for i in range(self.num_heads):
            new_head = factory.get(factory.insert_module_param(heads_list[i], "in_channels", in_channels[i]))
            fpn_strides.append(new_head.stride)
            setattr(self, f"head{i + 1}", new_head)

        self.fpn_strides = tuple(fpn_strides)

    def replace_num_classes(self, num_classes: int, compute_new_weights_fn: Callable[[nn.Module, int], nn.Module]):
        for i in range(self.num_heads):
            head = getattr(self, f"head{i + 1}")
            head.replace_num_classes(num_classes, compute_new_weights_fn)

        self.num_classes = num_classes

    @staticmethod
    def _insert_heads_list_params(
        heads_list: List[Union[HpmStruct, DictConfig]], factory: det_factory.DetectionModulesFactory, num_classes: int, reg_max: int
    ) -> List[Union[HpmStruct, DictConfig]]:
        """
        Injects num_classes and reg_max parameters into the heads_list.

        :param heads_list:  Input heads list
        :param factory:     DetectionModulesFactory
        :param num_classes: Number of classes
        :param reg_max:     Number of bins in the regression head
        :return:            Heads list with injected parameters
        """
        for i in range(len(heads_list)):
            heads_list[i] = factory.insert_module_param(heads_list[i], "num_classes", num_classes)
            heads_list[i] = factory.insert_module_param(heads_list[i], "reg_max", reg_max)
        return heads_list

    @torch.jit.ignore
    def _init_weights(self):
        if self.eval_size:
            device = infer_model_device(self)
            dtype = infer_model_dtype(self)

            anchor_points, stride_tensor = self._generate_anchors(dtype=dtype, device=device)
            self.anchor_points = anchor_points
            self.stride_tensor = stride_tensor

    def forward(self, feats: Tuple[Tensor, ...]) -> Union[YoloNasPoseDecodedPredictions, Tuple[YoloNasPoseDecodedPredictions, YoloNasPoseRawOutputs]]:
        """
        Runs the forward for all the underlying heads and concatenate the predictions to a single result.
        :param feats: List of feature maps from the neck of different strides
        :return: Return value depends on the mode:
        If tracing, a tuple of 5 tensors (decoded predictions) is returned:
        - fused_scores [B, Num Anchors, Num Classes] - class probability times detection quality
        - class_probabilities [B, Num Anchors, Num Classes] - conditional softmax class probabilities
        - quality_scores [B, Num Anchors, 1] - scalar foreground/localization quality
        - pred_pose_coords [B, Num Anchors, Num Keypoints, 2] - Predicted poses in XY format
        - pred_pose_scores [B, Num Anchors, Num Keypoints] - Predicted scores for each keypoint

        In training/eval mode, a tuple of 2 tensors returned:
        - decoded predictions - they are the same as in tracing mode
        - raw outputs - a tuple of 7 elements in total, this is needed for training the model.
        """

        cls_logits_list, quality_logits_list = [], []
        pose_regression_list = []
        pose_logits_list = []
        num_anchors_list: List[int] = []

        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            height_mul_width = h * w
            num_anchors_list.append(height_mul_width)
            cls_logit, quality_logit, pose_regression, pose_logits = getattr(self, f"head{i + 1}")(feat)
            cls_logits_list.append(cls_logit.reshape([b, -1, height_mul_width]))
            quality_logits_list.append(quality_logit.reshape([b, 1, height_mul_width]))

            pose_regression_list.append(torch.permute(pose_regression.flatten(3), [0, 3, 1, 2]))  # [B, J, 2, H, W] -> [B, H * W, J, 2]
            pose_logits_list.append(torch.permute(pose_logits.flatten(2), [0, 2, 1]))  # [B, J, H, W] -> [B, H * W, J]

        cls_logits = torch.cat(cls_logits_list, dim=-1).permute(0, 2, 1)  # [B, Anchors, C]
        quality_logits = torch.cat(quality_logits_list, dim=-1).permute(0, 2, 1)  # [B, Anchors, 1]

        pose_regression_list = torch.cat(pose_regression_list, dim=1)  # [B, Anchors, J, 2]
        pose_logits_list = torch.cat(pose_logits_list, dim=1)  # [B, Anchors, J]

        # Decode keypoints relative to the dense grid anchors. Boxes are not
        # predicted by the chess model.
        if self.eval_size:
            anchor_points_inference, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points_inference, stride_tensor = self._generate_anchors(feats)

        if self.pose_offset_multiplier != 1.0:
            pose_regression_list *= self.pose_offset_multiplier

        if self.compensate_grid_cell_offset:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2) - self.grid_cell_offset
        else:
            pose_regression_list += anchor_points_inference.unsqueeze(0).unsqueeze(2)

        pose_regression_list *= stride_tensor.unsqueeze(0).unsqueeze(2)

        class_probabilities = cls_logits.softmax(dim=-1)
        quality_scores = quality_logits.sigmoid()
        fused_scores = class_probabilities * quality_scores
        pred_pose_coords = pose_regression_list
        pred_pose_scores = pose_logits_list.sigmoid()

        decoded_predictions = fused_scores, class_probabilities, quality_scores, pred_pose_coords, pred_pose_scores

        if torch.jit.is_tracing() or self.inference_mode:
            return decoded_predictions

        raw_predictions = (
            cls_logits,
            quality_logits,
            pose_regression_list,
            pose_logits_list,
            anchor_points_inference * stride_tensor,
            num_anchors_list,
            stride_tensor,
        )
        return decoded_predictions, raw_predictions

    @property
    def out_channels(self):
        return None

    def _generate_anchors(self, feats=None, dtype=None, device=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []

        dtype = dtype or feats[0].dtype
        device = device or feats[0].device

        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            if torch_version_is_greater_or_equal(1, 10):
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            else:
                shift_y, shift_x = torch.meshgrid(shift_y, shift_x)

            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(dtype=dtype)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full([h * w, 1], stride, dtype=dtype))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)

        if device is not None:
            anchor_points = anchor_points.to(device)
            stride_tensor = stride_tensor.to(device)
        return anchor_points, stride_tensor
