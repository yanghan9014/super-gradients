import os
from typing import Optional, Tuple, List, Union

import cv2
import numpy as np
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.factories.type_factory import TypeFactory
from super_gradients.common.object_names import Datasets, Processings
from super_gradients.common.registry.registry import register_dataset
from super_gradients.training.datasets.data_formats.bbox_formats.xywh import xyxy_to_xywh
from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import AbstractPoseEstimationDataset
from super_gradients.training.datasets.pose_estimation_datasets.coco_utils import (
    CrowdAnnotationActionEnum,
    parse_coco_into_keypoints_annotations,
    parse_coco_into_multiclass_keypoints_annotations,
    segmentation2mask,
)
from super_gradients.training.samples import ChessPoseEstimationSample
# from super_gradients.training.samples import PoseEstimationSample
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform

logger = get_logger(__name__)


@register_dataset(Datasets.CHESS_POSE_ESTIMATION_DATASET)
class ChessPoseEstimationDataset(AbstractPoseEstimationDataset):
    """
    Dataset class for training pose estimation models using COCO format dataset.
    Please note that COCO annotations must have exactly one category (e.g. "person") and
    keypoints must be defined for this category.

    Compatible datasets are
    - COCO2017 dataset
    - CrowdPose dataset
    - Any other dataset that is compatible with COCO format

    """

    @resolve_param("transforms", TransformsFactory())
    # @resolve_param("crowd_annotations_action", TypeFactory.from_enum_cls(CrowdAnnotationActionEnum))
    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        json_file: str,
        include_empty_samples: bool,
        transforms: List[AbstractKeypointTransform],
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        remove_duplicate_annotations: bool = False,
        crowd_annotations_action: CrowdAnnotationActionEnum = CrowdAnnotationActionEnum.NO_ACTION,
        preserve_out_of_frame_board_annotations: bool = False,
        load_size: Optional[Tuple[int, int]] = None,
        piece_box_squares: Optional[float] = None,
        board_class_id: int = 12,
    ):
        """

        :param data_dir:                     Root directory of the COCO dataset
        :param images_dir:                   Path suffix to the images directory inside the dataset_root
        :param json_file:                    Path suffix to the json file inside the dataset_root
        :param include_empty_samples:        If True, images without any annotations will be included in the dataset.
                                             Otherwise, they will be filtered out.
        :param transforms:                   Transforms to be applied to the image & keypoints
        :param edge_links:                   Edge links between joints
        :param edge_colors:                  Color of the edge links. If None, the color will be generated randomly.
        :param keypoint_colors:              Color of the keypoints. If None, the color will be generated randomly.
        :param remove_duplicate_annotations: If True will remove duplicate instances from the dataset.
                                             It is known issue of COCO dataset - it has some duplicate annotations that affects the
                                             AP metric on validation greatly. This option allows to remove these duplicates.
                                             However, it is disabled by default to preserve backward compatibility with COCO evaluation.
                                             When remove_duplicate_annotations is False no action will be taken and these duplicate
                                             instances will be left unchanged. Default value is False.
        :param crowd_annotations_action:     Action to take for annotations with iscrowd=1. Can be one of the following:
                                             "drop_sample" - Samples with crowd annotations will be dropped from the dataset.
                                             "drop_annotation" - Crowd annotations will be dropped from the dataset.
                                             "mask_as_normal" - These annotations will be treated as normal (non-crowd) annotations.
                                             "no_action" - No action will be taken for crowd annotations.
        :param load_size:                    (width, height). If set, images are shrunk while loading to the smallest size that
                                             still covers it. Only valid when no later transform enlarges the image beyond that
                                             size (true of the board-framed crop and the letterbox), so the network sees the
                                             same detail while augmentation works on far fewer pixels.
        :param piece_box_squares:            If set, each piece's box becomes the image extent of a cell this many squares wide,
                                             centred on the piece in board coordinates. The annotated boxes are a fixed pixel size,
                                             so they don't follow piece size across photo resolutions. Boxes set the loss's
                                             anchor assignment and OKS scale. Pieces in images without a board fit keep theirs.
        :param board_class_id:               Class of the board, whose landmarks give the board coordinates.
        """
        json_file = os.path.join(data_dir, json_file)
        if not os.path.exists(json_file) or not os.path.isfile(json_file):
            raise FileNotFoundError(f"Annotation file {json_file} does not exist")

        self.category_name, self.joints, self.annotations = parse_coco_into_multiclass_keypoints_annotations(
            json_file,
            image_path_prefix=os.path.join(data_dir, images_dir),
            remove_duplicate_annotations=remove_duplicate_annotations,
            # crowd_annotations_action=crowd_annotations_action,
        )

        num_joints = len(self.joints)

        super().__init__(
            transforms=transforms,
            num_joints=num_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )
        if piece_box_squares is not None:
            for ann in self.annotations:
                _set_board_relative_piece_boxes(ann, piece_box_squares, board_class_id)
        self.load_size = tuple(load_size) if load_size is not None else None
        self.non_empty_annotation_indexes = np.argwhere([len(ann.ann_keypoints) > 0 for ann in self.annotations]).flatten()
        self.include_empty_samples = include_empty_samples
        self.transforms.preserve_out_of_frame_board_annotations = preserve_out_of_frame_board_annotations
        self.board_piece_counts = [_count_on_board_pieces(ann, board_class_id) for ann in self.annotations]

    def __len__(self):
        if self.include_empty_samples:
            return len(self.annotations)
        else:
            return len(self.non_empty_annotation_indexes)

    def load_sample(self, index: int) -> ChessPoseEstimationSample:
        """
        Read a sample from the disk and return a ChessPoseEstimationSample
        :param index: Sample index
        :return:      Returns an instance of ChessPoseEstimationSample that holds complete sample (image and annotations)
        """
        if not self.include_empty_samples:
            index = self.non_empty_annotation_indexes[index]
        ann = self.annotations[index]

        # gt_iscrowd = ann.ann_is_crowd.copy()
        gt_joints = ann.ann_keypoints.copy()
        gt_bboxes = ann.ann_boxes_xyxy.copy()
        # gt_segmentations = ann.ann_segmentations
        gt_areas = ann.ann_areas.copy()
        gt_class_id = ann.ann_class_labels.copy()

        orig_image, (scale_x, scale_y) = self._read_image(ann)
        image_height, image_width = orig_image.shape[:2]
        if (scale_x, scale_y) != (1.0, 1.0):
            gt_joints[:, :, 0] *= scale_x
            gt_joints[:, :, 1] *= scale_y
            gt_bboxes[:, [0, 2]] *= scale_x
            gt_bboxes[:, [1, 3]] *= scale_y
            gt_areas = gt_areas * (scale_x * scale_y)

        # Clip bboxes to image boundaries (Some annotations extend 1-2px outside of image boundaries)
        image_shape = (image_height, image_width)
        gt_bboxes[:, 0] = np.clip(gt_bboxes[:, 0], 0, image_width)
        gt_bboxes[:, 1] = np.clip(gt_bboxes[:, 1], 0, image_height)
        gt_bboxes[:, 2] = np.clip(gt_bboxes[:, 2], 0, image_width)
        gt_bboxes[:, 3] = np.clip(gt_bboxes[:, 3], 0, image_height)
        gt_bboxes_xywh = xyxy_to_xywh(gt_bboxes, image_shape=(image_height, image_width))

        # mask: np.ndarray = self._get_crowd_mask(gt_segmentations[gt_iscrowd], image_shape)
        # Nothing downstream reads the mask beyond carrying it through the transforms.
        mask = np.ones(image_shape, dtype=np.uint8)

        return ChessPoseEstimationSample(
            image=orig_image, mask=mask, joints=gt_joints, areas=gt_areas, labels=gt_class_id, bboxes_xywh=gt_bboxes_xywh,
            additional_samples=None, board_piece_counts=self.board_piece_counts[index].copy(),
        )

    def _read_image(self, ann) -> Tuple[np.ndarray, Tuple[float, float]]:
        """RGB image, shrunk to cover `load_size` if set, and the (x, y) scale from file to returned pixels.

        JPEG decodes directly at 1/2, 1/4 or 1/8 size where that still covers the target; an area resize does the rest.
        """
        file_size = np.array([ann.image_width, ann.image_height], dtype=np.float64)
        scale = 1.0 if self.load_size is None else min(1.0, float(np.max(np.divide(self.load_size, file_size))))
        reduction = next(r for r in (8, 4, 2, 1) if r * scale <= 1)
        flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[reduction]
        image = cv2.imread(ann.image_path, flags | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            # Some OpenCV versions cannot read a few files. Note: PIL applies no EXIF orientation here either.
            from PIL import Image

            image, reduction = np.array(Image.open(ann.image_path).convert("RGB")), 1
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        decoded = np.array(image.shape[1::-1], dtype=np.float64)
        # Reduced JPEG decoding rounds sizes up, other formats to the nearest pixel.
        if np.all(np.abs(decoded - file_size / reduction) <= 1):
            pass
        elif np.all(np.abs(decoded - file_size[::-1] / reduction) <= 1):
            # Annotated (H, W) is swapped relative to the file; use the file's own orientation.
            file_size = file_size[::-1]
        else:
            raise RuntimeError(f"Annotated image size {tuple(file_size[::-1])} does not match image size in file {ann.image_path}")

        if self.load_size is not None:
            scale = min(1.0, float(np.max(np.divide(self.load_size, file_size))))
        target = np.maximum(np.round(file_size * scale), 1).astype(int)
        if np.any(target > decoded):
            # Only possible for swapped sizes: decode again at full size.
            image = cv2.cvtColor(cv2.imread(ann.image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), cv2.COLOR_BGR2RGB)
            decoded = np.array(image.shape[1::-1], dtype=np.float64)
        if np.any(target != decoded):
            image = cv2.resize(image, tuple(int(v) for v in target), interpolation=cv2.INTER_AREA)
        scale_x, scale_y = (target / file_size).tolist() if np.any(target != file_size) else (1.0, 1.0)
        return image, (scale_x, scale_y)

    # def _get_crowd_mask(self, segmentations: List[str], image_shape: Tuple[int, int]) -> np.ndarray:
    #     """
    #     This method computes ignore mask, which describes crowd objects / objects w/o keypoints to exclude these predictions from contributing to the loss
    #     :return: Float mask of [H,W] shape (same as image dimensions),
    #         where 1.0 values corresponds to pixels that should contribute to the loss, and 0.0 pixels indicates areas that should be excluded.
    #     """
    #     m = np.zeros(image_shape, dtype=bool)

    #     for segmentation in segmentations:
    #         mask = segmentation2mask(segmentation, image_shape)
    #         m[mask] = True

    #     return (m < 0.5).astype(np.float32)

    def get_dataset_preprocessing_params(self) -> dict:
        """
        This method returns a dictionary of parameters describing preprocessing steps to be applied to the dataset.
        :return:
        """
        image_to_tensor = {Processings.ImagePermute: {"permutation": (2, 0, 1)}}
        pipeline = self.transforms.get_equivalent_preprocessing() + [image_to_tensor]
        params = dict(
            conf=0.05,
            image_processor={Processings.ComposeProcessing: {"processings": pipeline}},
            edge_links=self.edge_links,
            edge_colors=self.edge_colors,
            keypoint_colors=self.keypoint_colors,
        )
        return params


def _count_on_board_pieces(ann, board_class_id: int) -> np.ndarray:
    """Count original annotated piece centres inside the board, before any crop/filter.

    Missing/degenerate board geometry or unlocatable pieces leave the count unknown
    (-1), so orientation stays strict. Captured pieces outside the board do not count.
    """
    from super_gradients.training.metrics.board_homography_metrics import _fit_homography, _project

    labels = np.asarray(ann.ann_class_labels).reshape(-1)
    counts = np.full(len(labels), -1, dtype=np.int64)
    boards = np.flatnonzero(labels == board_class_id)
    if len(boards) != 1:
        return counts
    landmarks = ann.ann_keypoints[boards[0]]
    usable = (landmarks[:, 2] > 0) & np.isfinite(landmarks[:, :2]).all(axis=-1)
    homography = _fit_homography(landmarks[:, :2], usable, method=0)
    if homography is None:
        return counts
    # Reject inconsistent geometry rather than turning a bad fit into a small count.
    from super_gradients.training.metrics.board_homography_metrics import BOARD_REFERENCE_POINTS
    fitted = _project(landmarks[usable, :2], homography)
    if fitted is None or np.max(np.abs(fitted - BOARD_REFERENCE_POINTS[usable])) > 0.25:
        return counts
    pieces = ann.ann_keypoints[labels != board_class_id, 0]
    if len(pieces) == 0:
        counts[boards[0]] = 0
        return counts
    if not ((pieces[:, 2] > 0) & np.isfinite(pieces[:, :2]).all(axis=-1)).all():
        return counts
    centres = _project(pieces[:, :2], homography)
    if centres is not None:
        counts[boards[0]] = int(((centres >= -1e-6) & (centres <= 8 + 1e-6)).all(axis=-1).sum())
    return counts


def _set_board_relative_piece_boxes(ann, squares: float, board_class_id: int) -> None:
    """Replace piece boxes (and areas) with the image extent of a square cell centred on each piece in board coordinates.

    Board coordinates come from the board's annotated landmarks (one unit = one square). Images without exactly one
    board with 4+ usable landmarks, and cells that cross the board plane's horizon, keep their annotated boxes.
    """
    from super_gradients.training.metrics.board_homography_metrics import BOARD_REFERENCE_POINTS

    labels = np.asarray(ann.ann_class_labels).reshape(-1)
    if len(labels) == 0:
        return
    boards = np.flatnonzero(labels == board_class_id)
    pieces = np.flatnonzero((labels != board_class_id) & (ann.ann_keypoints[:, 0, 2] > 0))
    if len(boards) != 1 or len(pieces) == 0:
        return
    landmarks = ann.ann_keypoints[boards[0]].astype(np.float64)
    usable = (landmarks[:, 2] > 0) & np.isfinite(landmarks[:, :2]).all(axis=1)
    if usable.sum() < 4:
        return
    to_board, _ = cv2.findHomography(landmarks[usable, :2], BOARD_REFERENCE_POINTS[usable], 0)
    if to_board is None or not np.isfinite(to_board).all():
        return
    to_image = np.linalg.inv(to_board)

    centres = cv2.perspectiveTransform(ann.ann_keypoints[pieces, 0:1, :2].astype(np.float64), to_board).reshape(-1, 1, 2)
    half = squares / 2
    cells = centres + np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    projected = np.concatenate([cells, np.ones(cells.shape[:2] + (1,))], axis=-1) @ to_image.T
    # The homogeneous coordinate keeps the board centre's sign on the camera's side of the horizon.
    side = np.sign((np.array([4.0, 4.0, 1.0]) @ to_image.T)[2])
    keep = (projected[..., 2] * side > 1e-9).all(axis=1)
    corners = projected[keep, :, :2] / projected[keep, :, 2:3]
    boxes = np.concatenate([corners.min(axis=1), corners.max(axis=1)], axis=1)
    ann.ann_boxes_xyxy = ann.ann_boxes_xyxy.astype(np.float32, copy=True)
    ann.ann_areas = ann.ann_areas.astype(np.float32, copy=True)
    ann.ann_boxes_xyxy[pieces[keep]] = boxes
    ann.ann_areas[pieces[keep]] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
