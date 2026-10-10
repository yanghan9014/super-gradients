"""App-pipeline piece metrics on synthetic raw outputs; no model or training data."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from super_gradients.training.metrics.chess_piece_metrics import (
    BOARD_COORDS,
    EMPTY,
    annotated_squares,
    app_homography,
    app_square_readings,
    apply_homography,
    match_squares,
    square_name,
)
from super_gradients.training.metrics.chess_pose_estimation_metrics import ChessPoseEstimationMetrics

SIZE = (640, 480)
# Board units to model-input pixels: a mild perspective view of the board.
PIXELS_FROM_BOARD = np.array([[40.0, 6, 150], [-3, 26, 120], [0.0004, 0.002, 1]])


def to_pixels(board_points):
    return apply_homography(np.asarray(board_points, dtype=np.float32), PIXELS_FROM_BOARD.astype(np.float32))


def centre(square):
    """Pixel position of a square's centre, e.g. "e4"."""
    return to_pixels([("abcdefgh".index(square[0]) + 0.5, 8 - int(square[1]) + 0.5)])[0]


def raw_outputs(board_score=0.9, landmark_scores=None, pieces=()):
    """Per-anchor class scores, keypoints and keypoint scores: one board anchor, then the pieces."""
    anchors = 1 + len(pieces)
    scores = np.zeros((anchors, 13), dtype=np.float32)
    coords = np.zeros((anchors, 9, 2), dtype=np.float32)
    point_scores = np.zeros((anchors, 9), dtype=np.float32)
    scores[0, 12] = board_score
    coords[0] = to_pixels(BOARD_COORDS)
    point_scores[0] = 0.9 if landmark_scores is None else landmark_scores
    for anchor, (cls, confidence, square) in enumerate(pieces, start=1):
        scores[anchor, cls] = confidence
        coords[anchor, 0] = centre(square)
    return scores, coords, point_scores


def readings(**kwargs):
    found = app_square_readings(*raw_outputs(**kwargs), SIZE)
    return {square_name(s): (c, round(p, 3)) for s, (c, p) in found.items()}


def test_dlt_recovers_the_board_homography_on_normalized_points():
    normalized = to_pixels(BOARD_COORDS) / np.asarray(SIZE, dtype=np.float32)
    homography = app_homography(normalized, BOARD_COORDS)
    assert np.allclose(apply_homography(normalized, homography), BOARD_COORDS, atol=1e-3)


def test_most_confident_piece_wins_each_square_without_nms():
    found = readings(pieces=[(3, 0.6, "e4"), (9, 0.8, "e4"), (1, 0.3, "g1"), (4, 0.04, "d1")])
    # The 0.04 queen is below the app's 0.05 threshold.
    assert found == {"e4": (9, 0.8), "g1": (1, 0.3)}


def test_app_reads_nothing_without_a_board_or_four_landmarks():
    pieces = [(3, 0.9, "e4")]
    assert readings(board_score=0.04, pieces=pieces) == {}
    assert readings(landmark_scores=[0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], pieces=pieces) == {}
    assert readings(landmark_scores=[0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1], pieces=pieces) == {"e4": (3, 0.9)}


def test_off_board_pieces_are_dropped():
    scores, coords, point_scores = raw_outputs(pieces=[(3, 0.9, "e4")])
    coords[1, 0] = to_pixels([(9.5, 4.5)])[0]
    assert app_square_readings(scores, coords, point_scores, SIZE) == {}


def annotation(pieces, visible_landmarks=9):
    joints = np.zeros((1 + len(pieces), 9, 3), dtype=np.float32)
    joints[0, :, :2] = to_pixels(BOARD_COORDS)
    joints[0, :visible_landmarks, 2] = 1
    for i, (_, square) in enumerate(pieces, start=1):
        joints[i, 0] = [*centre(square), 1]
    return joints, np.array([12] + [cls for cls, _ in pieces])


def test_annotation_places_pieces_and_needs_a_usable_board():
    assert {square_name(s): c for s, c in annotated_squares(*annotation([(3, "e4"), (7, "e8")]), SIZE).items()} == {"e4": 3, "e8": 7}
    assert annotated_squares(*annotation([(3, "e4")], visible_landmarks=3), SIZE) is None


def test_matching_and_confusion():
    # Annotated: pawn e4, king g1, queen d1. Read: pawn e4 (right), knight g1 (wrong class), rook a8 (on an empty square).
    truth = {36: 3, 62: 1, 59: 4}
    found = {36: (3, 0.9), 62: (2, 0.7), 0: (5, 0.5)}
    per_class, confusion = match_squares(found, truth)
    assert per_class[3][0].tolist() == [True] and per_class[3][2] == 1
    assert per_class[2][0].tolist() == [False] and per_class[2][2] == 0
    assert per_class[1][0].tolist() == [] and per_class[1][2] == 1
    assert confusion[3, 3] == 1 and confusion[1, 2] == 1 and confusion[EMPTY, 5] == 1 and confusion[4, EMPTY] == 1
    assert confusion.sum() == 4


def make_metric():
    def no_poses(outputs):
        empty = SimpleNamespace(poses=np.zeros((0, 9, 2)), scores=np.zeros((0, 13)), pose_scores=np.zeros((0, 9)))
        return [empty] * len(outputs[0][1])

    return ChessPoseEstimationMetrics(post_prediction_callback=no_poses, num_joints=9, oks_sigmas=[0.5] * 9, board_class_id=12, piece_metrics=True)


def update(metric, raw, pieces):
    scores, coords, point_scores = (torch.from_numpy(x)[None] for x in raw)
    joints, labels = annotation(pieces)
    sample = SimpleNamespace(joints=joints, labels=labels, bboxes_xywh=np.tile([0, 0, 10, 10], (len(labels), 1)), areas=np.full(len(labels), 100.0))
    metric.update(preds=((torch.zeros(1, len(scores[0]), 4), scores, coords, point_scores),), target=None, gt_samples=[sample])


def test_metric_reports_piece_ap_ar_and_keeps_confusion_until_reset():
    metric = make_metric()
    # Pawn found; king read as a knight; queen missed.
    update(metric, raw_outputs(pieces=[(3, 0.9, "e4"), (2, 0.7, "g1")]), [(3, "e4"), (1, "g1"), (4, "d1")])
    result = metric.compute()
    # Pawn AP and AR are 1, king and queen are 0 (missed), the knight class has no annotations.
    assert result["Piece_AR"] == pytest.approx(1 / 3)
    assert result["Piece_AP"] == pytest.approx(1 / 3)
    assert {"AP", "AR", "Board_AP", "Board_AR"} <= set(result)
    assert metric.piece_confusion_matrix().sum() == 3
    metric.reset()
    assert metric.piece_confusion_matrix().sum() == 0


def test_logging_names_survive_lowercasing():
    from super_gradients.training.metrics.chess_piece_metrics import PIECE_NAMES, PIECE_SYMBOLS

    assert len({name.lower() for name in PIECE_NAMES}) == len(PIECE_SYMBOLS) == 12
    assert PIECE_NAMES[PIECE_SYMBOLS.index("b")] == "black_bishop"
