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
    confusion_percentages,
    confusion_figure,
    piece_average_precision,
    tune_piece_threshold,
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
    assert confusion.sum() == 64
    assert confusion[EMPTY, EMPTY] == 60


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


def test_metric_reports_macro_ap_color_and_keeps_confusion_until_reset():
    metric = make_metric()
    # Pawn found; king read as a knight; queen missed.
    update(metric, raw_outputs(pieces=[(3, 0.9, "e4"), (2, 0.7, "g1")]), [(3, "e4"), (1, "g1"), (4, "d1")])
    result = metric.compute()
    # Pawn AP is 1, king and queen are 0; absent classes are undefined and excluded.
    assert result["Piece_mAP"] == pytest.approx(1 / 3)
    assert result["Piece_AP/white_pawn"] == 1
    assert result["Piece_AP/white_king"] == 0
    assert result["Piece_AP/white_knight"] == -1
    assert "Piece_AP" not in result and "Piece_AR" not in result
    assert result["Piece_Color_Accuracy"] == 1  # Wrong type is still the correct color.
    assert result["Piece_EndToEnd_Incorrect_Squares"] == 2
    assert result["Piece_EndToEnd_Threshold"] == pytest.approx(0.9)
    assert {"AP", "AR", "Board_AP", "Board_AR"} <= set(result)
    assert list(result) == metric.component_names
    assert not metric.greater_component_is_better["Piece_EndToEnd_Incorrect_Squares"]
    assert metric.piece_confusion_matrix().sum() == 64
    metric.reset()
    assert metric.piece_confusion_matrix().sum() == 0
    assert metric.piece_predictions == metric.piece_color_counts == metric.piece_end_to_end_frames == []


def test_logging_names_survive_lowercasing():
    from super_gradients.training.metrics.chess_piece_metrics import PIECE_NAMES, PIECE_SYMBOLS

    assert len({name.lower() for name in PIECE_NAMES}) == len(PIECE_SYMBOLS) == 12
    assert PIECE_NAMES[PIECE_SYMBOLS.index("b")] == "black_bishop"


def test_trainer_logging_preserves_metric_names_and_values():
    from torchmetrics import MetricCollection
    from super_gradients.training.metrics.metric_utils import get_metrics_dict, get_metrics_results_tuple

    metric = make_metric()
    update(metric, raw_outputs(pieces=[(3, 0.8, "e4"), (9, 0.2, "g1")]), [(3, "e4"), (1, "g1")])
    expected = metric.compute()
    collection = MetricCollection([metric])
    logged = get_metrics_dict(get_metrics_results_tuple(collection), collection, [])
    assert logged == pytest.approx(expected)


def test_callback_reads_validation_metrics_from_the_trainer_context():
    from torchmetrics import MetricCollection

    from super_gradients.training.utils.callbacks import ChessPieceConfusionCallback
    from super_gradients.training.utils.callbacks.base_callbacks import PhaseContext

    metric = make_metric()
    update(metric, raw_outputs(pieces=[(3, 0.9, "e4")]), [(3, "e4"), (4, "d1")])
    logged = {}
    logger = SimpleNamespace(
        add_figure=lambda tag, figure, global_step: logged.update({tag: figure}), add_scalar=lambda tag, value, global_step: logged.update({tag: value})
    )
    # As in Trainer.train: metrics_compute_fn is the (empty) training collection at validation end.
    context = PhaseContext(epoch=9, metrics_compute_fn=MetricCollection([]), valid_metrics=MetricCollection([metric]), sg_logger=logger)
    ChessPieceConfusionCallback()(context)
    assert "Valid_Piece_Confusion" in logged
    assert logged["Valid_Piece_FN_Percent/white_queen"] == 100
    assert logged["Valid_Piece_FN_Percent/white_pawn"] == 0
    assert logged["Valid_Piece_FP_Percent/white_pawn"] == 0
    assert "Valid_Piece_FN_Percent/black_king" not in logged  # No GT black kings.


@pytest.mark.parametrize("board_score", [0.0, 0.9])
def test_recognition_uses_gt_board_and_no_piece_cutoff(board_score):
    metric = make_metric()
    raw = raw_outputs(board_score=board_score, pieces=[(3, 0.0001, "e4")])
    raw[2][:] = 0  # Predicted board has no usable landmarks regardless of its score.
    update(metric, raw, [(3, "e4")])
    result = metric.compute()
    assert result["Piece_mAP"] == 1
    assert result["Piece_Color_Accuracy"] == 1
    assert result["Piece_EndToEnd_Incorrect_Squares"] == 1  # Failed extraction remains a miss.
    assert metric.piece_confusion_matrix()[3, EMPTY] == 1


def test_color_counts_wrong_colors_but_ignores_type_and_missing_predictions():
    metric = make_metric()
    update(metric, raw_outputs(pieces=[(4, 0.001, "e4"), (9, 0.9, "g1")]), [(3, "e4"), (1, "g1"), (4, "d1")])
    result = metric.compute()
    assert result["Piece_Color_Accuracy"] == 0.5
    assert result["Piece_mAP"] == 0
    assert metric.piece_color_counts == [(1, 2)]


def test_ap_penalizes_bad_ranking_and_missing_targets_and_groups_ties():
    levels = np.linspace(0, 1, 101)
    ap = lambda matched, scores, targets=2: piece_average_precision(np.array(matched), np.array(scores), targets, levels)
    assert ap([True, True, False], [0.9, 0.8, 0.7]) == 1
    assert ap([False, True, True], [0.9, 0.8, 0.7]) == pytest.approx(2 / 3)
    assert ap([True], [0.9]) == pytest.approx(51 / 101)  # Missing target limits recall.
    assert ap([True, False], [0.5, 0.5], 1) == 0.5
    assert ap([False, True], [0.5, 0.5], 1) == 0.5
    assert ap([False, True, True], [0.09, 0.08, 0.07]) == ap([False, True, True], [0.9, 0.8, 0.7])


def test_macro_ap_weights_classes_equally_not_by_instance_count():
    metric = make_metric()
    update(metric, raw_outputs(pieces=[(3, 0.9, "e4"), (3, 0.8, "d4")]), [(3, "e4"), (3, "d4"), (1, "g1")])
    assert metric.compute()["Piece_mAP"] == 0.5


def test_end_to_end_tunes_piece_cutoff_without_changing_board_gate_or_confusion_cutoff():
    metric = make_metric()
    update(metric, raw_outputs(board_score=0.1, pieces=[(3, 0.8, "e4"), (5, 0.06, "a8")]), [(3, "e4")])
    result = metric.compute()
    assert result["Piece_EndToEnd_Threshold"] == pytest.approx(0.8)  # Board score is only 0.1.
    assert result["Piece_EndToEnd_Incorrect_Squares"] == 0
    assert metric.piece_confusion_matrix()[EMPTY, 5] == 1  # Existing fixed 0.05 diagnostics.


def test_sweep_matches_brute_force_including_ties_wrong_classes_and_failures():
    rng = np.random.default_rng(17)
    for _ in range(50):
        frames = []
        for _ in range(4):
            truth = {s: int(rng.integers(0, 12)) for s in range(12) if rng.random() < 0.6}
            found = {s: (int(rng.integers(0, 12)), float(rng.choice([0, 0.1, 0.5, 0.9, 1]))) for s in range(12) if rng.random() < 0.7}
            frames.append((found, truth))
        frames.append(({}, {0: 1}))  # Extraction failure is included in the denominator.
        thresholds = {1.0} | {score for found, _ in frames for _, score in found.values()}
        results = []
        for threshold in thresholds:
            errors = 0
            for found, truth in frames:
                accepted = {s: cls for s, (cls, score) in found.items() if score >= threshold}
                errors += sum(accepted.get(s, EMPTY) != truth.get(s, EMPTY) for s in range(64))
            results.append((errors / len(frames), -threshold))
        errors, negative_threshold = min(results)
        assert tune_piece_threshold(frames) == pytest.approx((errors, -negative_threshold))


def test_sweep_accepts_tied_scores_together_and_prefers_highest_threshold():
    assert tune_piece_threshold([({0: (3, 0.5), 1: (3, 0.5)}, {0: 3})]) == (1, 1)
    assert tune_piece_threshold([({0: (3, 1.0)}, {})]) == (1, 1)  # >= 1 still accepts score 1.
    assert tune_piece_threshold([({}, {})]) == (0, 1)
    assert tune_piece_threshold([]) == (-1, -1)


def test_confusion_percentages_use_all_empty_squares_and_render_below_counts():
    _, confusion = match_squares({0: (3, 0.9), 1: (5, 0.8)}, {0: 3, 2: 3})
    percentages = confusion_percentages(confusion)
    assert percentages[3, EMPTY] == 50
    assert percentages[EMPTY, 5] == pytest.approx(100 / 62)
    assert np.isnan(percentages[1]).all()
    figure = confusion_figure(confusion)
    assert "1\n(50.0%)" in [text.get_text() for text in figure.axes[0].texts]


def test_unscorable_annotation_is_excluded_and_empty_metrics_are_undefined():
    metric = make_metric()
    scores, coords, point_scores = (torch.from_numpy(x)[None] for x in raw_outputs())
    joints, labels = annotation([(3, "e4")], visible_landmarks=3)
    metric._update_pieces(((None, scores, coords, point_scores),), [SimpleNamespace(joints=joints, labels=labels)])
    result = metric.compute()
    assert result["Piece_mAP"] == result["Piece_Color_Accuracy"] == result["Piece_EndToEnd_Incorrect_Squares"] == -1
    assert metric.piece_confusion_matrix().sum() == 0


def test_distributed_states_produce_global_ap_color_and_threshold(monkeypatch):
    from copy import deepcopy

    left, right, combined = make_metric(), make_metric(), make_metric()
    first = raw_outputs(pieces=[(3, 0.8, "e4"), (5, 0.2, "a8")])
    second = raw_outputs(pieces=[(9, 0.7, "e4"), (3, 0.1, "d4")])
    update(left, first, [(3, "e4")])
    update(right, second, [(3, "e4"), (3, "d4")])
    update(combined, first, [(3, "e4")])
    update(combined, second, [(3, "e4"), (3, "d4")])
    states = iter(
        deepcopy(getattr(right, name))
        for name in (
            "predictions",
            "board_predictions",
            "board_relaxed_predictions",
            "piece_predictions",
            "piece_confusions",
            "piece_color_counts",
            "piece_end_to_end_frames",
        )
    )

    def gather(destination, local):
        destination[:] = [deepcopy(local), next(states)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    left.is_distributed, left.world_size, left.rank = True, 2, 0
    left._sync_dist()
    left.is_distributed = False
    assert left.compute() == pytest.approx(combined.compute())
    assert np.array_equal(left.piece_confusion_matrix(), combined.piece_confusion_matrix())
