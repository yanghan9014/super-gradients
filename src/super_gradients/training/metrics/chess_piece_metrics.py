"""Single-frame piece evaluation without NMS.

Recognition AP/color use annotated board geometry and no confidence cutoff. App diagnostics
use predicted geometry and the configured cutoff. End-to-end evaluation fixes board settings
and tunes only the piece cutoff from cached square winners (no additional model inference).
"""

from itertools import groupby
from typing import Dict, Optional, Tuple

import numpy as np

BOARD_CLASS_ID = 12
NUM_PIECE_CLASSES = 12
# Index order of the class axis in the confusion matrix: the 12 pieces, then an empty square.
EMPTY = NUM_PIECE_CLASSES
PIECE_SYMBOLS = ["B", "K", "N", "P", "Q", "R", "b", "k", "n", "p", "q", "r"]
# Spelled out for logging tags: SG lowercases tags, which would merge "B" and "b".
PIECE_NAMES = [f"{colour}_{piece}" for colour in ("white", "black") for piece in ("bishop", "king", "knight", "pawn", "queen", "rook")]

# Board landmarks a1, a8, h1, h8, center, a45, h45, 1de, 8de in board units (one square each),
# and the order PostProcessor.cornerPriority feeds them to the homography.
BOARD_COORDS = np.array([(0, 8), (0, 0), (8, 8), (8, 0), (4, 4), (0, 4), (8, 4), (4, 8), (4, 0)], dtype=np.float32)
CORNER_PRIORITY = [0, 1, 2, 3, 5, 6, 7, 8, 4]


def app_homography(src: np.ndarray, dst: np.ndarray) -> Optional[np.ndarray]:
    """PostProcessor.calculateHomography: plain float32 DLT solved by SVD, no point normalization.

    The app feeds keypoints normalized to [0, 1] by the model input size; because this DLT is not
    scale invariant, callers must normalize the same way to get the same matrix.
    """
    if len(src) < 4 or len(src) != len(dst):
        return None
    rows = []
    for (x, y), (u, v) in zip(src.astype(np.float32), dst.astype(np.float32)):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    try:
        _, _, vt = np.linalg.svd(np.array(rows, dtype=np.float32))
    except np.linalg.LinAlgError:
        return None
    h = vt[-1]
    if abs(h[8]) <= 1e-9:
        return None
    return (h / h[8]).reshape(3, 3)


def apply_homography(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    projected = np.column_stack((points, np.ones(len(points), dtype=np.float32))) @ homography.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return projected[:, :2] / projected[:, 2:3]


def board_square(point: np.ndarray) -> Optional[int]:
    """PostProcessor.boardSquare as an index file + 8 * row, row 0 being rank 8."""
    x, y = float(point[0]), float(point[1])
    if not (0.0 <= x < 8.0 and 0.0 <= y < 8.0):  # NaN fails these comparisons too
        return None
    return int(np.floor(y)) * 8 + int(np.floor(x))


def square_name(index: int) -> str:
    return "abcdefgh"[index % 8] + str(8 - index // 8)


def _board_homography(landmarks: np.ndarray, usable: np.ndarray, image_size: Tuple[int, int]) -> Optional[np.ndarray]:
    indices = [i for i in CORNER_PRIORITY if usable[i]]
    if len(indices) < 4:
        return None
    normalized = landmarks[indices] / np.asarray(image_size, dtype=np.float32)
    return app_homography(normalized, BOARD_COORDS[indices])


def _place(points: np.ndarray, homography: np.ndarray, image_size: Tuple[int, int]) -> list:
    normalized = points.astype(np.float32) / np.asarray(image_size, dtype=np.float32)
    return [board_square(p) for p in apply_homography(normalized, homography)]


def app_square_readings(
    class_scores: np.ndarray,
    pose_coords: np.ndarray,
    pose_scores: np.ndarray,
    image_size: Tuple[int, int],
    confidence_threshold: float = 0.05,
    board_keypoint_threshold: float = 0.2,
    piece_confidence_threshold: Optional[float] = None,
) -> Dict[int, Tuple[int, float]]:
    """One frame as the app reads it: {square: (piece class, confidence)}.

    Inputs are the raw per-anchor model outputs in model-input pixels: class scores [A, 13],
    keypoints [A, 9, 2] and keypoint scores [A, 9]. An empty result means the app would have
    no board state for this frame (no board, fewer than four landmarks, or no pieces on it).
    """
    homography = app_board_homography(class_scores, pose_coords, pose_scores, image_size, confidence_threshold, board_keypoint_threshold)
    if homography is None:
        return {}
    return square_readings(
        class_scores,
        pose_coords,
        homography,
        image_size,
        confidence_threshold if piece_confidence_threshold is None else piece_confidence_threshold,
    )


def app_board_homography(
    class_scores: np.ndarray,
    pose_coords: np.ndarray,
    pose_scores: np.ndarray,
    image_size: Tuple[int, int],
    confidence_threshold: float = 0.05,
    board_keypoint_threshold: float = 0.2,
) -> Optional[np.ndarray]:
    """The app's board geometry: its most confident board, fitted from landmarks above the keypoint cutoff."""
    board_scores = class_scores[:, BOARD_CLASS_ID]
    boards = np.flatnonzero(board_scores >= confidence_threshold)
    if len(boards) == 0:
        return None
    best = boards[np.argmax(board_scores[boards])]
    return _board_homography(pose_coords[best], pose_scores[best] >= board_keypoint_threshold, image_size)


def board_rotated_180(predicted: Optional[np.ndarray], annotated: np.ndarray) -> bool:
    """Whether the predicted board fits the annotation better turned 180 degrees, (x, y) -> (8 - x, 8 - y).

    The same whole-board test as Board_Relaxed_AP: the annotated reference points, read through the predicted
    homography, compared with and without the turn by their largest displacement. Ties and failures: not turned.
    """
    if predicted is None:
        return False
    with np.errstate(all="ignore"):
        image_points = apply_homography(BOARD_COORDS, np.linalg.inv(annotated).astype(np.float32))
        read = apply_homography(image_points, predicted)
        strict = np.abs(read - BOARD_COORDS).max()
        turned = np.abs((8.0 - read) - BOARD_COORDS).max()
    return bool(turned < strict)


def rotate_readings_180(readings: Dict[int, Tuple[int, float]]) -> Dict[int, Tuple[int, float]]:
    """The same readings on a board turned 180 degrees: a1 <-> h8, square index s -> 63 - s."""
    return {63 - square: reading for square, reading in readings.items()}


def square_readings(class_scores, pose_coords, homography, image_size, confidence_threshold=0.0):
    """Highest-scoring piece per square in the supplied geometry; no class-aware matching.

    A zero cutoff keeps even zero-score candidates. No candidates on a square means no reading.
    """
    piece_scores = class_scores[:, :NUM_PIECE_CLASSES]
    piece_class = piece_scores.argmax(axis=1)
    piece_conf = piece_scores.max(axis=1)
    pieces = np.flatnonzero(piece_conf >= confidence_threshold)
    readings: Dict[int, Tuple[int, float]] = {}
    # The piece's position is its first keypoint. Strictly higher confidence replaces the
    # current reading, so ties keep the earlier anchor, as in extractBoardState.
    for anchor, square in zip(pieces, _place(pose_coords[pieces, 0], homography, image_size)):
        if square is None:
            continue
        if square not in readings or piece_conf[anchor] > readings[square][1]:
            readings[square] = (int(piece_class[anchor]), float(piece_conf[anchor]))
    return readings


def annotated_board_homography(joints, labels, image_size):
    """First annotated board with at least four usable landmarks, or None."""
    for board in np.flatnonzero(labels == BOARD_CLASS_ID):
        homography = _board_homography(joints[board, :, :2], joints[board, :, 2] > 0, image_size)
        if homography is not None:
            return homography
    return None


def annotated_squares(joints: np.ndarray, labels: np.ndarray, image_size: Tuple[int, int], homography=None) -> Optional[Dict[int, int]]:
    """{square: piece class}, or None when the annotation has no usable board."""
    if homography is None:
        homography = annotated_board_homography(joints, labels, image_size)
    if homography is None:
        return None
    pieces = [i for i in range(len(labels)) if labels[i] < NUM_PIECE_CLASSES and joints[i, 0, 2] > 0]
    squares = _place(joints[pieces, 0, :2], homography, image_size) if pieces else []
    return {square: int(labels[i]) for i, square in zip(pieces, squares) if square is not None}


def match_squares(readings: Dict[int, Tuple[int, float]], truth: Dict[int, int]) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray, int]], np.ndarray]:
    """Per-class (matched, scores, number of annotated pieces) for AP/AR, and a 13x13 confusion matrix.

    A reading is a true positive when its square holds an annotated piece of the same class.
    Each square has one reading, so matches are one-to-one. Confusion rows are the annotated
    content of a square and columns the reading, with the last index meaning an empty square.
    """
    per_class = {}
    for cls in range(NUM_PIECE_CLASSES):
        squares = [s for s, (c, _) in readings.items() if c == cls]
        matched = np.array([truth.get(s) == cls for s in squares], dtype=bool)
        scores = np.array([readings[s][1] for s in squares], dtype=np.float32)
        targets = sum(1 for c in truth.values() if c == cls)
        if len(squares) or targets:
            per_class[cls] = (matched, scores, targets)
    confusion = np.zeros((NUM_PIECE_CLASSES + 1, NUM_PIECE_CLASSES + 1), dtype=np.int64)
    # Include true empty squares so the empty row measures phantom rates, not their composition.
    for square in range(64):
        confusion[truth.get(square, EMPTY), readings[square][0] if square in readings else EMPTY] += 1
    return per_class, confusion


def piece_average_precision(matched, scores, targets, recall_thresholds):
    """Interpolated AP at the supplied recall levels, with equal scores accepted together.

    -1 means no annotated examples of this class; exclude it from the macro average.
    Missing pieces remain in the recall denominator. No deployment score cutoff is applied.
    """
    if targets == 0:
        return -1.0
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    scores, matched = scores[order], matched[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(matched)[ends]
    recall = tp / targets
    precision = tp / (ends + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, recall_thresholds, side="left")
    return float(np.r_[precision, 0.0][indices].mean())


def tune_piece_threshold(frames):
    """Return (incorrect squares/frame, threshold) for cached (readings, truth) pairs.

    Optimize one global piece threshold with >= acceptance in [0, 1]. Board settings and
    square winners are fixed. Sweep distinct scores, accepting ties together; choose the
    highest threshold when errors tie. An extraction failure has empty readings, so every
    annotated piece still contributes an error. No scored frames returns (-1, -1).
    """
    if not frames:
        return -1.0, -1.0
    errors = sum(len(truth) for _, truth in frames)  # All readings initially rejected.
    events = []
    for readings, truth in frames:
        for square, (cls, score) in readings.items():
            # Accepting a correct piece fixes one miss; a wrong class leaves one error;
            # accepting a phantom adds one error.
            delta = -1 if truth.get(square) == cls else int(square not in truth)
            events.append((score, delta))
    best_errors, best_threshold = float("inf"), 1.0
    if not events or max(score for score, _ in events) < 1.0:
        best_errors = errors  # Threshold 1 rejects every reading.
    events.sort(reverse=True)
    for score, group in groupby(events, key=lambda event: event[0]):
        errors += sum(delta for _, delta in group)
        if errors < best_errors:
            best_errors, best_threshold = errors, score
    return float(best_errors / len(frames)), float(best_threshold)


def confusion_percentages(confusion):
    """Row percentages; unrepresented rows are NaN (undefined, not perfect)."""
    totals = confusion.sum(axis=1, keepdims=True)
    return np.divide(100.0 * confusion, totals, out=np.full(confusion.shape, np.nan), where=totals > 0)


def confusion_figure(confusion: np.ndarray, title: str = "Piece confusion"):
    """Heatmap with rows normalized to the share of each annotated class, raw counts in the cells.

    The bottom row (an empty square read as a piece) holds false positives; the right column
    (an annotated piece with no reading) holds false negatives.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = PIECE_SYMBOLS + ["empty"]
    totals = confusion.sum(axis=1, keepdims=True)
    percentages = confusion_percentages(confusion)
    shares = percentages / 100.0
    figure, axes = plt.subplots(figsize=(9, 8))
    axes.imshow(shares, cmap="viridis", vmin=0, vmax=1)
    for row in range(len(names)):
        for column in range(len(names)):
            if confusion[row, column]:
                colour = "black" if shares[row, column] > 0.5 else "white"
                axes.text(column, row, f"{confusion[row, column]}\n({percentages[row, column]:.1f}%)", ha="center", va="center", fontsize=7, color=colour)
    axes.set_xticks(range(len(names)), names)
    axes.set_yticks(range(len(names)), [f"{n} ({int(t)})" for n, t in zip(names, totals[:, 0])])
    axes.set_xlabel("Read by the app pipeline")
    axes.set_ylabel("Annotated (squares)")
    axes.set_title(f"{title}: count and row percentage (including empty->empty)")
    figure.tight_layout()
    return figure
