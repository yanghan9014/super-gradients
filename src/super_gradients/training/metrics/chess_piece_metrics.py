"""Piece metrics computed the way the iOS app reads a single frame.

This mirrors ChessVision_app (`YoloRunner.swift` decode, `PostProcessor.extractBoardState`):
no NMS, one decoupled confidence threshold for pieces and boards, the most confident board,
a homography from its landmarks that clear the keypoint threshold, then the most confident
piece on each square. The annotation is the ground truth: its pieces are placed on squares
with a homography fitted the same way to the annotated landmarks.
"""

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
) -> Dict[int, Tuple[int, float]]:
    """One frame as the app reads it: {square: (piece class, confidence)}.

    Inputs are the raw per-anchor model outputs in model-input pixels: class scores [A, 13],
    keypoints [A, 9, 2] and keypoint scores [A, 9]. An empty result means the app would have
    no board state for this frame (no board, fewer than four landmarks, or no pieces on it).
    """
    board_scores = class_scores[:, BOARD_CLASS_ID]
    boards = np.flatnonzero(board_scores >= confidence_threshold)
    if len(boards) == 0:
        return {}
    best = boards[np.argmax(board_scores[boards])]
    homography = _board_homography(pose_coords[best], pose_scores[best] >= board_keypoint_threshold, image_size)
    if homography is None:
        return {}
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


def annotated_squares(joints: np.ndarray, labels: np.ndarray, image_size: Tuple[int, int]) -> Optional[Dict[int, int]]:
    """{square: piece class} from an annotation, or None when it has no usable board.

    The first annotated board with at least four visible landmarks defines the homography.
    """
    for board in np.flatnonzero(labels == BOARD_CLASS_ID):
        homography = _board_homography(joints[board, :, :2], joints[board, :, 2] > 0, image_size)
        if homography is None:
            continue
        pieces = [i for i in range(len(labels)) if labels[i] < NUM_PIECE_CLASSES and joints[i, 0, 2] > 0]
        squares = _place(joints[pieces, 0, :2], homography, image_size) if pieces else []
        return {square: int(labels[i]) for i, square in zip(pieces, squares) if square is not None}
    return None


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
    for square in set(readings) | set(truth):
        confusion[truth.get(square, EMPTY), readings[square][0] if square in readings else EMPTY] += 1
    return per_class, confusion


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
    shares = np.divide(confusion, totals, out=np.zeros(confusion.shape, dtype=np.float64), where=totals > 0)
    figure, axes = plt.subplots(figsize=(9, 8))
    axes.imshow(shares, cmap="viridis", vmin=0, vmax=1)
    for row in range(len(names)):
        for column in range(len(names)):
            if confusion[row, column]:
                colour = "black" if shares[row, column] > 0.5 else "white"
                axes.text(column, row, str(confusion[row, column]), ha="center", va="center", fontsize=8, color=colour)
    axes.set_xticks(range(len(names)), names)
    axes.set_yticks(range(len(names)), [f"{n} ({int(t)})" for n, t in zip(names, totals[:, 0])])
    axes.set_xlabel("Read by the app pipeline")
    axes.set_ylabel("Annotated (squares)")
    axes.set_title(f"{title}: row share, cell = count; empty->empty not counted")
    figure.tight_layout()
    return figure
