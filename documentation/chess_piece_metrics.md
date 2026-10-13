# Chess piece validation metrics

These metrics read raw model output without NMS. Each square keeps the highest-scoring
piece candidate (class argmax over the 12 piece classes); equal scores keep the earlier
anchor. Pieces use their first keypoint for placement. Frames without a usable annotated
board are excluded from piece evaluation. Overall AP/AR and Board_* metrics are unchanged.

| Output | Definition and purpose |
| --- | --- |
| `Piece_mAP` | Unweighted mean of the per-class AP values: compare piece recognition across models without choosing a confidence cutoff. |
| `Piece_AP/<color_type>` | Exact class **and** square correctness using the annotated board homography. No piece-score prefilter. Predictions are ranked across the validation set separately per class. Equal scores enter together. Precision is interpolated at the configured recall levels (default: 101 points from 0 to 1). Missing pieces stay in the recall denominator. |
| `Piece_Color_Accuracy` | Fraction with correct color among annotated occupied squares having a predicted reading in annotated board geometry. No confidence cutoff; type is ignored. Missing readings are excluded from this conditional statistic, but remain misses in AP. |
| `Piece_EndToEnd_Incorrect_Squares` | Minimum mean incorrect squares per frame over a single global, validation-tuned piece threshold. Uses the predicted board. Missing, extra, or wrong-class readings each count once per square. Lower is better. |
| `Piece_EndToEnd_Threshold` | Selected piece threshold, not a quality metric. Board confidence and keypoint thresholds remain fixed. Ties in error choose the highest threshold. |

AP and color accuracy are fractions in [0, 1]. AP is -1 when a class has no annotated
examples; such classes are excluded from mAP rather than treated as perfect or failed.
All 12 classes contribute equally when represented. Color accuracy is -1 when there are
no occupied squares with readings. End-to-end outputs are -1 when no frames are scorable.
The old aggregate `Piece_AP` and `Piece_AR` outputs are removed.

The end-to-end sweep caches only square winners and annotations, at most 64 readings per
frame. It sorts distinct scores and updates error counts, O(N log N), with no additional
model passes. Acceptance uses score >= threshold, including tied scores together. The
search is over [0, 1]; a score of exactly 1 cannot be rejected. Failed predicted-board
extraction produces an empty reading and is included as missing pieces. The reported
result is validation-tuned, not a held-out test estimate. Freeze the selected threshold
before evaluating a separate test set. Existing checkpoint selection is unchanged.

## Existing fixed-cutoff diagnostics

The confusion matrix and FP/FN breakdowns still use the **predicted** board and the
configured `piece_confidence_threshold` (default 0.05), independent of the tuned score.

- `Valid_Piece_FP_Percent/<class>`: empty squares read as that class / all annotated empty squares, times 100.
- `Valid_Piece_FN_Percent/<class>`: annotated pieces of that class read as empty / all annotated pieces of that class, times 100.
- Undefined percentage denominators are not logged. Wrong-class substitutions appear in
  the confusion matrix, not in these empty/piece breakdowns.
- `Valid_Piece_Confusion` retains counts and adds row percentages beneath each nonzero
  count. Correctly empty squares now contribute to the empty row and its denominator.

The explicit `_Percent` tag names distinguish these rates from historical raw-count logs.
AP/color and end-to-end/confusion use different board geometry intentionally; they answer
recognition and application-performance questions respectively.
