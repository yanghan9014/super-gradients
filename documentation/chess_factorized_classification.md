# Shared piece types and separate color prediction

New chess N/S/L recipes enable `factorized_piece_color`. Each FPN scale keeps the
existing classification convolution tower, with two independent 1x1 projections:

```text
classification features ──┬── cls_pred:   P, N, B, R, Q, K, board (7 logits)
                         └── color_pred: white, black           (2 logits)
```

The six types are shared by white and black pieces. Board detection has its own
seventh logit and no color. Box regression, keypoints, dataset labels and geometry
are unchanged. There is no comparison between pieces and no extra feature tower.

Type/board scores retain the existing independent sigmoid detection semantics:
all can be low on background. Color is a two-way softmax conditional on a piece.
This avoids adding a separate objectness head just to reject empty locations.
Inference composes `sigmoid(type_logit) * softmax(color_logits)[color]` and appends
`sigmoid(board_logit)`. Decoded and traced outputs retain the same four tensors and
13 score columns: `P,N,B,R,Q,K,p,n,b,r,q,k,board`. Metrics, postprocessing and app
class ordering therefore remain compatible. This conditional factorization
assumes type and color are independent given the shared features; it cannot
express an arbitrary 12-class distribution at a single anchor.

## Training

Task-aligned assignment uses the composed 13 scores, retaining the original label
and board geometry contract. After assignment, white and black targets of each
type are merged into six type targets, with the board as target seven. These use
the existing focal loss (or BCE when configured), including background negatives.
Color uses cross-entropy on assigned pieces only. Its targets carry the same soft
assignment quality; board and background have exactly zero color gradient. Both
terms use the existing total assigned-score normalization, including DDP handling.

`loss_cls = classification_loss_weight * (type_and_board_loss +
piece_color_loss_weight * color_loss)`. The color multiplier defaults to 1.0; no
extra color penalty is introduced in this first architecture experiment. Existing
loss diagnostics include the color term in `piece_loss_cls`, with additional
`piece_loss_type` and `piece_loss_color` contributions when diagnostics are enabled.
Their values include configured weights. Geometry loss weights are unchanged.

The color bias starts neutral. Type detection biases use twice the old per-class
prior, so composition with neutral color retains the old colored-piece bias-only
prior; the board bias is unchanged. These are initialization priors, not calibrated
probabilities or a guarantee that old confidence thresholds remain optimal.

## Configuration and checkpoints

The flag lives at `arch_params.heads.ChessYoloNASPoseNDFLHeads.factorized_piece_color`.
The shared chess loss recipe interpolates it, keeping head and loss synchronized.
For a new legacy comparison run, set that architecture flag to `false`; both
constructor defaults are also `False`, so saved legacy resolved configurations
continue to reconstruct the original head and loss. Mismatched head/loss modes
raise an error.

The factorized head has different parameters (seven `cls_pred` outputs plus two
`color_pred` outputs). Start a new run with the usual backbone-only initialization.
Do not resume or strictly load a legacy full-model checkpoint into this head. Old
checkpoints must use their saved legacy architecture configuration; there is no
automatic conversion of their classifier weights. Resume factorized checkpoints
with their saved factorized configuration and this implementation. The external
`num_classes` remains 13; annotations must not be relabeled to seven classes.

Export keeps the score tensor contract through simple sigmoid, softmax and multiply
operations. CPU tracing tests cover that composition; an actual trained CoreML
export and device parity run remain necessary before deploying a new model.
Confidence calibration and recognition improvement require new training and the
existing ChessVision post-training evaluation; this code change proves neither.

## Focused verification

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m pytest -q \
  tests/unit_tests/chess_factorized_color_test.py \
  tests/unit_tests/chess_board_supervision_test.py \
  tests/unit_tests/chess_conditional_rotation_test.py \
  tests/unit_tests/chess_piece_metrics_test.py
```
