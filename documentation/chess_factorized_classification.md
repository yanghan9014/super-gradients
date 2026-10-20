# Piece presence, type and color

New chess N/S/L recipes enable both `factorized_piece_color` and `piece_objectness`.
The existing classification tower has three projections: `cls_pred` contains six
piece-type logits plus the unchanged board logit, `color_pred` contains two color
logits, and `objectness_pred` contains one piece-presence logit. Types use a six-way
softmax, colors a two-way softmax, and presence a sigmoid. The type order is
B,K,N,P,Q,R; color order is white,black.

For each anchor, `s(type,color) = objectness * P(type) * P(color)`. The twelve
piece scores sum to objectness. Board remains an independent sigmoid, with its
existing focal/BCE target, loss weight, normalization, geometry and postprocessing.
The joint score vector remains available for assignment and class labels.

## Supervision

Assignment still uses the composed 13 scores. Assigned piece labels provide hard
objectness positives, including low-quality assignments. Other candidates are
negatives; crowd candidates are ignored. Objectness uses BCEWithLogits. Type and
color use hard-label cross-entropy only on assigned pieces. All three piece losses
are summed and normalized by max(number of assigned pieces, 1), using the existing
DDP mean convention. Color retains `piece_color_loss_weight`. Board keeps the old
quality-weighted targets and total-quality normalization. No geometry weights,
training schedule, dataset labels, or resume rules change.

Separate diagnostics report piece objectness, type and color loss. Type/color
biases start neutral; objectness starts at 0.01; board initialization is unchanged.

## Output and tracker contract

Training/evaluation return `(decoded, raw)`. Raw outputs append color logits and
then objectness logits to the original eight tensors. Decoded/export outputs are:

1. boxes `[B,A,4]`
2. joint colored-piece/board scores `[B,A,13]`
3. keypoint coordinates `[B,A,9,2]`
4. keypoint confidence `[B,A,9]`
5. piece objectness `[B,A,1]`
6. conditional type probabilities `[B,A,6]`
7. conditional color probabilities `[B,A,2]`

Piece thresholding and candidate ranking use objectness. Board thresholding stays
unchanged. The separate tensors survive candidate selection, square mapping, the
tracker input and recording as `piece_classifications`. Each square has objectness,
all six type probabilities and both color probabilities. Display labels remain
argmax summaries. The current tracker retains these full outputs in three
independent evidence vectors: presence `[empty, occupied]`, six types, and two
colors. Each vector has its own evidence cap. Objectness supplies presence
evidence and weights incoming type/color evidence. Type and color disagreement
use their respective probability-derived evidence; neither uses only the hard
display label. Evidence changes smoothly over elapsed time without resetting
all three vectors when one label changes.

The exporter embeds `piece_objectness_separate=1` as well as
`piece_color_factorized=1`. This identifies the seven-output contract explicitly;
legacy exports with extra internal tensors still use their existing keypoint slot.
New models require a new training run and an updated client before promotion.
No automatic checkpoint conversion or app model promotion is performed.

## Validation score semantics

`Piece_mAP` matches colored piece classes by board square, using annotated board
geometry and no confidence cutoff. With separate objectness outputs,
`square_readings` selects the highest-objectness anchor per square and uses that
objectness as the score for the class-specific precision-recall curve. Its class
label still comes from the maximum joint type/color score.

Earlier models instead selected square winners and ranked predictions by the
maximum joint colored-piece score. Therefore an mAP comparison across this change
mixes changes in the model with changes in candidate selection and ranking. High
objectness does not imply high confidence in the selected type or color. A
controlled comparison must hold those scoring rules fixed; a lower mAP alone
cannot isolate which change caused it. The end-to-end incorrect-squares metrics
also use the new objectness-based square winners and tune the piece threshold.
Tracker inputs keep all three outputs separately regardless of metric scoring.

## Compatibility

Constructor defaults for both flags are false. Saved legacy configurations retain
13 independent sigmoids; prior factorized configurations with only
`factorized_piece_color=true` retain six type sigmoids and their color softmax.
Set `piece_objectness=false` to reproduce that earlier architecture. Set both flags
false for the original classifier. Each model must use its own resolved config;
strict checkpoint loading prevents accidental reinterpretation. Old exports keep
four decoded outputs. The current tracker's legacy adapter can consume them using
confidence as an occupancy proxy, hard type labels, and available color
probabilities (otherwise hard colors). This compatibility path cannot recover
independent objectness or a full type distribution that an old model did not emit.

The earlier factorized-color design is documented below for saved runs.

---

# Shared piece types and separate color prediction

New chess N/S/L recipes enable `factorized_piece_color`. Each FPN scale keeps the
existing classification convolution tower, with two independent 1x1 projections:

```text
classification features ──┬── cls_pred:   B, K, N, P, Q, R, board (7 logits)
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
13 score columns: `B,K,N,P,Q,R,b,k,n,p,q,r,board`. Metrics, postprocessing and app
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

Export preparation switches a factorized head to the four decoded tensors, so
`torch.export` does not expose training-only logits. Legacy export preparation is
unchanged. Export keeps the score tensor contract through simple sigmoid, softmax
and multiply operations. CPU tracing and full-model `torch.export` tests cover
that composition; an actual trained CoreML
export and device parity run remain necessary before deploying a new model.
The ChessVision exporter embeds `piece_color_factorized=1` as an ExecuTorch
constant method. The app can then recover the color softmax from the existing
scores (sum of white scores divided by sum of all piece scores) without changing
the forward tensors. Exports without that metadata retain legacy tracker behavior.
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

The upstream `tests.unit_tests` initializer eagerly imports unrelated optional
data-adapter/PDF tests. If their dependencies are unavailable, collect these
files with namespace-only `tests` / `tests.unit_tests` packages; production imports
and the selected tests must still run unchanged. No training-environment dependency
changes are needed for the focused suite.
