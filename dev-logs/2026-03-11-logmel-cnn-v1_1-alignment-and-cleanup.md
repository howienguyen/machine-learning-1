# Dev Log — 2026-03-11 — Log-Mel CNN v1.1 Alignment and Cleanup

## Scope

This log covers the review, documentation, implementation, and final cleanup of
`MelCNN-MGR/model_training/logmel_cnn_v1_1.py`, which was created as the next training
candidate after analyzing the first full `logmel_cnn_v1.py` run.

Covered changes:

1. Analyzed the original v1 training run and documented failure modes.
2. Created an analysis report with proposed v1.1 remedies.
3. Created `logmel_cnn_v1_1.py` as an improved training script.
4. Fixed Mixup and class-weight compatibility in the training pipeline.
5. Aligned training control with Macro-F1 instead of `val_loss`.
6. Refined augmentation behavior: per-sample Mixup, Mixup before SpecAugment, no label smoothing.
7. Cleaned up train-metric reporting so the accuracy plot is interpretable.
8. Updated run-report metadata to reflect the actual v1.1 behavior.

---

## Change 1 — Post-Run Analysis of `logmel_cnn_v1.py`

### Problem

The first full v1 run achieved usable but clearly limited performance:

- test accuracy: about 58%
- test Macro-F1: about 0.58
- strong train/test gap, indicating overfitting
- severe per-genre imbalance in behavior, especially Metal over-prediction and weak Pop/Rock recall

### Main findings

The review identified four dominant problems:

1. regularization was still too weak for the dataset size and class ambiguity
2. Metal class weighting was too aggressive
3. the training loop selected by `val_loss` while the actual target metric was Macro-F1
4. several genre groups remained heavily confusable in log-mel space over 10-second clips

### Output

A dedicated report was created:

- `docs/Analysis_logmel_cnn_v1_and_v1_1_Improvements.md`

That report became the design basis for v1.1.

---

## Change 2 — Created `logmel_cnn_v1_1.py`

### Purpose

`MelCNN-MGR/model_training/logmel_cnn_v1_1.py` was created as a direct successor to
`logmel_cnn_v1.py`, preserving the same dataset and overall architecture family while
testing a more aggressive regularization and training-control recipe.

### Initial v1.1 changes

The first v1.1 draft introduced:

- stronger SpecAugment
- class-weight capping
- more dropout in deeper conv blocks
- higher weight decay
- a small dense bottleneck in the classifier head
- lower learning-rate ceiling and longer warmup
- Mixup augmentation

This was directionally correct, but a later review found that parts of the training
logic were still internally inconsistent.

---

## Change 3 — Fixed Mixup and Class-Weight Compatibility

### Problem

The initial v1.1 implementation used Keras `class_weight` together with Mixup-softened
labels. That combination broke during `model.fit(...)` because Keras' internal
class-weight adapter expected ordinary label rank information and is not a good fit for
soft mixed labels.

### Fix

The training pipeline now computes explicit per-example sample weights after Mixup:

- class weights are still derived from capped per-class values
- `attach_class_weights(...)` converts mixed one-hot labels into sample weights
- training uses a dedicated `train_fit_ds` dataset yielding `(x, y, sample_weight)`

### Why this matters

This makes class reweighting compatible with Mixup while preserving the intended capped
imbalance correction.

---

## Change 4 — Training Control Is Now Truly Macro-F1-Driven

### Old issue

The earlier version still monitored `val_loss` for EarlyStopping even though the real
selection objective was Macro-F1.

### Fix

The training loop was reworked so Macro-F1 is now the real control metric:

- `_ValMacroF1` computes validation Macro-F1 every epoch
- `logs["val_macro_f1"]` is populated every epoch
- `ModelCheckpoint` monitors `val_macro_f1` in `mode="max"`
- `EarlyStopping` also monitors `val_macro_f1` in `mode="max"`

### Result

The final saved checkpoint, the stopping logic, and the stated objective are now aligned.

---

## Change 5 — Augmentation Logic Was Brought Into Line With the Design

### Review finding

The first v1.1 draft did not fully match the design intent around Mixup and label
softening.

### Fixes applied

1. `LABEL_SMOOTHING` was set to `0.0`
2. Mixup now uses one $\lambda$ per sample pair rather than one scalar per batch
3. augmentation order was changed to:

   - normalize
   - batch
   - Mixup
   - SpecAugment

### Why this matters

These changes avoid over-softening supervision and make the augmentation pipeline closer
to a standard, defensible audio-classification recipe.

---

## Change 6 — Clean Train Evaluation Was Separated From Fit-Time Augmentation

### Problem

Using an augmented Mixup training dataset for train-set reporting makes the resulting
"train accuracy" hard to interpret.

### Fix

The pipeline now keeps separate dataset roles:

- `train_ds` — shuffled, augmented, Mixup-enabled training data
- `train_fit_ds` — weighted version of `train_ds` for `model.fit(...)`
- `train_eval_ds` — clean, non-augmented train data for reporting and callbacks

### Result

Final train metrics are now measured on real clean examples rather than on synthetic
mixed-label batches.

---

## Change 7 — Accuracy Plot Cleanup

### Problem

The original training-history panel still plotted `hist["accuracy"]`, which came from the
fit-time augmented pipeline and was therefore not a reliable train-accuracy signal.

### Fix

A new callback was added:

- `_TrainEvalAccuracy(train_eval_ds)`

This callback evaluates clean train accuracy at the end of each epoch and stores:

- `train_eval_accuracy`
- `train_eval_accuracy_per_epoch`

The accuracy plot now shows:

- `Train (clean eval)`
- `Validation`

instead of the misleading Mixup-based training accuracy trace.

### Why this matters

The curves are now more interpretable when checking for overfitting or underfitting.

---

## Change 8 — Run Report Metadata Cleanup

### Problem

After the callback refactor, the run-report metadata still contained a stale string for
early stopping patience.

### Fix

The metadata now correctly records:

- `early_stopping_patience`: `9 -> 10`

The report also now records:

- `train_eval_accuracy_per_epoch`
- the Macro-F1-driven training-control shift
- label-smoothing removal under Mixup
- augmentation ordering

### Result

The run report is now a more trustworthy record of the actual experiment configuration.

---

## Final State of `logmel_cnn_v1_1.py`

At the end of this work, the script now has the following properties:

- Macro-F1 is the true checkpointing and early-stopping metric
- Mixup uses per-sample coefficients
- label smoothing is disabled
- Mixup happens before SpecAugment
- capped class weights are applied via sample weights, not Keras `class_weight`
- clean train evaluation is separated from fit-time augmentation
- training-history accuracy plots are based on clean train evaluation
- run-report metadata reflects the actual configuration

This does not guarantee that v1.1 will outperform v1, but it does mean the experiment is
now internally coherent and the resulting metrics should be much easier to interpret.

---

## Change 9 — Added `inference_logmel_cnn_v1_1.py` and Example CLI

### Purpose

After the v1.1 training script was stabilized, a dedicated inference module was added so
the best v1.1 checkpoint can be reused without reopening the training script.

### New files

- `MelCNN-MGR/inference_logmel_cnn_v1_1.py`
- `MelCNN-MGR/examples/inference_logmel_cnn_v1_1_example.py`

### Inference module behavior

The new module:

- prefers `best_model_macro_f1.keras`
- falls back to `logmel_cnn_v1_1.keras` if needed
- loads `norm_stats.npz` for normalization
- registers `CosineAnnealingWithWarmup` so saved models deserialize correctly
- supports both `single_crop` and `three_crop` inference

### Example CLI behavior

The example script supports:

- explicit `--files ...`
- random selection from a test manifest via `--subset ... --random N`
- `single_crop` or `three_crop` inference modes

### Why this matters

This completes the v1.1 workflow end to end:

- training
- best-checkpoint selection
- evaluation
- reusable inference on unseen audio
