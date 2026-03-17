# Dev Log — 2026-03-17 — Stabilizing `logmel_cnn_v2_4_cuda_tf.py`

## Scope

This session applied the runtime and input-pipeline stabilization patches
documented in `docs/Development Guide-Stabilizing-logmel_cnn_v2_4_cuda_tf-2026-03-17.md`.

Updated artifacts:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_4_cuda_tf.py`
2. `dev-logs/2026-03-17-logmel-cnn-v2_4-stabilization.md`

## Summary

The `logmel_cnn_v2_4_cuda_tf.py` training entry point had two separate issues.

The first issue was functional: training consumed a finite `tf.data.Dataset`
without `.repeat()` and without explicit `steps_per_epoch`, so Keras could
exhaust the input pipeline and stop after the first full pass through the
training set.

The second issue was runtime-related: the script defined a helper that disables
TensorFlow's Grappler layout optimizer, but that helper was never actually
called. As a result, the runtime banner implied layout optimization was
disabled even though the process was still running with default behavior.

This session fixed the training-loop behavior first, then aligned the runtime
configuration and logging so the script now behaves the way it claims to.

## Change 1 — TensorFlow Runtime Configuration Is Now Applied

The script already included `_configure_tensorflow_runtime()` with:

1. `tf.config.optimizer.set_experimental_options({"layout_optimizer": False})`

But the call site was commented out, so the protection was never active.

The script now defines:

1. `LAYOUT_OPTIMIZER_DISABLED = True`

and calls `_configure_tensorflow_runtime()` before the early runtime banner is
printed.

This makes the layout-optimizer behavior explicit and ensures the script
actually applies the setting before model construction and training.

## Change 2 — Runtime Banner Now Reports Layout Status Truthfully

Previously the startup log always printed:

1. `Layout opt  : disabled`

regardless of whether the runtime configuration had been applied.

The banner now reflects the real state through the same
`LAYOUT_OPTIMIZER_DISABLED` flag, so startup output and behavior are aligned.

## Change 3 — Training Dataset Now Repeats Across Epochs

The finite training dataset construction was left intact for one clean pass of
preprocessing, shuffling, Mixup, and SpecAugment, but the dataset used for
`model.fit(...)` now becomes:

1. `train_fit_ds = train_ds.repeat()`

This keeps the training input available across epochs instead of allowing Keras
to run out of data after a single full traversal.

Validation and test datasets remain finite and deterministic.

## Change 4 — `model.fit(...)` Now Uses Explicit Epoch Step Counts

Once the training dataset repeats, epoch boundaries must be defined
explicitly.

The staged training call in `_run_training_stage(...)` now passes:

1. `steps_per_epoch=steps_per_epoch`
2. `validation_steps=_batch_count(len(val_df), BATCH_SIZE)`

This makes each epoch consume the intended number of training batches and keeps
validation consumption fixed as well.

## Change 5 — Dataset Cardinality Diagnostics Added

To make troubleshooting easier, the script now logs cardinality for:

1. `train_fit_ds`
2. `val_ds`

through a small helper that safely prints `.cardinality().numpy()` when
available.

This is mainly a debugging aid, but it is useful for quickly confirming that
the fit dataset is now repeated while validation remains finite.

## Why These Changes Matter

The main failure mode was not the CNN architecture, TFRecord format, or loss
configuration. It was the training-loop contract between `tf.data.Dataset` and
Keras.

With the updated behavior:

1. training no longer depends on accidental dataset exhaustion to define epoch
   boundaries
2. epoch length is stable and explicit
3. validation batch consumption is stable and explicit
4. TensorFlow layout-optimizer status is now both applied and reported
   correctly

That is a much safer base for evaluating any remaining runtime warnings,
including whether `SpatialDropout2D` participates in the layout-optimizer
warning path.

## Validation Status

Validated in this session:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_4_cuda_tf.py` passed static error
   checks after the patch
2. the script now repeats the training dataset used by `model.fit(...)`
3. the script now sets explicit `steps_per_epoch` and `validation_steps`
4. the runtime configuration helper is now actually invoked before the startup
   banner

Not yet validated end-to-end in this session:

1. a full training run confirming the previous `Your input ran out of data`
   interruption no longer appears
2. whether the Grappler layout warning fully disappears with the layout
   optimizer disabled
3. the optional diagnostic path with `DISABLE_SPATIAL_DROPOUT = True`