# Dev Log — 2026-03-08 — Baseline Log-Mel CNN v20a1 (Accuracy Improvements)

## Scope

This log covers the creation of `baseline_logmel_cnn_v20a1.py` — a direct upgrade of
`baseline_logmel_cnn_v20a.py` targeting improved classification accuracy on FMA Medium.
Nine changes were applied across configuration, architecture, and training strategy.

1. Reduced `N_MELS` from 256 to 128.
2. Increased `BATCH_SIZE` from 16 to 32.
3. SpecAugment parameters unchanged from v20a (15/25/2).
4. Separate cache namespace (`logmel_v20a1_10s`).
5. Deeper 5-block CNN architecture (32→64→128→256→256 filters).
6. Proper Conv→BN→ReLU ordering throughout.
7. `SpatialDropout2D(0.10)` after blocks 2 and 3; head `Dropout(0.2)`.
8. Cosine annealing LR schedule with 3-epoch linear warmup.
9. `EarlyStopping` re-enabled + `_MacroF1Checkpoint` added.

---

## Change 1 — Reduce N_MELS from 256 to 128

### Problem
`N_MELS=256` with `n_fft=512` produces a mel filterbank with only 257 unique FFT bins,
so 256 mel bands provides almost no frequency smoothing — the representaton is nearly a
1:1 mapping from FFT bins to mel bands.  This preserves high-frequency noise and makes
the model optimise over redundant frequency detail.

### Fix
```python
# v20a
N_MELS = 256

# v20a1
N_MELS = 128
```

Downstream impact: `LOGMEL_SHAPE` changes from `(256, 861)` to `(128, 861)`.  The
cache namespace is changed (see Change 4) so old and new `.npy` files do not collide.
Halving the mel dimension also reduces the per-forward-pass computation noticeably.

---

## Change 2 — Increase BATCH_SIZE from 16 to 32

### Problem
A batch size of 16 on FMA Medium (~12,500 training tracks) produces ~781 gradient
updates per epoch — each computed from a very small sample.  This introduces high
gradient variance and makes learning curves noisier.

### Fix
```python
# v20a
BATCH_SIZE = 16

# v20a1
BATCH_SIZE = 32
```

Larger batches provide more stable gradient estimates, which is especially beneficial
when used together with a cosine annealing schedule (Change 8).

---

## Change 3 — SpecAugment Parameters (Unchanged)

SpecAugment parameters were kept at v20a values.  Although an increase was originally
planned, it was not applied in v20a1.

```python
# v20a and v20a1 — identical
SPEC_AUG_FREQ_MASK  = 15
SPEC_AUG_TIME_MASK  = 25
SPEC_AUG_NUM_MASKS  = 2
```

The `spec_augment()` function was improved with dimension clamping to prevent crashes
if mask sizes ever exceed the actual tensor dimensions:

```python
f_max = tf.minimum(freq_mask, freq_dim)
f  = tf.random.uniform([], 0, f_max + 1, dtype=tf.int32)
f0 = tf.random.uniform([], 0, tf.maximum(freq_dim - f, 1), dtype=tf.int32)
```

---

## Change 4 — New Cache Namespace

### Problem
`baseline_logmel_cnn_v20a.py` writes `.npy` files of shape `(256, 861)`.
`baseline_logmel_cnn_v20a1.py` targets `(128, 861)`.  Sharing a cache directory would
cause silent shape-mismatch errors during index validation.

### Fix
```python
# v20a
LOGMEL_CACHE_DIR = CACHE_DIR / "logmel_v20a_10s" / ...
index_path = CACHE_DIR / f"logmel_v20a_10s_index_{split_name}_{SUBSET}.parquet"

# v20a1
LOGMEL_CACHE_DIR = CACHE_DIR / "logmel_v20a1_10s" / ...
index_path = CACHE_DIR / f"logmel_v20a1_10s_index_{split_name}_{SUBSET}.parquet"
```

---

## Change 5 — Deeper 5-Block CNN Architecture

### Problem
The v20a model has 4 convolutional blocks topping out at 128 filters, ending with a
128-dim GAP vector for a 16-class problem.  With ~12,500 training tracks this leaves
capacity headroom that is not utilised.

### Fix
A fifth convolutional block was added and filter counts in the upper blocks were
doubled:

| Block | v20a filters | v20a1 filters |
|-------|-------------|--------------|
| 1     | 32          | 32           |
| 2     | 64          | 64           |
| 3     | 128         | 128          |
| 4     | 128         | 256          |
| 5     | —           | 256          |

The GAP vector grows from 128-dim to 256-dim before the final Dense layer.

```python
# v20a1 — blocks 4 and 5 (new)
x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4")(x)
x = layers.BatchNormalization(name="bn4")(x)
x = layers.ReLU(name="relu4")(x)
x = layers.MaxPool2D((2, 2), name="pool4")(x)

x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv5")(x)
x = layers.BatchNormalization(name="bn5")(x)
x = layers.ReLU(name="relu5")(x)
x = layers.MaxPool2D((2, 2), name="pool5")(x)
```

---

## Change 6 — Proper Conv→BN→ReLU Ordering

### Problem
In v20a the activation was fused into `Conv2D(activation="relu")`, which means the
actual execution order was:

```
Conv2D(ReLU)  →  BatchNormalization
```

Batch normalization is intended to normalise **pre-activation** values so that the
distribution entering each activation is controlled.  Applying ReLU before BN discards
the negative half of the distribution before normalisation, reducing BN's effectiveness.

### Fix
All convolutional layers now use `activation=None` (default, `use_bias=False`) and are
followed by an explicit `BatchNormalization` then `ReLU`:

```python
# v20a (incorrect ordering)
x = layers.Conv2D(32, (5,5), padding="same", activation="relu", name="conv1")(inputs)
x = layers.BatchNormalization(name="bn1")(x)

# v20a1 (correct ordering)
x = layers.Conv2D(32, (5,5), padding="same", use_bias=False, name="conv1")(inputs)
x = layers.BatchNormalization(name="bn1")(x)
x = layers.ReLU(name="relu1")(x)
```

`use_bias=False` is set on all Conv2D layers because the immediately-following
BatchNormalization layer already includes a trainable bias term (`beta`), making the
Conv2D bias redundant.

---

## Change 7 — SpatialDropout2D After Blocks 2 and 3

### Problem
The only dropout in v20a is a single `Dropout(0.3)` after GAP, which regularises the
flattened feature vector.  Intermediate feature maps from blocks 2 and 3 — where most
of the discriminative patterns are learned — receive no stochastic regularisation.

### Fix
`SpatialDropout2D(0.10)` is inserted after the ReLU of blocks 2 and 3, before pooling.
Spatial dropout drops entire feature-map channels rather than individual values, which
is more appropriate for correlated spatial data like spectrograms.

```python
# Block 2 (same pattern for block 3)
x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
x = layers.BatchNormalization(name="bn2")(x)
x = layers.ReLU(name="relu2")(x)
x = layers.SpatialDropout2D(0.10, name="sdrop2")(x)   # new
x = layers.MaxPool2D((2, 2), name="pool2")(x)
```

The head dropout was also reduced from 0.3 to 0.2:

```python
x = layers.Dropout(0.2, name="dropout")(x)
```

---

## Change 8 — Cosine Annealing LR Schedule with Warmup

### Problem
`ReduceLROnPlateau` is reactive — it waits for validation loss to plateau before
reducing the learning rate.  This can lock the optimizer into a staged staircase LR
pattern that may skip over better regions of the loss landscape, particularly during the
early epochs when gradient estimates are noisiest.

### Fix
A custom `CosineAnnealingWithWarmup` schedule is introduced:

- **Warmup phase** (epochs 0–2): LR rises linearly from `lr_min=1e-6` to `lr_max=1e-3`
- **Cosine decay phase** (epochs 3–59): LR follows a half-cosine decay back to `lr_min`

The schedule is step-based (not epoch-based) so it integrates cleanly with AdamW:

```python
class CosineAnnealingWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = lr_min + (lr_max - lr_min) * (step / warmup_steps)
        progress  = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(π * progress))
        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)
```

`ReduceLROnPlateau` is removed entirely.  The `_LRLogger` callback was updated to read
the LR from the schedule object rather than the deprecated `optimizer.lr` attribute.

---

## Change 9 — EarlyStopping Re-enabled + _MacroF1Checkpoint Added

### Problem
In v20a `EarlyStopping` was commented out with the note
`"Temporally stop early stopping regularization - do not remove this"`.
Without it, training always runs all 60 epochs and saves the *last* epoch's weights
regardless of whether overfit has occurred.  The best weights only exist in memory
during the `model.fit()` call — a kernel crash or disconnection loses them.

### Fix
Both callbacks are now active:

```python
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=9, restore_best_weights=True, verbose=1,
),
_MacroF1Checkpoint(
    val_ds=val_ds,
    genre_classes=GENRE_CLASSES,
    filepath=str(RUN_DIR / "best_model_macro_f1.keras"),
    check_freq=2,
),
```

`EarlyStopping` monitors val_loss as a stable stopping signal.
`_MacroF1Checkpoint` runs a separate validation pass every 2 epochs to compute
macro-F1 and saves the model when it improves.  Section 9 evaluation loads
`best_model_macro_f1.keras` so that all reported metrics are from the
primary-metric checkpoint, not the EarlyStopping-restored weights.

The final `model.save()` call at the end of the section still runs, capturing the
EarlyStopping-restored weights as `baseline_logmel_cnn_v20a1.keras` for reference.

---

## Files Changed

| File | Action |
|------|--------|
| `MelCNN-MGR/notebooks/baseline_logmel_cnn_v20a1.py` | Created (copied from v20a, all changes applied) |

---

## Post-creation review fixes (2026-03-08 ~22:30 UTC)

After code review, the following corrections and improvements were applied:

### Documentation / comment sync
- **Header prose**: Removed "stronger augmentation" (SpecAugment is unchanged at 15/25/2).
- **Changelog row 3**: "SpecAugment strengthened (27/50/3)" → "unchanged (15/25/2)".
- **Changelog row 7**: `SpatialDropout2D(0.15)` → `0.10` (matches code).
- **Changelog row 10**: "ModelCheckpoint" → "_MacroF1Checkpoint (val_macro_f1)".
- **Comparison table**: Fixed SpatialDropout, SpecAugment, and checkpoint cells.
- **Section 6 architecture table**: `SpatialDrop(0.15)` → `0.10`; `Dropout(0.3)` → `0.2`.
- **Config inline comments**: `(was 15 in v20a)` → `(unchanged from v20a)`.

### `_MacroF1Checkpoint` improvements
- Added `check_freq` parameter (default 1, set to 2) — reduces extra validation
  overhead by running the macro-F1 pass every 2 epochs instead of every epoch.
  On skipped epochs, `f1_history` repeats the last value to stay epoch-aligned.

### `_LRLogger` compatibility fix
- Replaced two-branch `opt.lr` / `opt.learning_rate.__call__` heuristic with a
  single path through `opt.learning_rate` — handles `LRSchedule`, `tf.Variable`,
  and scalar LR without relying on the deprecated `opt.lr` alias.

### `spec_augment` bounds clamping
- Clamped mask sizes: `tf.minimum(freq_mask, freq_dim)` and `tf.minimum(time_mask, time_dim)`.
- Clamped start positions: `tf.maximum(freq_dim - f, 1)` and `tf.maximum(time_dim - t, 1)`.
- Prevents `tf.random.uniform` crash if config mask sizes ever exceed tensor dimensions.

### Evaluation consistency (EarlyStopping vs macro-F1 model)
- Sections 9 and 10 now load `best_model_macro_f1.keras` into `_eval_model` before
  evaluation and inference, instead of using the in-memory model (which holds
  EarlyStopping-restored best-val_loss weights). This ensures all reported metrics
  and predictions come from the primary-metric checkpoint.

### Accuracy framing
- Softened "misleading under class weights" wording to "secondary — interpret
  alongside Macro-F1 and per-genre F1" across Section 7 print, Section 8 plot
  title, and Section 9 summary. Accuracy is valid but incomplete, not invalid.

---

## Further improvements (2026-03-09)

### `CosineAnnealingWithWarmup` — Keras serialization registration
- Added `@tf.keras.saving.register_keras_serializable(package="MelCNN")` decorator.
- Without this, `tf.keras.models.load_model()` on a saved checkpoint raised:
  `TypeError: Cannot deserialize object of type 'CosineAnnealingWithWarmup'`.
- Re-running training produces a loadable checkpoint; models saved before this fix
  cannot be loaded without also registering the class manually.

### Waveform cropping — random crop for train, center-crop for val/test
- `normalize_to_fixed_duration()` gains a `random_crop: bool = False` parameter.
- `_process_one_track()` passes `random_crop=(split_name == "train")`.
- Training tracks now receive a uniformly random 10-second window, exposing the
  model to varied temporal positions across cache rebuilds.
- Val/test tracks retain deterministic center-crop for reproducible evaluation.
- Config entry updated: `"random_crop_train__center_crop_val_test__center_pad_short"`.
- **Cache note**: the crop position is baked into `.npy` files; clearing the train
  split cache (parquet index + `.npy` files) is required to obtain new random crops.

### `_MacroF1Checkpoint` — three-phase frequency schedule
Previous two-phase (sparse/dense) replaced with three-phase:

| Epoch range | `check_freq` | Extra val passes |
|-------------|-------------|-----------------|
| 0 – 29      | every 3     | ~10             |
| 30 – 59     | every 2     | ~15             |
| 60+         | every 1     | all             |

New parameters: `mid_freq=2`, `mid_freq_from_epoch=30`, `dense_from_epoch=60`.
Default `check_freq` changed from 1 → 3 (sparse phase default).
Call-site updated with all four explicit kwargs for clarity.
