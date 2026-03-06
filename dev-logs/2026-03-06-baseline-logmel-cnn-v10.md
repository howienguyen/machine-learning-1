# Dev Log — 2026-03-06 — baseline_logmel_cnn_v10.ipynb

## Scope

This log documents the evolution of the Log-Mel CNN baseline from
`baseline_logmel_cnn_v1.ipynb` to `baseline_logmel_cnn_v10.ipynb`.

**Root cause analysis reference:** `docs/Conv1 Kernel Design Issue in Log-Mel Baseline.md`

The v10 notebook fixes a fundamental architectural mismatch in v1 where the first
convolution layer was incorrectly designed for a log-mel spectrogram input.
All data infrastructure (extraction, caching, preprocessing) is unchanged.

---

## Background — What v1 Got Wrong

`baseline_logmel_cnn_v1.ipynb` was deliberately designed as a controlled counterpart
to `baseline_mfcc_cnn_v5.ipynb`. The goal was to isolate the effect of input
representation by keeping the CNN architecture identical between the two notebooks.
This meant applying the same kernel rule `(freq_bins, 10)` to both:

| Notebook | Input shape | Conv1 kernel |
|----------|------------|-------------|
| `baseline_mfcc_cnn_v5.ipynb` | (13, 2582, 1) | (13, 10) |
| `baseline_logmel_cnn_v1.ipynb` | (128, 2582, 1) | (128, 10) |

While this preserved **ablation fairness** (same architecture, different input), it
violated **representation fairness** — the architecture was inappropriate for the
structure of log-mel data.

### Why `(128, 10)` is wrong for log-mel

A log-mel spectrogram is a 2D time-frequency image. CNNs learn from such images
hierarchically: early layers detect local textures, later layers combine them into
global patterns. Using `Conv2D(3, (128, 10))` in the first layer collapses the entire
128-band frequency axis immediately, producing output shape `(1, 644, 3)`. From that
point the remaining Conv2 and Conv3 layers operated on a 1×T slice — the CNN never
learned local spectro-temporal patterns.

In contrast, the same `(13, 10)` kernel is appropriate for MFCCs because MFCC
coefficients are derived via a DCT — they are global compressed summaries of the
spectrum, not spatially local features. Spanning all 13 MFCC bins is not a problem.

This mismatch explains the counter-intuitive result observed in v1:

```
MFCC baseline accuracy > Log-Mel baseline accuracy
```

despite log-mel theoretically containing more information.

---

## Changes: v1 → v10

### Change 1 — Notebook header (Cell 1)

**v1:**
```
# FMA Baseline - Log-Mel -> 2D CNN (Version 1)
Goal-1 controlled counterpart to baseline_mfcc_cnn_v5.ipynb.
...
Same training recipe: SGD lr=1e-3, 20 epochs, batch size 16
```

**v10:**
```
# FMA Baseline - Log-Mel → 2D CNN (Version 10)
Representation-fair architecture for log-mel spectrograms.
Fixes the Conv1 kernel design issue identified in
docs/Conv1 Kernel Design Issue in Log-Mel Baseline.md.
...
```

Updated to describe the representation-fair design intent, list all key changes,
and reference the root cause analysis document.

---

### Change 2 — Training epochs (Section 2, Configuration)

| Parameter | v1 | v10 |
|-----------|-----|------|
| `EPOCHS` | `20` | `30` |

The deeper 4-block architecture and Adam optimizer benefit from more training
budget. EarlyStopping (patience=5) guards against overfitting — actual epochs
run may be fewer than 30.

---

### Change 3 — Model architecture (Section 6) ← CORE CHANGE

**v1 architecture:**
```python
inputs = keras.Input(shape=(128, 2582, 1))

x = layers.Conv2D(3,  (N_MELS, 10), strides=(1, 4), padding="valid", activation="relu")(inputs)
# output: (1, 644, 3)  ← frequency axis fully collapsed after layer 1
x = layers.Conv2D(15, (1, 10),      strides=(1, 4), padding="valid", activation="relu")(x)
x = layers.Conv2D(65, (1, 10),      strides=(1, 4), padding="valid", activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(n_classes, activation="softmax")(x)
```

Problems:
- `(128, 10)` kernel spans the entire frequency axis in Conv1 — 1,280 params/filter
- Frequency axis collapses to 1 after the very first layer
- No pooling — aggressive stride-only downsampling
- No BatchNorm or Dropout — no regularization
- `Flatten` before dense layer → parameter-heavy classifier head
- Model name: `logmel_2dcnn_baseline`

**v10 architecture:**
```python
inputs = keras.Input(shape=(128, 2582, 1))

# Block 1 — local spectro-temporal features
x = layers.Conv2D(32, (5, 5), padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D((2, 4))(x)          # → (64, 645, 32)

# Block 2 — mid-level motifs
x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D((2, 4))(x)          # → (32, 161, 64)

# Block 3 — higher-level patterns
x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D((2, 4))(x)          # → (16, 40, 128)

# Block 4 — global structure
x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPool2D((2, 2))(x)          # → (8, 20, 128)

# Classifier head
x = layers.GlobalAveragePooling2D()(x)   # → (128,)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(n_classes, activation="softmax")(x)
```

Key improvements:
- Conv1 uses `(5, 5)` kernel — 25 params/filter vs 1,280 in v1 (**~51× fewer per filter**)
- Frequency axis preserved across all 4 blocks → hierarchical local feature learning
- `MaxPool2D` for gradual spatial reduction instead of aggressive strides
- `BatchNormalization` after every conv block → stable training
- `GlobalAveragePooling2D` replaces `Flatten` → compact classifier, better generalization
- `Dropout(0.3)` before the dense layer → regularization
- Model name: `logmel_2dcnn_v10`

Architecture shape trace:

| Block | Layer | Output shape |
|-------|-------|-------------|
| Input | — | (128, 2582, 1) |
| Block 1 | Conv2D(32,(5,5)) + BN + MaxPool(2,4) | (64, 645, 32) |
| Block 2 | Conv2D(64,(3,3)) + BN + MaxPool(2,4) | (32, 161, 64) |
| Block 3 | Conv2D(128,(3,3)) + BN + MaxPool(2,4) | (16, 40, 128) |
| Block 4 | Conv2D(128,(3,3)) + BN + MaxPool(2,2) | (8, 20, 128) |
| Head | GAP + Dropout(0.3) + Dense(N_CLASSES) | (N_CLASSES,) |

---

### Change 4 — Optimizer (Section 7)

| Parameter | v1 | v10 |
|-----------|-----|------|
| Optimizer | `SGD(lr=1e-3)` | `Adam(lr=1e-3)` |

SGD with a fixed learning rate is appropriate for shallow architectures with stable
gradients. For a 4-block architecture with BatchNormalization, Adam converges more
reliably because its adaptive per-parameter learning rates handle the non-uniform
gradient scales that BatchNorm introduces.

---

### Change 5 — Training callbacks (Section 7)

**v1:** No callbacks — training ran for a fixed 20 epochs.

**v10:** Two callbacks added:

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1,
    ),
]
```

- **EarlyStopping** (`patience=5`): stops training if `val_loss` does not improve for
  5 consecutive epochs and restores the best weights automatically.
- **ReduceLROnPlateau** (`patience=3`, `factor=0.5`): halves the learning rate when
  `val_loss` plateaus for 3 epochs, allowing finer convergence before early stopping.

---

### Change 6 — Run artifact naming (Section 7)

| Artifact | v1 | v10 |
|----------|-----|------|
| Run directory | `models/logmel-cnn-{ts}/` | `models/logmel-cnn-v10-{ts}/` |
| Saved model | `baseline_logmel_cnn.keras` | `baseline_logmel_cnn_v10.keras` |
| Report `run_id` | `logmel-cnn-{ts}` | `logmel-cnn-v10-{ts}` |
| Report `version` field | _(absent)_ | `"v10"` |
| Report `optimizer` field | `"SGD"` | `"Adam"` |
| Report `epochs_max` / `epochs_actual` | `epochs` (single field) | split into `epochs_max` + `epochs_actual` |
| Callback config in report | _(absent)_ | `early_stopping` + `reduce_lr_on_plateau` blocks |

---

### Change 7 — Plot and display titles

| Location | v1 | v10 |
|----------|-----|------|
| Training history `suptitle` | `"Training history -- Log-Mel 2D CNN baseline"` | `"Training history -- Log-Mel 2D CNN v10"` |
| Confusion matrix title | `"Confusion matrix -- test set (log-mel)"` | `"Confusion matrix -- test set (log-mel v10)"` |
| Inference bar chart title | `"Genre probabilities -- {name}"` | `"Genre probabilities (v10) -- {name}"` |

---

## Unchanged from v1

All of the following sections are carried over verbatim from `baseline_logmel_cnn_v1.ipynb`:

- **Imports** (Section 1) — same libraries
- **Device setup** — CUDA → Intel XPU → CPU fallback, smoke test
- **Configuration** (Section 2) — all parameters except `EPOCHS`; same SUBSET, SEED, audio params, cache dirs
- **Manifest loading** (Section 3) — same parquet splits, genre distribution plot
- **Log-mel extraction & caching** (Section 4) — same 128-band log-mel, `log1p` compression, `.npy` cache, corrupt-repair logic
- **Preprocessing** (Section 5) — same per-band z-normalization, same `tf.data` pipeline
- **Evaluation logic** (Section 9) — same `eval_dataset`, classification report, confusion matrix
- **Inference cell** (Section 10) — same single-sample inference
- **Runtime summary** — same timing display

---

## Expected Impact

| Metric | v1 (expected) | v10 (expected) |
|--------|--------------|----------------|
| Test accuracy | Below MFCC baseline | Above MFCC baseline |
| Macro-F1 | Degraded by global kernel | Improved by local features |
| Training stability | Unstable (large kernel, SGD) | Stable (BN + Adam + callbacks) |
| Overfitting | Possible (no regularization) | Reduced (BN + Dropout) |

The architectural fix is expected to produce results consistent with the broader
audio ML literature, where log-mel CNN models reliably outperform MFCC CNN models
when the architecture is properly designed for 2D spectrogram inputs.
