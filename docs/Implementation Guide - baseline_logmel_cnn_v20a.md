# Implementation Guide — `baseline_logmel_cnn_v20a.ipynb`

Fri Mar  6 15:02:34 UTC 2026

## 1. Overview

This document is the **implementation plan** for upgrading
`MelCNN-MGR/notebooks/baseline_logmel_cnn_v20.ipynb` into
`MelCNN-MGR/notebooks/baseline_logmel_cnn_v20a.ipynb`.

The rationale for each change is described in
`docs/Proposed Quality Improvement Plan for baseline_logmel_cnn_v20.ipynb.md`.
This document focuses exclusively on **what to change, where, and how**.

---

## 2. Source notebook structure (v20)

The v20 notebook has **30 cells** organized into these sections:

| Cell(s) | Section | Purpose |
|---------|---------|---------|
| 1–2 | Header + Changelog | Title, description, v10→v20 diff table |
| 3 | Section 1 | Imports |
| 4 | Device selection | CUDA / XPU / CPU runtime |
| 5–6 | Section 2 | Configuration & hyperparameters |
| 7–8 | Device setup | `configure_runtime_device()` |
| 9–11 | Section 3 | Load manifest splits + genre distribution plot |
| 12–13 | Section 4 header | Log-mel feature extraction docs |
| 14–15 | Section 4 code | Feature extraction + cache/index build |
| 16 | Section 4b | Log-mel sample plot |
| 17 | Section 5 header | Preprocessing header |
| 18 | Section 5 code | Label encoding, train stats, `make_dataset()`, tf.data pipeline |
| 19 | Section 6 header | Build model docs + shape table |
| 20 | Section 6 code | `build_model()` — the CNN architecture |
| 21 | Section 7 header | Compile & train header |
| 22 | Section 7 code | Compile, callbacks, `model.fit()`, run report |
| 23–24 | Section 8 | Training history plot |
| 25–26 | Section 9 | `eval_dataset()`, confusion matrix |
| 27 | Section 10 header | Inference header |
| 28 | Section 10 code | Single-sample inference |
| 29–30 | Runtime summary | Timing table |

---

## 3. Summary of changes

Seven changes, grouped by the notebook section they affect:

| # | Change | v20 section affected | Priority |
|---|--------|---------------------|----------|
| 1 | Keep 10s fixed-length | None (preserve) | Foundation |
| 2 | Class weights | Section 5, Section 7 | Very High |
| 3 | Milder time pooling | Section 6 | High |
| 4 | SpecAugment | Section 5 | High |
| 5 | AdamW optimizer | Section 7 | Medium |
| 6 | Label smoothing = 0.05 | Section 7 | Medium |
| 7 | 3-crop inference | Section 10 | High |

---

## 4. Cell-by-cell implementation plan

### 4.0. General: version renaming

Every cell that references `v20` in names, titles, file paths, or run IDs
must be updated to `v20a`. This includes:

- header markdown (cell 1): title, description
- changelog markdown (cell 2): new v20→v20a table
- configuration (cell 6): cache dir `logmel_v20a_10s`, index prefix `logmel_v20a_10s_index_*`
- model name: `logmel_2dcnn_v20a`
- run directory: `logmel-cnn-v20a-{ts}`
- model file: `baseline_logmel_cnn_v20a.keras`
- run report: `"version": "v20a"`
- plot titles: `v20a`

### 4.1. Cell 1 — Header (markdown)

Replace the title and description. Describe the notebook as an upgraded version
of v20 with class weights, milder pooling, SpecAugment, AdamW, label smoothing,
and 3-crop inference.

### 4.2. Cell 2 — Changelog (markdown)

Replace the v10→v20 changelog with a new **v20→v20a** changelog table:

| # | Section | What changed | Why |
|---|---------|-------------|-----|
| 1 | Config | Cache dir `logmel_v20_10s → logmel_v20a_10s` | Separate cache namespace |
| 2 | Config | Index parquets prefixed `logmel_v20a_10s_index_*` | Prevent collision with v20 index files |
| 3 | Config | Added `LABEL_SMOOTHING = 0.05` | Configurable label smoothing |
| 4 | Config | Added `WEIGHT_DECAY = 1e-4` | Configurable AdamW weight decay |
| 5 | Preprocessing (Section 5) | Added `spec_augment()` in training pipeline | SpecAugment for training-time robustness |
| 6 | Preprocessing (Section 5) | Added class-weight computation | Handle genre imbalance in training loss |
| 7 | Model (Section 6) | MaxPool changed from `(2,4),(2,4),(2,4),(2,2)` to `(2,2)` × 4 | Milder temporal pooling |
| 8 | Compile (Section 7) | `Adam → AdamW(weight_decay=1e-4)` | Decoupled weight decay regularization |
| 9 | Compile (Section 7) | Loss → `CategoricalCrossentropy(label_smoothing=0.05)` | Reduce overconfidence |
| 10 | Train (Section 7) | `model.fit(class_weight=class_weight_dict)` | Imbalance-aware training |
| 11 | Inference (Section 10) | 3 deterministic crops + averaged probabilities | More robust prediction |
| 12 | Run report | Updated version, added new config fields | Accurate metadata |

### 4.3. Cell 6 — Configuration (code)

Add new hyperparameter constants after the existing training hyperparameters:

```python
# -- v20a improvements ---------------------------------------------------------
LABEL_SMOOTHING = 0.05          # label smoothing for cross-entropy loss
WEIGHT_DECAY    = 1e-4          # AdamW decoupled weight decay

# -- SpecAugment parameters ----------------------------------------------------
SPEC_AUG_FREQ_MASK  = 15       # max frequency bands to mask
SPEC_AUG_TIME_MASK  = 25       # max time frames to mask
SPEC_AUG_NUM_MASKS  = 2        # number of masks per axis
```

Update cache directory:

```python
LOGMEL_CACHE_DIR = CACHE_DIR / "logmel_v20a_10s" / ("shared" if LOGMEL_CACHE_SHARED else SUBSET)
```

**Important note on cache reuse:** The v20a notebook uses the same 10-second
center-crop/pad extraction as v20 — the raw `.npy` spectrogram files are
identical. The cache directory rename is for namespace hygiene only. An
alternative implementation could reuse `logmel_v20_10s` directly. Either
approach is acceptable; the key constraint is that v20a index parquets must
not collide with v20 index parquets.

### 4.4. Cell 18 — Section 5: Preprocessing (code)

This cell currently contains:
- label encoding
- `compute_train_stats()` — streaming mean/std
- `make_dataset()` — tf.data pipeline
- dataset construction

#### 4.4.1. Add class-weight computation

After label encoding, before `compute_train_stats()`, add:

```python
from sklearn.utils.class_weight import compute_class_weight

_train_labels_int = train_index_u["label_int"].to_numpy()
_class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(N_CLASSES),
    y=_train_labels_int,
)
class_weight_dict = {i: float(w) for i, w in enumerate(_class_weights)}
print("Class weights:")
for i, g in enumerate(GENRE_CLASSES):
    print(f"  {g:<20s}  {class_weight_dict[i]:.4f}")
```

This dictionary will be passed to `model.fit()` in Section 7.

#### 4.4.2. Add SpecAugment to training pipeline

Add a `spec_augment()` function and apply it inside the training `make_dataset()` call only.

Implementation — as a tf.data map function applied **after** normalization,
**before** batching, **only for the training dataset**:

```python
def spec_augment(x, freq_mask=SPEC_AUG_FREQ_MASK, time_mask=SPEC_AUG_TIME_MASK,
                 num_masks=SPEC_AUG_NUM_MASKS):
    """SpecAugment: random frequency and time masking on a (N_MELS, N_FRAMES, 1) tensor."""
    shape = tf.shape(x)
    freq_dim = shape[0]   # N_MELS = 128
    time_dim = shape[1]   # N_FRAMES = 861

    for _ in range(num_masks):
        # Frequency mask
        f = tf.random.uniform([], 0, freq_mask, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, freq_dim - f, dtype=tf.int32)
        freq_mask_tensor = tf.concat([
            tf.ones([f0, time_dim, 1]),
            tf.zeros([f, time_dim, 1]),
            tf.ones([freq_dim - f0 - f, time_dim, 1]),
        ], axis=0)
        x = x * freq_mask_tensor

        # Time mask
        t = tf.random.uniform([], 0, time_mask, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, time_dim - t, dtype=tf.int32)
        time_mask_tensor = tf.concat([
            tf.ones([freq_dim, t0, 1]),
            tf.zeros([freq_dim, t, 1]),
            tf.ones([freq_dim, time_dim - t0 - t, 1]),
        ], axis=1)
        x = x * time_mask_tensor

    return x
```

Modify `make_dataset()` to accept an `augment` parameter:

```python
def make_dataset(index_df, batch_size, shuffle, augment=False):
    # ... existing code through _load_and_norm ...

    ds = ds.map(_load_and_norm, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(lambda x, y: (spec_augment(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds
```

Then construct datasets:

```python
train_ds = make_dataset(train_index_u, BATCH_SIZE, shuffle=True, augment=True)
val_ds   = make_dataset(val_index_u,   BATCH_SIZE, shuffle=False, augment=False)
test_ds  = make_dataset(test_index_u,  BATCH_SIZE, shuffle=False, augment=False)
```

**Key constraint:** SpecAugment is applied only to training data.
Validation and test data are never augmented.

### 4.5. Cell 19 — Section 6: Build Model markdown

Update the shape table to reflect milder pooling:

| Block | Layer | Kernel | Pool | Output shape |
|-------|-------|--------|------|-------------|
| 1 | Conv2D(32) + BN + MaxPool(2,2) | (5,5) | (2,2) | (64, 430, 32) |
| 2 | Conv2D(64) + BN + MaxPool(2,2) | (3,3) | (2,2) | (32, 215, 64) |
| 3 | Conv2D(128) + BN + MaxPool(2,2) | (3,3) | (2,2) | (16, 107, 128) |
| 4 | Conv2D(128) + BN + MaxPool(2,2) | (3,3) | (2,2) | (8, 53, 128) |
| Head | GAP + Dropout(0.3) + Dense | — | — | (N_CLASSES,) |

Note: with (2,2) × 4 pooling, the feature map before GAP is `(8, 53, 128)`
instead of `(8, 6, 128)` in v20. This preserves significantly more temporal
resolution. GAP absorbs the larger spatial dimensions automatically.

### 4.6. Cell 20 — Section 6: `build_model()` (code)

Change all four MaxPool layers to `(2, 2)`:

```python
def build_model(n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*LOGMEL_SHAPE, 1), name="logmel")

    # Block 1 — local spectro-temporal features
    x = layers.Conv2D(32, (5, 5), padding="same", activation="relu", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPool2D((2, 2), name="pool1")(x)      # was (2, 4)

    # Block 2 — mid-level motifs
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPool2D((2, 2), name="pool2")(x)      # was (2, 4)

    # Block 3 — higher-level patterns
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPool2D((2, 2), name="pool3")(x)      # was (2, 4)

    # Block 4 — global structure
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPool2D((2, 2), name="pool4")(x)      # unchanged

    # Classifier head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="fc_out")(x)

    return keras.Model(inputs, outputs, name="logmel_2dcnn_v20a")
```

The only changes:
- `pool1`: `(2, 4)` → `(2, 2)`
- `pool2`: `(2, 4)` → `(2, 2)`
- `pool3`: `(2, 4)` → `(2, 2)`
- `pool4`: `(2, 2)` → `(2, 2)` (unchanged)
- model name: `logmel_2dcnn_v20` → `logmel_2dcnn_v20a`

No new layers are added. No layer types change. The Conv2D filters,
kernel sizes, BatchNorm placements, GAP, Dropout, and Dense head are
all identical to v20.

### 4.7. Cell 22 — Section 7: Compile & Train (code)

Three changes in this cell:

#### 4.7.1. Replace Adam with AdamW

```python
# v20:
# optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)

# v20a:
optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=WEIGHT_DECAY)
```

#### 4.7.2. Replace loss string with label-smoothing loss

```python
# v20:
# loss="categorical_crossentropy"

# v20a:
loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
```

#### 4.7.3. Pass class weights to model.fit()

```python
# v20:
# history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# v20a:
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weight_dict,
)
```

#### 4.7.4. Update run report metadata

In the report dictionary:
- `"version": "v20a"`
- `run_id`: `logmel-cnn-v20a-{ts}`
- `config.training.optimizer`: `"AdamW"`
- `config.training.weight_decay`: `WEIGHT_DECAY`
- `config.training.label_smoothing`: `LABEL_SMOOTHING`
- `config.training.class_weights`: `class_weight_dict`
- Add `config.training.spec_augment`: `{"freq_mask": SPEC_AUG_FREQ_MASK, "time_mask": SPEC_AUG_TIME_MASK, "num_masks": SPEC_AUG_NUM_MASKS}`
- Add `config.architecture_changes`: `"milder_pooling_2x2_all_blocks"`

### 4.8. Cell 28 — Section 10: Inference (code)

Replace the single-crop inference with **3 deterministic crops + probability averaging**.

#### 4.8.1. Three-crop extraction function

```python
def extract_three_crops(y: np.ndarray, sr: int, target_sec: float) -> list[np.ndarray]:
    """Extract 3 deterministic 10-second crops from a waveform.

    For clips longer than 3 * target_sec:
      - early crop:  centered at 25% of clip duration
      - middle crop: centered at 50% of clip duration
      - late crop:   centered at 75% of clip duration

    For clips between target_sec and 3 * target_sec:
      - early crop:  starting from the beginning
      - middle crop: centered at midpoint
      - late crop:   ending at the end

    For clips <= target_sec:
      - return 3 copies of the same center-padded clip
    """
    target_len = int(round(target_sec * sr))
    n = len(y)

    if n <= target_len:
        padded = normalize_to_fixed_duration(y, sr, target_sec)
        return [padded, padded, padded]

    def _crop_at(center):
        half = target_len // 2
        start = center - half
        end = start + target_len
        if start < 0:
            start, end = 0, target_len
        if end > n:
            end = n
            start = n - target_len
        return y[start:end]

    if n >= 3 * target_len:
        return [_crop_at(n // 4), _crop_at(n // 2), _crop_at(3 * n // 4)]
    else:
        return [
            y[:target_len],                                     # early
            _crop_at(n // 2),                                   # middle
            y[n - target_len:],                                 # late
        ]
```

#### 4.8.2. Multi-crop inference logic

```python
try:
    y_raw = _load_audio_simple(INFER_PATH, sr=SAMPLE_RATE, mono=True, duration=None)
    ok, reason = _sanity_check_audio(y_raw, SAMPLE_RATE)
    if not ok:
        raise ValueError(f"sanity_check_failed:{reason}")

    crops = extract_three_crops(y_raw, SAMPLE_RATE, CLIP_DURATION)
    crop_logmels = [_logmel_fixed_shape(c) for c in crops]
except Exception as exc:
    print(f"Could not load/extract log-mel: {exc}")
    crop_logmels = None

if crop_logmels is not None:
    crop_probs = []
    for logmel in crop_logmels:
        x_infer = ((logmel[np.newaxis, ..., np.newaxis] - mu) / std).astype(np.float32)
        p = model.predict(x_infer, verbose=0)[0]
        crop_probs.append(p)

    avg_probs  = np.mean(crop_probs, axis=0)
    pred_idx   = int(np.argmax(avg_probs))
    pred_genre = GENRE_CLASSES[pred_idx]
    confidence = float(avg_probs[pred_idx])

    print(f"\nPredicted genre : {pred_genre}  (confidence: {confidence:.2%})")
    print(f"True genre      : {true_genre}")
    print(f"Inference mode  : 3-crop average")

    # Show per-crop predictions for transparency
    for i, p in enumerate(crop_probs):
        ci = int(np.argmax(p))
        print(f"  Crop {i+1}: {GENRE_CLASSES[ci]} ({float(p[ci]):.2%})")

    # Plot averaged probabilities
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ["steelblue" if g != pred_genre else "tomato" for g in GENRE_CLASSES]
    ax.barh(GENRE_CLASSES, avg_probs, color=colors)
    ax.set_xlabel("Probability (3-crop avg)")
    ax.set_title(f"Genre probabilities (v20a, 10s, 3-crop) -- {INFER_PATH.name}")
    ax.axvline(1 / N_CLASSES, color="grey", linestyle="--", linewidth=0.8,
               label=f"Chance ({1/N_CLASSES:.2%})")
    ax.legend(fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()
```

---

## 5. Cells that remain unchanged

These v20 cells carry over to v20a without modification:

| Cell(s) | Section | Reason |
|---------|---------|--------|
| 4 | Imports | No new imports needed at the top (sklearn import added inline in Section 5) |
| 5–6 | Device selection | Same runtime selection logic |
| 7–8 | Device setup code | Same `configure_runtime_device()` |
| 9–11 | Section 3 — Load manifest + genre plot | Same manifest loading; imbalance is now handled by class weights instead |
| 12–15 | Section 4 — Feature extraction | Same 10s center-crop/pad extraction pipeline and caching |
| 16 | Section 4b — Log-mel sample plot | Unchanged |
| 23–24 | Section 8 — Training history plot | Update plot title only (`v20` → `v20a`) |
| 25–26 | Section 9 — Evaluation + confusion matrix | Unchanged (same `eval_dataset()` logic) |
| 29–30 | Runtime summary | Unchanged |

---

## 6. Feature extraction cache policy

The v20a notebook uses **exactly the same** spectrogram extraction as v20:

- same `CLIP_DURATION = 10.0`
- same `normalize_to_fixed_duration()` center-crop/pad
- same `_logmel_fixed_shape()` extraction
- same `LOGMEL_SHAPE = (128, 861)`

The per-track `.npy` files are **byte-identical** to v20's cached spectrograms.

The cache directory is renamed to `logmel_v20a_10s` only for namespace separation.
If desired, v20a could alternatively reuse v20's `logmel_v20_10s` cache directly
by keeping the same cache path. Both strategies are valid.

---

## 7. Architecture comparison: v20 vs v20a

### Pooling behavior

| Block | v20 pool | v20a pool | v20 time dim after pool | v20a time dim after pool |
|-------|----------|-----------|------------------------|-------------------------|
| 1 | (2, 4) | (2, 2) | 861 → 215 | 861 → 430 |
| 2 | (2, 4) | (2, 2) | 215 → 53 | 430 → 215 |
| 3 | (2, 4) | (2, 2) | 53 → 13 | 215 → 107 |
| 4 | (2, 2) | (2, 2) | 13 → 6 | 107 → 53 |

**v20 pre-GAP feature map:** `(8, 6, 128)` = 6,144 values per sample
**v20a pre-GAP feature map:** `(8, 53, 128)` = 54,272 values per sample

The time axis retains ~9× more resolution before global averaging. This
preservation is the mechanism by which the model can learn richer temporal
patterns.

### Parameter count

The model parameter count is **identical** between v20 and v20a. Pooling layers
have no trainable parameters, and all Conv2D/BN/Dense layers are unchanged.

---

## 8. Training behavior changes summary

| Aspect | v20 | v20a |
|--------|-----|------|
| Optimizer | `Adam(lr=1e-3)` | `AdamW(lr=1e-3, weight_decay=1e-4)` |
| Loss | `categorical_crossentropy` | `CategoricalCrossentropy(label_smoothing=0.05)` |
| Class weights | None | Balanced, computed from training split |
| Augmentation | None | SpecAugment (freq + time masking, training only) |
| Pooling | (2,4), (2,4), (2,4), (2,2) | (2,2) × 4 |
| Inference | Single center-crop | 3 deterministic crops, averaged probabilities |

---

## 9. Implementation sequence

Implement in this order to keep each step verifiable:

1. **Copy v20 → v20a** — duplicate the notebook file
2. **Rename all version references** — v20 → v20a in all strings, paths, titles
3. **Update pooling** — change MaxPool sizes in `build_model()`
4. **Add class weights** — add computation in Section 5, pass to `model.fit()`
5. **Add SpecAugment** — add function + augment flag in `make_dataset()`
6. **Switch to AdamW + label smoothing** — update compile call
7. **Add 3-crop inference** — replace Section 10 inference logic
8. **Update run report** — add new config fields
9. **Review all markdown cells** — update shape tables, descriptions, titles
10. **Smoke test** — run on `tiny` or `small` subset to verify no errors

---

## 10. Verification checklist

After implementation, confirm:

- [ ] Notebook runs end-to-end on `small` subset without errors
- [ ] `model.summary()` shows the same layer types and param count as v20
- [ ] Pre-GAP feature map shape is `(None, 8, 53, 128)` (not `(None, 8, 6, 128)`)
- [ ] Class weights are printed and look reasonable (rare genres > 1.0, common genres < 1.0)
- [ ] SpecAugment only applies to `train_ds`, not `val_ds` or `test_ds`
- [ ] Optimizer printed as `AdamW` in training output
- [ ] Loss uses label smoothing (verify via `model.loss.get_config()`)
- [ ] Inference prints 3 per-crop predictions before the averaged result
- [ ] Run report JSON contains all new v20a config fields
- [ ] Cache directory is `logmel_v20a_10s` (or documented reuse of v20 cache)
- [ ] No references to `v20` remain (except in the changelog comparing v20→v20a)
