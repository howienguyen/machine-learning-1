# Dev Log — 2026-03-08 — Reusable Inference Modules

## Scope

This log covers the development of standalone, reusable inference modules for each
trained model in the MelCNN-MGR project:

1. Renamed existing logmel v20a inference files for clarity.
2. Added `norm_stats.npz` persistence to `baseline_mfcc_cnn_v5.ipynb`.
3. Created `inference_mfcc_v5.py` — inference module for the MFCC CNN v5 baseline.
4. Created `inference_mfcc_v5_example.py` — CLI example for the MFCC CNN v5 module.
5. Added `norm_stats.npz` persistence to `baseline_logmel_cnn_v10.ipynb`.
6. Created `inference_logmel_v10.py` — inference module for the Log-Mel CNN v10 baseline.
7. Created `inference_logmel_v10_example.py` — CLI example for the Log-Mel CNN v10 module.

---

## Change 1 — Rename logmel v20a inference files

### Motivation
The original filenames `inference.py` and `inference_example.py` were generic and did
not communicate which model they served.  As a second inference module was being added
for the MFCC baseline, explicit naming became necessary to avoid confusion.

### Renames

| Old path | New path |
|---|---|
| `MelCNN-MGR/inference.py` | `MelCNN-MGR/inference_logmel_v20a.py` |
| `MelCNN-MGR/examples/inference_example.py` | `MelCNN-MGR/examples/inference_logmel_v20a_example.py` |

Internal docstrings and import statements inside both files were updated to reflect the
new filenames.

---

## Change 2 — Save `norm_stats.npz` in `baseline_mfcc_cnn_v5.ipynb`

### Problem
`baseline_mfcc_cnn_v5.ipynb` computed per-coefficient µ/σ normalization statistics
from the training data (streaming pass), but never wrote them to disk.  Without these
stats, post-training inference against unseen audio would require re-running the entire
training notebook just to recover `mu` and `std`.

### Fix
Section 7 (Compile & Train) of the notebook now saves a `norm_stats.npz` file
immediately after the model `.keras` file:

```python
_norm_path = RUN_DIR / "norm_stats.npz"
np.savez(
    str(_norm_path),
    mu=mu,                              # shape (1, 13, 1, 1)
    std=std,                            # shape (1, 13, 1, 1)
    genre_classes=np.array(GENRE_CLASSES),
)
```

This mirrors the same pattern already in place for `baseline_logmel_cnn_v20a.py`.

### Run directory layout after this change

```
MelCNN-MGR/models/mfcc-cnn-YYYYMMDD-HHMMSS/
    baseline_mfcc_cnn.keras    ← Keras model weights + architecture
    norm_stats.npz             ← mu (1,13,1,1), std (1,13,1,1), genre_classes [NEW]
    run_report_<subset>.json   ← config, training history, evaluation metrics
```

---

## Change 3 — `MelCNN-MGR/inference_mfcc_v5.py`

A new standalone inference module for the MFCC CNN v5 baseline, mirroring the design
of `inference_logmel_v20a.py`.

### Public API

```python
from inference_mfcc_v5 import MFCCCNNInference, PredictionResult

engine = MFCCCNNInference("MelCNN-MGR/models/mfcc-cnn-20260308-120000")

# Single-crop prediction (default)
result = engine.predict("path/to/song.mp3")
print(result.genre, result.confidence)

# Three-crop prediction (3 deterministic 30s crops, averaged)
result = engine.predict("path/to/song.mp3", mode="three_crop")

# Batch prediction
results = engine.predict_batch(["a.mp3", "b.mp3", "c.mp3"])

# Top-k genres
for genre, prob in result.top_k(3):
    print(genre, prob)
```

### Key constants (must match training in v5 notebook)

| Constant | Value |
|---|---|
| `SAMPLE_RATE` | 22 050 Hz |
| `N_MFCC` | 13 |
| `N_FFT` | 512 |
| `HOP_LENGTH` | 256 |
| `CLIP_DURATION` | 30.0 s |
| `N_FRAMES` | 2582 |
| `MFCC_SHAPE` | (13, 2582) |

### Data classes

- `PredictionResult` — genre, confidence, full probability vector, top_k(), mode, per-crop details
- `CropDetail` — per-crop genre, confidence, probs

### Default prediction mode

`single_crop` is the default (unlike `inference_logmel_v20a.py` which defaults to
`three_crop`).  Rationale: the MFCC model was trained on 30 s clips so the full-clip
single-crop is already the canonical input.  Three-crop is available as an option for
longer tracks.

### Audio backend

Same ffmpeg-first / librosa-fallback policy as the v5 notebook:
- `ffmpeg` used if `shutil.which("ffmpeg")` is not None (fast MP3 decoding).
- `librosa.load()` as fallback.

---

## Change 4 — `MelCNN-MGR/examples/inference_mfcc_v5_example.py`

CLI demonstration script, symmetric to `inference_logmel_v20a_example.py`.

### Usage

```bash
# Predict on specific files
python MelCNN-MGR/examples/inference_mfcc_v5_example.py \
    --run-dir MelCNN-MGR/models/mfcc-cnn-20260308-120000 \
    --files song1.mp3 song2.mp3 song3.mp3

# Random samples from the test manifest
python MelCNN-MGR/examples/inference_mfcc_v5_example.py \
    --run-dir MelCNN-MGR/models/mfcc-cnn-20260308-120000 \
    --subset small --random 5

# Three-crop mode
python MelCNN-MGR/examples/inference_mfcc_v5_example.py \
    --run-dir MelCNN-MGR/models/mfcc-cnn-20260308-120000 \
    --subset small --random 3 --mode three_crop
```

### Arguments

| Argument | Description |
|---|---|
| `--run-dir` | Path to timestamped run directory (required) |
| `--files` | One or more audio files to classify |
| `--subset` | FMA subset name; used with `--random` |
| `--random N` | Pick N random samples from `test_<subset>.parquet` |
| `--mode` | `single_crop` (default) or `three_crop` |

---

## File layout summary (after all changes)

```
MelCNN-MGR/
    inference_logmel_v20a.py          ← logmel v20a inference module (renamed)
    inference_mfcc_v5.py              ← MFCC v5 inference module     [NEW]
    inference_logmel_v10.py           ← logmel v10 inference module  [NEW]
    examples/
        inference_logmel_v20a_example.py   ← logmel v20a CLI example (renamed)
        inference_mfcc_v5_example.py       ← MFCC v5 CLI example     [NEW]
        inference_logmel_v10_example.py    ← logmel v10 CLI example  [NEW]
    notebooks/
        baseline_logmel_cnn_v20a.py   ← training script (unchanged)
        baseline_mfcc_cnn_v5.ipynb    ← training notebook (norm_stats.npz added)
        baseline_logmel_cnn_v10.ipynb ← training notebook (norm_stats.npz added)
```

---

## Change 5 — Save `norm_stats.npz` in `baseline_logmel_cnn_v10.ipynb`

### Problem
`baseline_logmel_cnn_v10.ipynb` computed per-band µ/σ normalization stats from
the training data but never wrote them to disk.  Without these stats, re-running
inference against unseen audio would require executing the full training notebook
to recover `mu` and `std`.

### Fix
Section 7 (Compile & Train) of the notebook now saves a `norm_stats.npz` file
immediately after the model `.keras` file:

```python
_norm_path = RUN_DIR / "norm_stats.npz"
np.savez(str(_norm_path), mu=mu, std=std, genre_classes=np.array(GENRE_CLASSES))
print(f"Norm stats   -> {_norm_path}")
```

### Run directory layout after this change

```
MelCNN-MGR/models/logmel-cnn-v10-YYYYMMDD-HHMMSS/
    baseline_logmel_cnn_v10.keras  ← Keras model weights + architecture
    norm_stats.npz                 ← mu (1,128,1,1), std (1,128,1,1), genre_classes [NEW]
    run_report_<subset>.json       ← config, training history, evaluation metrics
```

---

## Change 6 — `MelCNN-MGR/inference_logmel_v10.py`

A new standalone inference module for the Log-Mel CNN v10 baseline.

### Public API

```python
from inference_logmel_v10 import LogMelV10Inference, PredictionResult

engine = LogMelV10Inference("MelCNN-MGR/models/logmel-cnn-v10-20260308-120000")

# Single-crop prediction (default)
result = engine.predict("path/to/song.mp3")
print(result.genre, result.confidence)

# Three-crop prediction (3 deterministic 30s crops, averaged)
result = engine.predict("path/to/song.mp3", mode="three_crop")

# Batch prediction
results = engine.predict_batch(["a.mp3", "b.mp3", "c.mp3"])

# Top-k genres
for genre, prob in result.top_k(3):
    print(genre, prob)
```

### Key constants (must match training in v10 notebook)

| Constant | Value |
|---|---|
| `SAMPLE_RATE` | 22 050 Hz |
| `N_MELS` | 128 |
| `N_FFT` | 512 |
| `HOP_LENGTH` | 256 |
| `CLIP_DURATION` | 30.0 s |
| `N_FRAMES` | 2582 |
| `LOGMEL_SHAPE` | (128, 2582) |

### Default prediction mode

`single_crop` is the default.  Rationale: v10 was trained on full 30 s clips
(no crop augmentation), so the full-clip single-crop is the canonical input.

---

## Change 7 — `MelCNN-MGR/examples/inference_logmel_v10_example.py`

CLI demonstration script.

### Usage

```bash
# Predict on specific files
python MelCNN-MGR/examples/inference_logmel_v10_example.py \
    --run-dir MelCNN-MGR/models/logmel-cnn-v10-20260308-120000 \
    --files song1.mp3 song2.mp3 song3.mp3

# Random samples from the test manifest
python MelCNN-MGR/examples/inference_logmel_v10_example.py \
    --run-dir MelCNN-MGR/models/logmel-cnn-v10-20260308-120000 \
    --subset small --random 5

# Three-crop mode
python MelCNN-MGR/examples/inference_logmel_v10_example.py \
    --run-dir MelCNN-MGR/models/logmel-cnn-v10-20260308-120000 \
    --subset small --random 3 --mode three_crop
```

---

## Difference table — logmel v20a vs MFCC v5 vs logmel v10 inference modules

| Aspect | `inference_logmel_v20a.py` | `inference_mfcc_v5.py` | `inference_logmel_v10.py` |
|---|---|---|---|
| Feature | Log-mel spectrogram | MFCC (13 coefficients) | Log-mel spectrogram |
| Feature shape | (128, 861) | (13, 2582) | (128, 2582) |
| Clip duration | 10 s | 30 s | 30 s |
| Default mode | `three_crop` | `single_crop` | `single_crop` |
| Model file | `baseline_logmel_cnn_v20a.keras` | `baseline_mfcc_cnn.keras` | `baseline_logmel_cnn_v10.keras` |
| Run dir prefix | `logmel-cnn-v20a-*` | `mfcc-cnn-*` | `logmel-cnn-v10-*` |
| Inference class | `MelCNNInference` | `MFCCCNNInference` | `LogMelV10Inference` |

—
Generated during March 8, 2026 session.

Update note, March 14, 2026:

1. the current config-driven v2-family inference entry point is now `MelCNN-MGR/model_inference/inference_logmel_cnn_v2_x.py`
2. current service and direct-inference examples should prefer `--model-dir` instead of `--run-dir`
