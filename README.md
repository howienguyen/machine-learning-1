
# MelCNN-MGR — Music Genre Recognition with MFCC and Log-Mel CNNs

A study comparing **MFCC** and **Log-Mel spectrograms** as input representations for
2D CNN music genre classification, using the
[FMA (Free Music Archive)](https://github.com/mdeff/fma) dataset.

Project codename: **MelCNN MGR** (MGR = Music Genre Recognition)

---

## Project Goals

| Goal | Description |
|------|-------------|
| **Goal 1 — Controlled Study** | Compare MFCC vs Log-Mel using the **same CNN architecture and training setup**, changing only the input feature. |
| **Goal 2 — Best-Result Engineering** | Starting from the Goal-1 Log-Mel baseline, improve performance with architecture and training upgrades. |
| **Goal 3 — Production Robustness** | Address fixed-length dependency, class imbalance, and CPU-bound pipeline. |

See [`docs/Final-Project-Proposal.md`](docs/Final-Project-Proposal.md) for the full proposal.

---

## Prerequisites

### 1. Dataset

Download the FMA dataset and place it at `FMA/`:

- `FMA/fma_small/` — 8 genres, ~7.2 GB (used for quick runs with `--subset small`)
- `FMA/fma_medium/` — 16 genres, ~22 GB
- `FMA/fma_metadata/` — metadata CSVs (`tracks.csv`, etc.)

### 2. Build the metadata manifest (one-time step)

All training notebooks/scripts read from pre-built parquet manifests.
Run this before any training:

```bash
python MelCNN-MGR/preprocessing/build_manifest.py --subset small
# or for the full medium dataset:
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium
```

Outputs are written to `MelCNN-MGR/data/processed/`.

### 3. Python environment

```bash
# intel-extension-for-tensorflow (optional — for Intel XPU acceleration)
pip install --upgrade intel-extension-for-tensorflow[xpu] -f https://developer.intel.com
pip install --upgrade intel-extension-for-tensorflow[cpu] -f https://developer.intel.com
```

All notebooks/scripts auto-detect and fall back through CUDA GPU → Intel XPU → CPU.

---

## Main Notebooks & Scripts

### `MelCNN-MGR/notebooks/baseline_mfcc_cnn_v5.ipynb`
**Goal 1 — MFCC baseline (reference implementation)**

A modernized reimplementation of the official FMA ConvNet-on-MFCC baseline
([`FMA/fma-repo/baselines.ipynb`](FMA/fma-repo/baselines.ipynb) §3.1,
Li et al. IMECS 2010). The core architecture is kept **faithful to the original**;
modern engineering layers are added around it (ffmpeg decoding, parquet manifest,
`tf.data` pipeline, CUDA→XPU→CPU fallback, timestamped run directories).

| Property | Value |
|---|---|
| Feature | MFCC — 13 coefficients, 30 s clips |
| Input shape | `(13, 2582, 1)` |
| Architecture | 3 × Conv2D (global kernel on freq axis) → Flatten → Dense |
| Optimizer | SGD lr=1e-3 |
| Epochs | 20 |
| Run artifacts | `MelCNN-MGR/models/mfcc-cnn-<timestamp>/` |

**Run:**
```bash
# Open and run all cells in VS Code / Jupyter
MelCNN-MGR/notebooks/baseline_mfcc_cnn_v5.ipynb
```

---

### `MelCNN-MGR/notebooks/baseline_logmel_cnn_v10.ipynb`
**Goal 1 — Log-Mel baseline (representation-fair counterpart to v5)**

A representation-fair log-mel CNN baseline designed for a controlled comparison with
`baseline_mfcc_cnn_v5.ipynb`. Fixes the Conv1 kernel design issue present in v1
(which collapsed the entire 128-band frequency axis in a single layer), replacing it
with a modern 4-block CNN with local (5×5 / 3×3) kernels, BatchNorm, GlobalAveragePooling,
and adaptive callbacks. Uses **30-second clips** — the same duration as the MFCC baseline.

| Property | Value |
|---|---|
| Feature | Log-Mel spectrogram — 128 bands, 30 s clips |
| Input shape | `(128, 2582, 1)` |
| Architecture | 4 × (Conv2D + BN + MaxPool) → GAP → Dropout(0.3) → Dense |
| Optimizer | Adam lr=1e-3 with EarlyStopping & ReduceLROnPlateau |
| Epochs | 30 max (early stopping patience=5) |
| Run artifacts | `MelCNN-MGR/models/logmel-cnn-v10-<timestamp>/` |

**Run:**
```bash
MelCNN-MGR/notebooks/baseline_logmel_cnn_v10.ipynb
```

Further reading: [`docs/Conv1 Kernel Design Issue in Log-Mel Baseline.md`](docs/Conv1%20Kernel%20Design%20Issue%20in%20Log-Mel%20Baseline.md)

---

### `MelCNN-MGR/notebooks/baseline_logmel_cnn_v20a.py`
**Goal 2 — Quality-improved Log-Mel baseline (production-like)**

The most capable model in the project. Builds on the v20 10-second center-crop
pipeline and adds a suite of training improvements targeted at better generalization
and handling of class imbalance. Runs as a plain Python script (can be executed
directly or imported).

| Property | Value |
|---|---|
| Feature | Log-Mel spectrogram — 128 bands, 10 s clips (center-crop/pad) |
| Input shape | `(128, 861, 1)` |
| Architecture | 4 × (Conv2D + BN + MaxPool(2,2)) → GAP → Dropout(0.3) → Dense |
| Optimizer | AdamW lr=1e-3, weight decay 1e-4 |
| Loss | Categorical cross-entropy with label smoothing 0.05 |
| Class weights | Balanced via `compute_class_weight("balanced")` |
| Augmentation | SpecAugment (freq + time masking) during training |
| Inference | 3-crop (early/middle/late averaged probabilities) |
| Epochs | 50 max (early stopping patience=7) |
| Run artifacts | `MelCNN-MGR/models/logmel-cnn-v20a-<timestamp>/` |

**Run:**
```bash
cd MelCNN-MGR/notebooks
python baseline_logmel_cnn_v20a.py
```

Further reading: [`docs/Implementation Guide - baseline_logmel_cnn_v20a.md`](docs/Implementation%20Guide%20-%20baseline_logmel_cnn_v20a.md)

---

## Reusable Inference Modules

Each trained model has a companion inference module that loads the saved `.keras`
model and `norm_stats.npz` from a run directory and classifies new audio files
without re-running the training notebook.

| Training source | Inference module | Example script |
|---|---|---|
| `baseline_logmel_cnn_v20a.py` | `MelCNN-MGR/inference_logmel_v20a.py` | `MelCNN-MGR/examples/inference_logmel_v20a_example.py` |
| `baseline_logmel_cnn_v20.ipynb` | `MelCNN-MGR/inference_logmel_v20.py` | `MelCNN-MGR/examples/inference_logmel_v20_example.py` |
| `baseline_mfcc_cnn_v5.ipynb` | `MelCNN-MGR/inference_mfcc_v5.py` | `MelCNN-MGR/examples/inference_mfcc_v5_example.py` |

**Example usage:**
```bash
# Classify 3 specific files using the v20a model
python MelCNN-MGR/examples/inference_logmel_v20a_example.py \
    --run-dir MelCNN-MGR/models/logmel-cnn-v20a-20260308-013615 \
    --files FMA/fma_small/000/000002.mp3 FMA/fma_small/000/000005.mp3

# Pick 5 random test samples from the MFCC model
python MelCNN-MGR/examples/inference_mfcc_v5_example.py \
    --run-dir MelCNN-MGR/models/mfcc-cnn-20260308-120000 \
    --subset small --random 5
```

See [`dev-logs/2026-03-08-inference-modules.md`](dev-logs/2026-03-08-inference-modules.md)
for full documentation.

---

## Project Structure

```
MelCNN-MGR/
├── notebooks/
│   ├── baseline_mfcc_cnn_v5.ipynb          ← MFCC baseline (Goal 1)
│   ├── baseline_logmel_cnn_v10.ipynb       ← Log-Mel baseline (Goal 1, representation-fair)
│   ├── baseline_logmel_cnn_v20a.py         ← Quality-improved Log-Mel (Goal 2)
│   └── baseline_logmel_cnn_v20.ipynb       ← Log-Mel 10s center-crop (intermediate)
├── preprocessing/
│   └── build_manifest.py                   ← One-time metadata preprocessing
├── inference_logmel_v20a.py                ← Inference module for v20a
├── inference_logmel_v20.py                 ← Inference module for v20
├── inference_mfcc_v5.py                    ← Inference module for MFCC v5
├── examples/
│   ├── inference_logmel_v20a_example.py
│   ├── inference_logmel_v20_example.py
│   └── inference_mfcc_v5_example.py
├── models/                                 ← Timestamped run directories (gitignored)
├── cache/                                  ← Per-track .npy feature cache (gitignored)
└── data/processed/                         ← Manifest parquets from build_manifest.py
FMA/
├── fma_small/                              ← Audio files (8 genres)
├── fma_medium/                             ← Audio files (16 genres)
└── fma_metadata/                           ← tracks.csv, genres.csv, etc.
docs/                                       ← Design documents and guidelines
dev-logs/                                   ← Session-by-session development logs
```

---

## Dev Logs

| Date | Log |
|------|-----|
| 2026-03-08 | [Reusable inference modules](dev-logs/2026-03-08-inference-modules.md) |
| 2026-03-06 | [Log-Mel v10 → v20 update](dev-logs/2026-03-06-logmel-v10-v20-update.md) |
| 2026-03-06 | [Baseline Log-Mel CNN v10](dev-logs/2026-03-06-baseline-logmel-cnn-v10.md) |
| 2026-03-05 | [Training pipeline improvements](dev-logs/2026-03-05-melcnn-mgr-training-pipeline-improvements.md) |
| 2026-03-04 | [MelCNN-MGR preprocessing pipeline](dev-logs/2026-03-04-melcnn-mgr-preprocessing-pipeline.md) |
| 2026-03-04 | [Dequantization warning fix](dev-logs/2026-03-04-dequantization-warning-fix.md) |

