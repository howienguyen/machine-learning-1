
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

### 2. Build the metadata manifests

All training notebooks/scripts read from pre-built parquet manifests under
`MelCNN-MGR/data/processed/`.

Build the `small` manifest:

```bash
python MelCNN-MGR/preprocessing/build_manifest.py --subset small
```

Build the `medium` manifest:

```bash
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium
```

> [!IMPORTANT]
> When `--audio-root` is omitted, `build_manifest.py` now derives it from `--subset` automatically:
> `small -> FMA/fma_small`, `medium -> FMA/fma_medium`, `large -> FMA/fma_large`.
> Pass `--audio-root` explicitly only when you want to override that mapping.

Current manifest behavior:

- Full outputs: `metadata_manifest_{subset}.parquet`, config JSON, text report
- Split outputs: `train_{subset}.parquet`, `val_{subset}.parquet`, `test_{subset}.parquet`
- Identity columns are included in manifests and split parquets: `sample_id`, `source`
- The currently excluded top-level label is `International`

### 3. Build the unified dataset/sample manifests for training

After the FMA metadata manifest exists, build the MelCNN-MGR unified manifests:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py
```

Run Stage 1 only to produce the intermediate manifests:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
    --mode stage1
```

Run Stage 2 only to consume an existing `manifest_all_samples.parquet` and rebuild just the final manifest:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
    --mode stage2
```

Run both stages explicitly:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
    --mode both
```

Run with explicit FMA subset and INFO logging:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
    --mode both \
    --fma-subset medium \
    --log-level INFO
```

Force an FMA rescan from `tracks.csv` instead of reusing the cached metadata parquet:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
    --mode both \
    --fma-subset medium \
    --force-rescan \
    --log-level INFO
```

This script supports three execution modes:

- `--mode stage1`: writes `manifest_all_datasets.parquet` and `manifest_all_samples.parquet`
- `--mode stage2`: reads the existing Stage 1 sample manifest and writes `manifest_final_samples.parquet`
- `--mode both`: runs the full pipeline and writes all three parquet outputs

Output files are written under `MelCNN-MGR/data/processed/` by default:

- `manifest_all_datasets.parquet`
- `manifest_all_samples.parquet`
- `manifest_final_samples.parquet`

Report and config outputs depend on mode:

- `--mode stage1` and `--mode both`: `manifest_all_datasets.report.txt`, `manifest_all_datasets.config.json`
- `--mode stage2`: `manifest_final_samples.report.txt`, `manifest_final_samples.config.json`

Use custom output locations if needed:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
    --mode both \
    --all-datasets-out /tmp/manifest_all_datasets.parquet \
    --all-samples-out /tmp/manifest_all_samples.parquet \
    --final-samples-out /tmp/manifest_final_samples.parquet
```

For Stage 2-only runs, `--all-samples-out` acts as the input path of the existing Stage 1 sample manifest:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
    --mode stage2 \
    --all-samples-out /tmp/manifest_all_samples.parquet \
    --final-samples-out /tmp/manifest_final_samples.parquet
```

Further details: [`docs/MelCNN-MGR-build_all_datasets_and_samples.md`](docs/MelCNN-MGR-build_all_datasets_and_samples.md)

### 4. Refresh derived data before reruns

Before rerunning preprocessing for `small` and `medium`, remove only derived artifacts.
Do not delete the raw FMA dataset.

Recommended cleanup:

```bash
rm -f MelCNN-MGR/data/processed/metadata_manifest_small.parquet
rm -f MelCNN-MGR/data/processed/metadata_manifest_medium.parquet
rm -f MelCNN-MGR/data/processed/train_small.parquet
rm -f MelCNN-MGR/data/processed/val_small.parquet
rm -f MelCNN-MGR/data/processed/test_small.parquet
rm -f MelCNN-MGR/data/processed/train_medium.parquet
rm -f MelCNN-MGR/data/processed/val_medium.parquet
rm -f MelCNN-MGR/data/processed/test_medium.parquet
rm -f MelCNN-MGR/data/processed/metadata_manifest_config_small.json
rm -f MelCNN-MGR/data/processed/metadata_manifest_config_medium.json
rm -f MelCNN-MGR/data/processed/metadata_manifest_report_small.txt
rm -f MelCNN-MGR/data/processed/metadata_manifest_report_medium.txt
rm -f MelCNN-MGR/data/processed/extra_samples_for_small_dataset.json
rm -rf MelCNN-MGR/cache
```

Why clear `MelCNN-MGR/cache/`:

- `baseline_logmel_cnn_v20a.py`, `baseline_logmel_cnn_v20a1.py`, and `baseline_logmel_cnn_v21.py` cache feature `.npy` files under `MelCNN-MGR/cache/`
- those cache filenames are still keyed by `track_id`
- once multi-source supplementation is introduced, stale cache entries can collide across sources unless the cache is refreshed

### 5. Collect extra samples for the small dataset

After rebuilding both manifests, generate the supplementation candidate JSON:

```bash
python MelCNN-MGR/preprocessing/collect_extra_samples_for_small_dataset.py
```

This writes:

- `MelCNN-MGR/data/processed/extra_samples_for_small_dataset.json`

The collector currently:

- loads `metadata_manifest_small.parquet` to exclude exact-small track IDs
- loads `metadata_manifest_medium.parquet` to recover FMA-medium filepaths
- scans external datasets under `additional_datasets/dortmund-university/` and `additional_datasets/gtzan/`
- writes candidate records with source provenance

Configuration lives in `MelCNN-MGR/settings.json` under `small_dataset_supplementation`.

### 6. Load selected extra samples into the small split parquets

After generating `extra_samples_for_small_dataset.json`, run:

```bash
python MelCNN-MGR/preprocessing/load_extra_samples_for_small_dataset_splits.py
```

This script reads:

- `MelCNN-MGR/settings.json`
- `MelCNN-MGR/data/processed/extra_samples_for_small_dataset.json`
- `MelCNN-MGR/data/processed/metadata_manifest_medium.parquet`
- the existing `train_small.parquet`, `val_small.parquet`, and `test_small.parquet`

Current behavior:

- target subset is `small`
- for each genre in `small_dataset_supplementation.target_genres`, it takes up to `n_extra_expected` selected rows
- it applies a deterministic per-genre shuffle before assigning rows to splits
- it uses `train_n_val_test_split_ratio` for train allocation
- the remaining rows are split evenly across validation and test
- FMA-medium rows keep their original integer `track_id`
- external rows get deterministic negative integer surrogate `track_id` values
- all appended rows preserve provenance through `sample_id` and `source`
- rerunning the script is idempotent: existing FMA-medium rows and external `source + filepath` rows are skipped

Example dry run to a separate directory:

```bash
python MelCNN-MGR/preprocessing/load_extra_samples_for_small_dataset_splits.py \
    --out-dir /tmp/melcnn_small_extra_splits
```

By default the script overwrites the live `train_small.parquet`, `val_small.parquet`, and `test_small.parquet` in `MelCNN-MGR/data/processed/`.

### 7. Supplementation status

`extra_samples_for_small_dataset.json` remains the candidate-selection artifact, while
`load_extra_samples_for_small_dataset_splits.py` is the step that materializes the
selected rows into the final `small` split parquet files.

Because this workflow now allocates extra rows into train, validation, and test,
those supplemented validation/test splits define a new benchmark variant. If you want
to keep the original FMA validation/test benchmark untouched, write the supplemented
outputs to a separate directory first and train from those explicit paths.

For provenance-aware EDA of this workflow, use:

- `MelCNN-MGR/notebooks/Data-Understanding-Train-Val-Test-Genre-Distribution.ipynb` for the generic split-shape sanity check
- `MelCNN-MGR/notebooks/Data-Understanding-Train-Val-Test-Genre-Distribution-Supplementation-Aware.ipynb` for official-vs-current-vs-projected small-split comparisons

### 8. Python environment

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

### `MelCNN-MGR/notebooks/baseline_logmel_cnn_v21.py`
**Goal 2 — 15-second Extended Version (Long-Context Version)**

Similar to v20a1 but trained on **15-second clips** to capture more temporal context,
using random cropping during training to improve robustness.

| Property | Value |
|---|---|
| Feature | Log-Mel spectrogram — 128 bands, 15 s clips (random-crop) |
| Input shape | `(128, 1291, 1)` |
| Random Crop | 15.0 s from original 30.0 s clip |
| Augmentation | Freq/Time masking + Random temporal cropping |
| Cache | `logmel_v21_15s` (separate from v20a) |
| Run artifacts | `MelCNN-MGR/models/logmel-cnn-v21-<timestamp>/` |

**Run:**
```bash
cd MelCNN-MGR/notebooks
python baseline_logmel_cnn_v21.py
```

---

### `MelCNN-MGR/notebooks/baseline_logmel_cnn_v20a1.py`
**Goal 2 — Quality-improved Log-Mel baseline (production-ready)**

The most stable and optimized 10-second reference script in the project. Refines
the original v20a by adding a three-phase dynamic F1 checkpoint frequency,
random temporal cropping (10s window), and enhanced model loading stability.

| Property | Value |
|---|---|
| Feature | Log-Mel spectrogram — 128 bands, 10 s clips (random-crop) |
| Input shape | `(128, 861, 1)` |
| Architecture | 4 × (Conv2D + BN + MaxPool(2,2)) → GAP → Dropout(0.3) → Dense |
| Optimizer | AdamW lr=1e-3, weight decay 1e-4 |
| Checkpoints | Dynamic (Phase 1: every 3, Phase 2: every 2, Phase 3: every 1) |
| Random Crop | 10.0 s random window from 30 s clip |
| Inference | 3-crop (early/middle/late averaged probabilities) |
| Epochs | 50 (with CosineAnnealingWarmup) |
| Run artifacts | `MelCNN-MGR/models/logmel-cnn-v20a1-<timestamp>/` |

**Run:**
```bash
cd MelCNN-MGR/notebooks
python baseline_logmel_cnn_v20a1.py
```

---

### `MelCNN-MGR/notebooks/baseline_logmel_cnn_v20a.py`
**Goal 2 — Log-Mel baseline (legacy v20a script)**

Further reading: [`docs/Implementation Guide - baseline_logmel_cnn_v20a.md`](docs/Implementation%20Guide%20-%20baseline_logmel_cnn_v20a.md)

---

## Reusable Inference Modules

Each trained model has a companion inference module that loads the saved `.keras`
model and `norm_stats.npz` from a run directory and classifies new audio files
without re-running the training notebook.

| Training source | Inference module | Example usage script |
|---|---|---|
| `baseline_logmel_cnn_v21.py` | `MelCNN-MGR/inference_logmel_v20a1.py` | `MelCNN-MGR/examples/inference_v20a1_v21_batch.py` |
| `baseline_logmel_cnn_v20a1.py` | `MelCNN-MGR/inference_logmel_v20a1.py` | `MelCNN-MGR/examples/inference_v20a1_v21_batch.py` |
| `baseline_logmel_cnn_v20a.py` | `MelCNN-MGR/inference_logmel_v20a.py` | `MelCNN-MGR/examples/inference_logmel_v20a_example.py` |
| `baseline_logmel_cnn_v20.ipynb` | `MelCNN-MGR/inference_logmel_v20.py` | `MelCNN-MGR/examples/inference_logmel_v20_example.py` |
| `baseline_mfcc_cnn_v5.ipynb` | `MelCNN-MGR/inference_mfcc_v5.py` | `MelCNN-MGR/examples/inference_mfcc_v5_example.py` |

> [!NOTE]
> `inference_logmel_v20a1.py` is the most robust and features built-in custom schedule registration. It is used for both v20a1 (10s) and v21 (15s) models.

**Example usage (v20a1/v21 Batch):**
```bash
# Process several files using v21 (15s) model
python MelCNN-MGR/examples/inference_v20a1_v21_batch.py \
    --run-dir MelCNN-MGR/models/logmel-cnn-v21-20260309-120000 \
    --files track1.mp3 track2.wav
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
| 2026-03-09 | [Manifest refresh and small-dataset supplementation workflow](dev-logs/2026-03-09-manifest-refresh-and-small-dataset-supplementation.md) |
| 2026-03-08 | [Reusable inference modules](dev-logs/2026-03-08-inference-modules.md) |
| 2026-03-08 | [Baseline Log-Mel CNN v20a1](dev-logs/2026-03-08-baseline-logmel-cnn-v20a1.md) |
| 2026-03-06 | [Log-Mel v10 → v20 update](dev-logs/2026-03-06-logmel-v10-v20-update.md) |
| 2026-03-06 | [Baseline Log-Mel CNN v10](dev-logs/2026-03-06-baseline-logmel-cnn-v10.md) |
| 2026-03-05 | [Training pipeline improvements](dev-logs/2026-03-05-melcnn-mgr-training-pipeline-improvements.md) |
| 2026-03-04 | [MelCNN-MGR preprocessing pipeline](dev-logs/2026-03-04-melcnn-mgr-preprocessing-pipeline.md) |
| 2026-03-04 | [Dequantization warning fix](dev-logs/2026-03-04-dequantization-warning-fix.md) |

