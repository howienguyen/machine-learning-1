# MelCNN-MGR: The Production-Like Pipeline

*Date: 2026-03-10*

This document provides a comprehensive, end-to-end description of the MelCNN-MGR music genre recognition system — from raw audio acquisition through model training and inference. It covers every stage of the production-like pipeline, the data flow between stages, the key design decisions, and the configuration that governs each step.

---

## Table of Contents

1. [Solution Overview](#1-solution-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Stage 0: Data Acquisition — Jamendo Track Downloading](#3-stage-0-data-acquisition--jamendo-track-downloading)
4. [Stage 0.5: Audio Segment Extraction from Jamendo Downloads](#4-stage-05-audio-segment-extraction-from-jamendo-downloads)
5. [Data Sources](#5-data-sources)
6. [Stage 1: Unified Manifest Builder — `1_build_all_datasets_and_samples_v1_1.py`](#6-stage-1-unified-manifest-builder--1_build_all_datasets_and_samples_v1_1py)
7. [Stage 2: Log-Mel Feature Builder — `2_build_log_mel_dataset.py`](#7-stage-2-log-mel-feature-builder--2_build_log_mel_datasetpy)
8. [Stage 3: Exploratory Data Analysis — `MelCNN_MGR_Manifest_LogMel_EDA.ipynb`](#8-stage-3-exploratory-data-analysis--melcnn_mgr_manifest_logmel_edaipynb)
9. [Stage 4: Model Training — `logmel_cnn_v1.py`](#9-stage-4-model-training--logmel_cnn_v1py)
10. [Inference](#10-inference)
11. [Configuration Reference — `settings.json`](#11-configuration-reference--settingsjson)
12. [Directory Layout and Artifact Map](#12-directory-layout-and-artifact-map)
13. [Design Decisions and Rationale](#13-design-decisions-and-rationale)

---

## 1. Solution Overview

MelCNN-MGR (Mel-spectrogram Convolutional Neural Network for Music Genre Recognition) is a multi-class audio classification system that maps fixed-length audio clips, configured through `settings.json`, to one of 10 musical genres:

> **Blues · Classical · Country · Electronic · Folk · Hip-Hop · Jazz · Metal · Pop · Rock**

The system uses **Log-Mel spectrograms** as the input representation — a 2D time-frequency image computed from raw audio — fed into a **2D Convolutional Neural Network (CNN)**. This approach treats genre classification as an image-recognition-style problem over spectral "pictures" of sound.

### Goals

| # | Goal | Description |
|---|------|-------------|
| 1 | **Fair comparison** | Compare MFCC-based and Log-Mel-based CNN approaches for genre recognition under controlled conditions. |
| 2 | **Quality baseline** | Establish a strong, reproducible Log-Mel CNN baseline using modern training techniques (SpecAugment, AdamW, cosine annealing, class weighting). |
| 3 | **Reproducibility** | Every step — from manifest building to model evaluation — produces deterministic, auditable artifacts. |
| 4 | **Multi-source data** | Merge audio from multiple datasets (FMA, MTG-Jamendo, GTZAN, Dortmund) under a single unified manifest with controlled contribution ratios. |

---

## 2. Pipeline Architecture

The full production-like pipeline consists of five logical stages connected by well-defined parquet/npy artifacts:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     UPSTREAM DATA ACQUISITION                          │
│                                                                        │
│   download_by_genre_limits.py                                          │
│        │                                                               │
│        ▼                                                               │
│   genre_downloads/  (Jamendo MP3s by genre folder)                     │
│        │                                                               │
│        ▼                                                               │
│   extract_mtg_processed_samples.py                                     │
│        │                                                               │
│        ▼                                                               │
│   additional_datasets/data/mtg-jamendo/  (fixed-length WAV segments)   │
│                                                                        │
│   + additional_datasets/data/gtzan/                                    │
│   + additional_datasets/data/dortmund-university/                      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             │  FMA/fma_medium/ + FMA/fma_metadata/
                             │        │
                             ▼        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Manifest Builder                                             │
│  MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py     │
│                                                                        │
│  Outputs:                                                              │
│    → manifest_fma_datasets.parquet        (file-level FMA discovery)   │
│    → manifest_additional_datasets.parquet (file-level extra discovery) │
│    → manifest_fma_all_samples.parquet     (segment-level FMA expansion)│
│    → manifest_additional_all_samples.parquet (extra segment expansion) │
│    → manifest_final_samples.parquet  (selected segments + splits)      │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Log-Mel Feature Builder                                      │
│  MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py                   │
│                                                                        │
│  Outputs:                                                              │
│    → cache/logmel_dataset_<sample_length_sec>s/train/ (.npy log-mels)  │
│    → cache/logmel_dataset_<sample_length_sec>s/val/                    │
│    → cache/logmel_dataset_<sample_length_sec>s/test/                   │
│    → cache/logmel_dataset_<sample_length_sec>s/logmel_manifest_*.parquet│
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3 (optional): EDA Notebook                                      │
│  MelCNN-MGR/notebooks/MelCNN_MGR_Manifest_LogMel_EDA.ipynb             │
│                                                                        │
│  Reads all intermediate artifacts for visual data quality auditing.    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Model Training                                               │
│  MelCNN-MGR/notebooks/logmel_cnn_v1.py                                 │
│                                                                        │
│  Outputs:                                                              │
│    → models/logmel-cnn-v1-{timestamp}/logmel_cnn_v1.keras              │
│    → models/logmel-cnn-v1-{timestamp}/best_model_macro_f1.keras        │
│    → models/logmel-cnn-v1-{timestamp}/norm_stats.npz                   │
│    → models/logmel-cnn-v1-{timestamp}/run_report_logmel_cnn_v1.json    │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  INFERENCE                                                             │
│  MelCNN-MGR/inference_logmel_v20a.py  (MelCNNInference class)          │
│  MelCNN-MGR/examples/inference_logmel_v20a_example.py                  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key principle:** Each stage produces self-contained artifacts (parquet manifests, `.npy` files, `.keras` models) that the next stage consumes by path. Stages can be re-run independently — for example, retraining without re-extracting features, or refreshing features without rebuilding manifests.

---

## 3. Stage 0: Data Acquisition — Jamendo Track Downloading

**Script:** `download_by_genre_limits.py` (workspace root, ~487 lines)

### Purpose

Download Creative-Commons licensed music tracks from the [Jamendo](https://www.jamendo.com) API, organized by primary genre, to supplement the main FMA dataset with additional genre coverage — particularly for under-represented genres like Country, Pop, and Blues.

### How It Works

1. **Load Jamendo metadata** from `additional_datasets/mtg-jamendo-dataset-repo/data/autotagging_genre.tsv`.
2. **Parse genre tags** from the TSV's structured `genre---<name>` format. Each track may carry multiple genre tags (e.g., `genre---pop`, `genre---rock`).
3. **Filter by tag count**: Only tracks with ≤ `MAX_GENRE_TAGS_PER_TRACK` (default: 2) genre tags are eligible. This keeps primarily genre-pure tracks.
4. **Select tracks per genre**: For each genre in `GENRE_LIMITS`, pick up to the limit. Tracks with fewer tags are prioritized first (tag-count ascending, then track-ID ascending).
5. **Download**: Hit the Jamendo v3.0 API for each track's download URL and save to `genre_downloads/<genre>/<track_id>.mp3`.
6. **Failure handling**: Write failed track IDs to `genre_downloads/failed_<run_id>.tsv`. The `--mode retry-failed` flag can re-attempt failed downloads.

### Configuration Constants

| Constant | Default | Purpose |
|----------|---------|---------|
| `GENRE_LIMITS` | `{"country": 150, "pop": 200, "blues": 350}` | Max tracks to download per genre |
| `MAX_GENRE_TAGS_PER_TRACK` | 2 | Genre-purity filter threshold |
| `META_FILE` | `additional_datasets/mtg-jamendo-dataset-repo/data/autotagging_genre.tsv` | Jamendo metadata TSV |
| `SLEEP_BETWEEN` | 0.2s | Rate-limit between API calls |

### Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| Downloaded MP3s | `genre_downloads/<genre>/<track_id>.mp3` | Raw Jamendo audio files per genre folder |
| Selection TSVs | `genre_downloads/<genre>_track_ids_<run_id>.tsv` | Track IDs selected for each genre |
| Genre mapping | `genre_downloads/selected_track_genre_mapping_<run_id>.tsv` | Track → genre assignment map |
| Failed TSV | `genre_downloads/failed_<run_id>.tsv` | Tracks that failed to download |

### CLI Usage

```bash
# Fresh download
python download_by_genre_limits.py

# Retry previously failed downloads
python download_by_genre_limits.py --mode retry-failed --failed-tsv genre_downloads/failed_<run_id>.tsv
```

---

## 4. Stage 0.5: Audio Segment Extraction from Jamendo Downloads

**Script:** `extract_mtg_processed_samples.py` (workspace root, ~282 lines)

### Purpose

Extract fixed-length audio segments from the downloaded Jamendo MP3s and convert them to WAV format, ready for ingestion by the manifest builder. This step bridges the raw downloads (variable-length MP3s) and the standardized `additional_datasets/data/` directory structure expected downstream.

### How It Works

1. **Scan** all MP3/WAV files in `genre_downloads/<genre>/` subfolders.
2. **Filter by duration**: Skip files shorter than `--min-duration-seconds` (default: 30s).
3. **Determine segment count** per file:
   - duration ≤ 30s → skip (below min-duration threshold)
   - 30s < duration ≤ 60s → extract **1** segment
   - 60s < duration → extract up to `--max-num-segments` (default: 3) segments
4. **Place segments**: Use an edge-buffer strategy (default: 20s) to avoid intro/outro silence. For multiple segments, positions are evenly spaced between `[edge_buffer, duration - edge_buffer - segment_length]`.
5. **Write** each segment as `<stem>__seg<N>_start<seconds>s.wav` in `mtg-processed-samples/<genre>/`.

### Configuration Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `--segment-seconds` | 11.0 | Length of each extracted segment (slightly > 10s for padding margin) |
| `--max-num-segments` | 3 | Max segments per file |
| `--min-duration-seconds` | 30.0 | Skip files shorter than this |
| `--edge-buffer-seconds` | 20.0 | Avoid first/last N seconds of track |

### Outputs

| Artifact | Path | Description |
|----------|------|-------------|
| WAV segments | `mtg-processed-samples/<genre>/<stem>__seg<N>_start<Xs>s.wav` | Fixed-length audio segments for downstream use |

### CLI Usage

```bash
python extract_mtg_processed_samples.py \
    --input-dir genre_downloads \
    --output-dir mtg-processed-samples \
    --segment-seconds 11 \
    --max-num-segments 3 \
    --min-duration-seconds 30 \
    --edge-buffer-seconds 20
```

After extraction, the resulting WAV files are placed under `additional_datasets/data/mtg-jamendo/` (organized by genre subfolder) for the manifest builder to discover.

---

## 5. Data Sources

The pipeline fuses audio from multiple sources into a single unified training set. All sources feed into Stage 1 via two entry points: the FMA metadata path and the `additional_datasets/data/` directory tree.

### 5.1 FMA (Free Music Archive) — Primary Source

| Property | Value |
|----------|-------|
| **Role** | Primary dataset providing the majority of training samples |
| **Subset used** | `medium` (25,000 30-second clips, 16 top-level genres) |
| **Audio location** | `FMA/fma_medium/` (MP3 files organized as `<tid[:3]>/<tid>.mp3`) |
| **Metadata** | `FMA/fma_metadata/tracks.csv` (two-level column header, 106,574 total tracks) |
| **Official split** | `tracks.csv` contains `(set, split)` ∈ {training, validation, test} and `(set, subset)` ∈ {small, medium, large, full} |
| **Genre labels** | `(track, genre_top)` — one of 16 root genres per track |
| **Artist filter** | Tracks by the same artist belong to a single split (prevents artist-effect leakage) |
| **10 target genres** | FMA Medium is filtered to only the 10 genres defined in `settings.json` (`International` is excluded) |

FMA provides the backbone of the dataset. Its source clips are segmented into multiple fixed-length samples according to `settings.data_sampling_settings.sample_length_sec`.

### 5.2 Additional Datasets — Supplementary Sources

All supplementary audio lives under `additional_datasets/data/<source_name>/<genre_folder>/`. The manifest builder's `collect_additional_candidates()` function auto-discovers these by walking the directory tree.

| Source | Location | Notes |
|--------|----------|-------|
| **MTG-Jamendo** | `additional_datasets/data/mtg-jamendo/<genre>/` | Processed segments from `extract_mtg_processed_samples.py`. ~11s WAV files. Folder names like `hiphop` are alias-mapped to canonical genre `Hip-Hop`. |
| **GTZAN** | `additional_datasets/data/gtzan/<genre>/` | Classic 30s WAV genre dataset (1,000 clips, 10 genres). |
| **Dortmund University** | `additional_datasets/data/dortmund-university/<genre>/` | Additional academic audio collection. |

### Genre Alias Resolution

The manifest builder maps non-standard folder names to canonical genre labels via `_genre_alias_targets()`:

| Folder key | → Canonical genre(s) |
|------------|---------------------|
| `hiphop` | `Hip-Hop` |
| `raphiphop` | `Hip-Hop` |
| `folkcountry` | `Folk`, `Country` (both, if both are target genres) |
| Direct matches (e.g., `blues`, `classical`) | Matched case-insensitively against `target_genres` |

### Contribution Control

The settings parameter `additional_samples_contribution_ratio_expected_each_genre` (default: 0.39) controls the **maximum fraction** of final samples per genre that may come from additional sources. This prevents supplementary data from overwhelming the primary FMA distribution while ensuring under-represented genres receive meaningful augmentation.

---

## 6. Stage 1: Unified Manifest Builder — `1_build_all_datasets_and_samples_v1_1.py`

**Script:** `MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py` (~1,800 lines)

### Purpose

Build auditable, intermediate parquet manifests that unify all data sources into a single schema, expand audio files into fixed-length sample segments, and assign deterministic train/validation/test splits. This is the heart of the data pipeline — it defines **what data exists, what is usable, and how it is divided**.

### Execution Modes

The script supports three modes via `--mode`:

| Mode | Description |
|------|-------------|
| `stage1` | Build the selected Stage 1a and Stage 1b per-source manifests only |
| `stage2` | Build `manifest_final_samples.parquet` from existing Stage 1 outputs |
| `both` | Execute both stages sequentially (default) |

This two-stage design allows re-running split assignment (Stage 2) without re-probing thousands of audio files (Stage 1).

### Stage 1: File-Level Discovery and Segment Expansion

#### Step 1a — FMA Candidate Loading

Two paths are supported, selected automatically:

- **Cached manifest path**: If `metadata_manifest_medium.parquet` exists (previously built by `build_manifest.py`), load it directly without probing audio files.
- **From-scratch path** (`--force-rescan`): Load `tracks.csv` directly, filter by subset + target genres, resolve filepaths, probe actual audio durations via `ffprobe` or `soundfile`, and assign reason codes.

**Reason codes** assigned to each FMA track (priority order):

| Code | Meaning |
|------|---------|
| `NO_AUDIO_FILE` | Expected MP3 path does not exist on disk |
| `AUDIO_READ_FAILED` | File exists but duration could not be probed |
| `NO_LABEL` | `genre_top` is missing or empty |
| `EXCLUDED_LABEL` | Genre is in the exclusion set (e.g., `International`) |
| `NO_SPLIT` | Track has no valid split assignment |
| `TOO_SHORT` | Duration < `sample_length_sec - min_duration_delta` |
| `OK` | Passes all filters and is eligible for sampling |

#### Step 1b — Additional Dataset Scanning

`collect_additional_candidates()` walks `additional_datasets/data/`, probes every audio file's duration, maps folder names to canonical genres, and produces rows in the same schema as FMA.

#### Step 1c — Merge and Segment Expansion

```
FMA candidates + Additional candidates
    → per-source dataset manifests    → manifest_fma_datasets.parquet + manifest_additional_datasets.parquet
    → per-source sample manifests     → manifest_fma_all_samples.parquet + manifest_additional_all_samples.parquet
```

**Segment expansion rule**: For each eligible audio file:

```
num_segments = floor(duration_s / sample_length_sec)
```

Each segment gets a unique `sample_id` following the pattern:
```
<artifact_id>:seg<NNNN>
```

For example: `fma:12345:seg0000`, `fma:12345:seg0001`, `fma:12345:seg0002` for a 45s FMA track yielding three 15s segments under the current settings snapshot.

#### Duration Normalization

Measured audio durations are normalized using a conservative rounding scheme:

```python
ceil(round(duration_s, 1) - 1e-6)
```

This ensures values like 29.90s and 29.99s both normalize to 30s for consistent segment counting.

### Stage 2: Final Split Assignment

`assign_final_splits()` produces `manifest_final_samples.parquet` — the final training-ready manifest.

#### Split Assignment Algorithm

For each genre independently:

1. **Compute targets** from `number_of_samples_expected_each_genre` (default: 1,300):
   - Training: `floor(1300 × 0.8) = 1040`
   - Validation: `(1300 - 1040) / 2 = 130`
   - Test: `130`

2. **Group by source sample identity**: All segments from the same audio file (identified by stripping the `:segNNNN` suffix from `sample_id`) are assigned to the same split. This prevents **segment leakage** — if segments from the same track appeared in both training and test, the model could memorize track-specific features.

3. **Allocate additional-source quota**: Up to `additional_samples_contribution_ratio` of each split's target comes from supplementary sources.

4. **Assign splits**:
    - Deterministically shuffle the additional-source sample manifest before grouped split assignment
   - Shuffle primary (FMA) groups (seeded), assign to remaining quota
   - Fallback: if either pool is short, the other fills the gap

5. **Determinism**: All shuffles use `seed + genre_index` offsets for full reproducibility.

### Output Artifacts — Stage 1

| Artifact | Schema | Description |
|----------|--------|-------------|
| `manifest_fma_datasets.parquet` | `source, artifact_id, source_track_id, track_id, genre_top, filepath, audio_exists, filesize_bytes, actual_duration_s, duration_s, reason_code, sampling_eligible, sampling_num_segments, sampling_exclusion_reason, manifest_origin` | One row per discovered FMA audio file candidate. |
| `manifest_additional_datasets.parquet` | Same schema as `manifest_fma_datasets.parquet` | One row per discovered additional-source audio file candidate. |
| `manifest_fma_all_samples.parquet` | `sample_id, source, genre_top, filepath, track_id, sample_length_sec, segment_index, segment_start_sec, segment_end_sec, total_segments_from_audio, duration_s, actual_duration_s, reason_code` | One row per fixed-length FMA segment from eligible files. |
| `manifest_additional_all_samples.parquet` | Same schema as `manifest_fma_all_samples.parquet` | One row per fixed-length additional-source segment from eligible files. |
| `manifest_final_samples.parquet` | Same as `all_samples` + `final_split` column | Selected subset with deterministic split assignment (training/validation/test). |

### CLI Usage

```bash
# Full pipeline (both stages)
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py

# Stage 1 only (file discovery + segment expansion)
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py --mode stage1

# Stage 2 only (split assignment from existing Stage 1 output)
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py --mode stage2

# Force FMA rebuild from tracks.csv (ignore cached manifest)
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py --force-rescan
```

---

## 7. Stage 2: Log-Mel Feature Builder — `2_build_log_mel_dataset.py`

**Script:** `MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py` (~625 lines)

### Purpose

Transform the segment-level manifest from Stage 1 into actual **Log-Mel spectrogram `.npy` files** organized by train/val/test split, with corresponding parquet index files. This bridges raw audio → model-ready tensors.

### How It Works

1. **Load** `manifest_final_samples.parquet` and group by `final_split`.
2. **For each segment** in each split:
   a. Load the audio segment (respecting `segment_start_sec` / `segment_end_sec`) using either `ffmpeg` subprocess or `librosa.load()`.
   b. Resample to `SAMPLE_RATE` (default: 22,050 Hz).
   c. Normalize the waveform to exactly `sample_length_sec × SAMPLE_RATE` samples via center-crop or center-pad.
   d. Compute the Log-Mel spectrogram: `log(1 + mel_spectrogram)`.
   e. Pad or truncate to the exact target shape `(N_MELS, N_FRAMES)`.
   f. Save the result as `<sample_id_hash>.npy`.
3. **Write split parquet indexes** mapping `sample_id → .npy path, genre, metadata`.
4. **Generate build report** with per-split/per-genre statistics and error counts.

### Audio Processing Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `SAMPLE_RATE` | 22,050 Hz | Standard audio sample rate |
| `N_MELS` | 192 | Number of mel filter bands |
| `N_FFT` | 512 | FFT window size |
| `HOP_LENGTH` | 256 | Hop between STFT frames |
| `sample_length_sec` | manifest-driven at runtime; default output-root suffix follows settings with a `15.0` fallback if settings cannot be read | Segment duration |
| `N_FRAMES` | computed: `int(sample_length_sec × SAMPLE_RATE / HOP_LENGTH)` | Exact number of time-axis frames used by the current builder |

The Log-Mel computation:

```
        raw audio (sample_length_sec, 22050 Hz)
            │
            ▼
    STFT(n_fft=512, hop=256)
            │
            ▼
    mel filterbank (192 bands)
            │
            ▼
    log(1 + S)  (log compression)
            │
            ▼
    NumPy array shape (192, N_FRAMES)
```

### Audio Backend Selection

The builder supports two backends:

1. **ffmpeg** (preferred): Uses a subprocess call to decode the specific segment range directly from the source file. More efficient for MP3s and supports seeking.
2. **librosa**: Fallback when ffmpeg is unavailable. Loads the full file and slices in memory.

Backend selection is automatic: ffmpeg is used if `ffmpeg` is on `$PATH`, otherwise librosa.

### Parallel Processing

Samples are processed using Python's `concurrent.futures.ProcessPoolExecutor` with configurable `--workers` (default: CPU count). Each worker processes one sample independently and returns a result tuple (success/failure + metadata).

### Output Structure

```
MelCNN-MGR/cache/logmel_dataset_<sample_length_sec>s/
├── config.json                          # Build configuration snapshot
├── build_report.txt                     # Human-readable build statistics
├── logmel_manifest_all.parquet          # Combined index across all splits
├── train/
│   ├── logmel_manifest_train.parquet    # Split index: sample_id → npy path, genre, etc.
│   ├── Blues/
│   │   ├── <hash>.npy                  # (192, N_FRAMES) float32 arrays
│   │   └── ...
│   ├── Rock/
│   └── ...
├── val/
│   ├── logmel_manifest_val.parquet
│   └── <genre>/<hash>.npy
└── test/
    ├── logmel_manifest_test.parquet
    └── <genre>/<hash>.npy
```

### Split Parquet Schema

Each `logmel_manifest_{split}.parquet` contains:

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | string | Unique sample identifier (e.g., `fma:12345:seg0001`) |
| `logmel_path` | string | Relative path to the `.npy` file |
| `genre_top` | string | Genre label |
| `source` | string | Data source identifier (e.g., `fma-medium`, `mtg-jamendo`) |
| `split` | string | Split name: `train`, `val`, or `test` |
| `status` | string | `ok` or error description |

### CLI Usage

```bash
python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py

# With explicit worker count
python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py --workers 8
```

---

## 8. Stage 3: Exploratory Data Analysis — `MelCNN_MGR_Manifest_LogMel_EDA.ipynb`

**Notebook:** `MelCNN-MGR/notebooks/MelCNN_MGR_Manifest_LogMel_EDA.ipynb` (14 cells)

### Purpose

An interactive data quality audit notebook that loads every intermediate artifact produced by Stages 1 and 2 and produces visual diagnostics before training. This step is **optional** but strongly recommended for catching data issues before committing GPU time.

### What It Inspects

The notebook is organized into 8 logical sections:

#### Section 1: Initialization & Path Resolution
- Resolves workspace root by searching for `MelCNN-MGR/settings.json`
- Loads all path constants for downstream artifacts

#### Section 2: Helper Functions
Defines reusable utilities:
- `display_frame_overview()` — DataFrame shape, head, dtypes, null counts
- `imbalance_table()` — count/proportion/imbalance-ratio table for categorical variables
- `plot_bar()`, `plot_hist()`, `plot_heatmap()` — standardized visualization helpers
- `artifact_id_from_sample_id()` — strips `:segNNNN` suffix for group-level analysis
- `safe_read_json()`, `safe_read_parquet()` — error-tolerant file loading

#### Section 3: Load Configuration & Artifacts
- Loads `settings.json`, build configs, all three parquet manifests, and log-mel split indexes
- Builds a shape summary table comparing artifact existence and row/column counts

#### Section 4: Schema Contract Checks
- Validates that each parquet's columns match the expected schema
- Reports missing or extra columns per artifact

#### Section 5: File-Level Discovery Analysis
Analyzes the Stage 1a dataset manifests:
- Audio file counts by source, genre, reason code, eligibility
- Duration distributions (histograms, summary statistics)
- Identification of skipped or unusable files

#### Section 6: Segment Expansion & Final Split Audits
Analyzes the transition from the Stage 1b sample manifests → `manifest_final_samples.parquet`:
- **Segment retention ratio**: How many emitted segments were selected vs. dropped
- **Genre distribution**: Bar plots of final segment counts by genre
- **Split balance**: Genre × split crosstab (counts and percentages)
- **Segment leakage check**: Verifies no audio-file-level ID appears in multiple splits
- **Additional-source contribution**: FMA vs. supplementary data share per genre, compared against the configured target ratio

#### Section 7: Downstream Log-Mel Cache Audit
Checks the log-mel feature builder's outputs:
- Do split indexes exist? How many rows?
- Alignment check: Do all `manifest_final_samples` entries have corresponding log-mel `.npy` files?
- Status distribution (ok vs. error)

#### Section 8: Readiness Summary
Computes and displays 9 key findings as a markdown summary:
- Final split counts, smallest genre, dominant file-level genre
- Top reason code for skipped files
- Mean emitted segments per audio file
- Training share percentage
- Additional-source gap analysis
- Log-mel readiness flag

### Key Diagnostics Produced

| Diagnostic | Purpose |
|------------|---------|
| Genre distribution bar plot | Visual class balance check |
| Genre × split heatmap | Split stratification verification |
| Duration histograms | Detect outlier audio files |
| Leakage count | Must be 0 for valid splits |
| Additional-source contribution table | Verify supplementary data ratio |
| Log-mel alignment summary | Catch missing feature files before training |

---

## 9. Stage 4: Model Training — `logmel_cnn_v1.py`

**Script:** `MelCNN-MGR/notebooks/logmel_cnn_v1.py` (~1,170 lines)

### Purpose

Train a 2D CNN on prebuilt Log-Mel spectrograms from the Stage 2 cache. This is the main training entry point for the production-like pipeline.

### Architecture

The model is a 5-block 2D CNN with the following structure:

```
Input: (N_MELS, N_FRAMES, 1) — log-mel spectrogram as a single-channel "image"
    │
    ▼
┌───────────────── Block 1 ─────────────────┐
│ Conv2D(32, 3×3, padding=same)             │
│ BatchNormalization                         │
│ ReLU                                       │
│ SpatialDropout2D                           │
│ MaxPooling2D(2×2)                          │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────── Block 2 ─────────────────┐
│ Conv2D(64, 3×3, padding=same)             │
│ BatchNorm → ReLU → SpatialDropout → Pool  │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────── Block 3 ─────────────────┐
│ Conv2D(128, 3×3, padding=same)            │
│ BatchNorm → ReLU → SpatialDropout → Pool  │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────── Block 4 ─────────────────┐
│ Conv2D(256, 3×3, padding=same)            │
│ BatchNorm → ReLU → SpatialDropout → Pool  │
└───────────────────────────────────────────┘
    │
    ▼
┌───────────────── Block 5 ─────────────────┐
│ Conv2D(256, 3×3, padding=same)            │
│ BatchNorm → ReLU → SpatialDropout → Pool  │
└───────────────────────────────────────────┘
    │
    ▼
GlobalAveragePooling2D
    │
    ▼
Dense(N_CLASSES, softmax)
```

**Design rationale:**
- `3×3` kernels allow the network to learn local spectro-temporal patterns (unlike the v1.0 `(128,10)` kernel that collapsed the entire frequency axis).
- `BatchNormalization` stabilizes gradients and allows higher learning rates.
- `SpatialDropout2D` drops entire feature maps (more effective than element-wise dropout for conv layers).
- `GlobalAveragePooling2D` instead of `Flatten` reduces overfitting by eliminating dense connections from spatial positions.
- Progressive channel widening (32→64→128→256→256) captures increasingly abstract features.

### Training Recipe

| Component | Setting | Purpose |
|-----------|---------|---------|
| **Optimizer** | AdamW (weight decay) | Better generalization than plain Adam |
| **LR schedule** | Cosine annealing with warmup: warmup=3 epochs, lr_max=1e-3, lr_min=1e-6 | Smooth convergence with warm start |
| **Loss** | Categorical crossentropy with label_smoothing=0.02 | Regularization against overconfident predictions |
| **Epochs** | 99 (max) | High ceiling; early stopping controls actual length |
| **Batch size** | 32 | Balanced between gradient noise and memory |
| **Class weights** | `compute_class_weight('balanced', ...)` | Correct for genre imbalance |
| **SpecAugment** | Frequency masking + time masking (training only) | Data augmentation for robustness |
| **EarlyStopping** | Monitor `val_loss`, patience configurable | Prevent overfitting |

### SpecAugment Data Augmentation

Applied only during training, SpecAugment randomly masks rectangular regions of the Log-Mel spectrogram:

- **Frequency masking**: Random contiguous band of mel bins set to zero
- **Time masking**: Random contiguous span of time frames set to zero

This forces the model to learn from partial spectral information, improving generalization.

### Preprocessing Pipeline (`tf.data`)

```
Split Parquet (logmel_manifest_train.parquet)
    │
    ▼
Read .npy paths → tf.data.Dataset
    │
    ▼
Parallel load: np.load(path)  [tf.py_function]
    │
    ▼
Z-normalize: (x - mean) / std  [computed from training set only]
    │
    ▼
SpecAugment (training only)
    │
    ▼
Batch(32) → Prefetch
    │
    ▼
Model input: (batch, N_MELS, N_FRAMES, 1)
```

**Normalization**: Per-band (frequency-axis) mean and standard deviation are computed from the entire training set before training begins. The same statistics are saved to `norm_stats.npz` and applied at inference time.

### Callbacks

| Callback | Behavior |
|----------|----------|
| **EarlyStopping** | Stops training if `val_loss` hasn't improved for N epochs |
| **ReduceLROnPlateau** | Reduces learning rate when `val_loss` plateaus |
| **_MacroF1Checkpoint** | Custom callback that evaluates Macro-F1 on the validation set every N epochs and saves the best model. Always evaluates on the final epoch to guarantee a `best_model_macro_f1.keras` is produced. |

### Class Weight Handling

Class weights are computed using `sklearn.compute_class_weight('balanced', ...)`:

```python
present_train_classes = np.unique(train_labels)
weights = compute_class_weight('balanced', classes=present_train_classes, y=train_labels)
class_weight_dict = {cls_id: w for cls_id, w in zip(present_train_classes, weights)}
```

If any class is absent from training data, a warning is logged. This hardened approach prevents crashes when a rare genre is missing from a training split.

### Evaluation

After training, the script evaluates on the test set:

- **Test accuracy** and **test Macro-F1**
- **Per-genre precision, recall, F1** via `classification_report()` with explicit `labels=np.arange(N_CLASSES)` to ensure all genres appear even if missing from predictions
- **Confusion matrix** with explicit label set
- **Genre-level analysis**: Identifies best/worst performing genres

### Run Artifacts

Each training run produces a timestamped directory:

```
MelCNN-MGR/models/logmel-cnn-v1-YYYYMMDD-HHMMSS/
├── logmel_cnn_v1.keras           # Final-epoch model
├── best_model_macro_f1.keras     # Best Macro-F1 checkpoint
├── norm_stats.npz                # Training-set mean/std for inference
└── run_report_logmel_cnn_v1.json # Full metrics, config, timings
```

The JSON run report includes:
- All hyperparameters and configuration
- Per-epoch training/validation loss and accuracy
- Per-epoch learning rates
- Test metrics (accuracy, macro-F1, per-genre F1)
- Confusion matrix
- Section-level timing breakdown
- Runtime environment details (Python version, TF version, device)

### CLI Usage

```bash
python MelCNN-MGR/notebooks/logmel_cnn_v1.py
```

The script auto-detects the best available device (GPU > XPU > CPU) and configures TensorFlow accordingly.

---

## 10. Inference

**Module:** `MelCNN-MGR/inference_logmel_v20a.py`
**Example:** `MelCNN-MGR/examples/inference_logmel_v20a_example.py`

### Purpose

Standalone inference module for classifying audio files using a trained Log-Mel CNN model. Designed to be independent of the training pipeline — it loads only a model `.keras` file, `norm_stats.npz`, and genre labels from a run directory.

### Inference Modes

| Mode | Description |
|------|-------------|
| `single_crop` | Extract a center 10s clip, compute log-mel, predict once |
| `three_crop` | Extract 3 deterministic 10s clips (early/middle/late), predict each, average probabilities |

The `three_crop` mode generally yields higher accuracy by reducing the dependence on a single 10s window.

### Inference Parameters

| Parameter | Value | Match to Training? |
|-----------|-------|-------------------|
| Sample rate | 22,050 Hz | Yes |
| N_MELS | 128 | Note: inference module uses 128, training v1 uses 192 — use the matching inference module for the model version |
| N_FFT | 512 | Yes |
| HOP_LENGTH | 256 | Yes |
| Clip duration | 10.0s | Yes |

### Usage

```python
from inference_logmel_v20a import MelCNNInference

engine = MelCNNInference("MelCNN-MGR/models/logmel-cnn-v20a-<timestamp>")
result = engine.predict("path/to/audio.mp3", mode="three_crop")

print(result.genre)        # e.g., "Rock"
print(result.confidence)   # e.g., 0.87
print(result.top_k(3))     # Top 3 genre predictions with probabilities
```

### CLI Example

```bash
python MelCNN-MGR/examples/inference_logmel_v20a_example.py \
    --run-dir MelCNN-MGR/models/logmel-cnn-v20a-<ts> \
    --subset small --random 5
```

---

## 11. Configuration Reference — `settings.json`

**File:** `MelCNN-MGR/settings.json`

This single JSON file governs the data sampling behavior of the manifest builder and, transitively, the downstream dataset shape.

```json
{
    "data_sampling_settings": {
        "target_genres": [
            "Hip-Hop", "Pop", "Folk", "Rock", "Metal",
            "Electronic", "Classical", "Jazz", "Country", "Blues"
        ],
        "number_of_samples_expected_each_genre": 1300,
        "additional_samples_contribution_ratio_expected_each_genre": 0.39,
        "train_n_val_test_split_ratio_each_genre": 0.8,
        "sample_length_sec": 15,
        "min_duration_delta": 0.001
    }
}
```

### Parameter Descriptions

| Parameter | Value | Description |
|-----------|-------|-------------|
| `target_genres` | 10 genres | The canonical genre vocabulary. Only audio matching these labels is included. |
| `number_of_samples_expected_each_genre` | 1,300 | Target sample count per genre in the final manifest. Actual count depends on available audio. |
| `additional_samples_contribution_ratio_expected_each_genre` | 0.39 | Up to 39% of each genre's samples may come from supplementary sources. |
| `train_n_val_test_split_ratio_each_genre` | 0.8 | 80% training / 10% validation / 10% test per genre. |
| `sample_length_sec` | 15 | Each sample is a 15-second audio segment in the current settings snapshot. |
| `min_duration_delta` | 0.001 | Tolerance for minimum duration filter: min_duration = `sample_length_sec - 0.001`. |

### Derived Values

From these settings, the pipeline derives:

| Derived | Calculation | Value |
|---------|-------------|-------|
| Segments per 30s track | `floor(30 / 10)` | 3 |
| Training samples per genre | `floor(1300 × 0.8)` | 1,040 |
| Validation samples per genre | `(1300 - 1040) / 2` | 130 |
| Test samples per genre | `(1300 - 1040) / 2` | 130 |
| Total target samples (all genres) | `1300 × 10` | 13,000 |
| Min eligible duration | `10 - 0.001` | 9.999s |

---

## 12. Directory Layout and Artifact Map

```
machine-learning-1/                              ← Workspace root
│
├── download_by_genre_limits.py                  ← [Stage 0]  Jamendo downloader
├── extract_mtg_processed_samples.py             ← [Stage 0.5] Segment extractor
│
├── FMA/
│   ├── fma_medium/                              ← FMA audio (MP3s)
│   └── fma_metadata/
│       └── tracks.csv                           ← FMA track metadata
│
├── additional_datasets/
│   ├── data/
│   │   ├── mtg-jamendo/<genre>/                 ← Extracted Jamendo WAV segments
│   │   ├── gtzan/<genre>/                       ← GTZAN audio
│   │   └── dortmund-university/<genre>/         ← Dortmund audio
│   └── mtg-jamendo-dataset-repo/
│       └── data/autotagging_genre.tsv           ← Jamendo metadata TSV
│
├── genre_downloads/                             ← Raw Jamendo MP3 downloads
│   ├── <genre>/<track_id>.mp3
│   └── *.tsv                                   ← Selection/failure logs
│
├── MelCNN-MGR/
│   ├── settings.json                            ← Central configuration
│   │
│   ├── preprocessing/
│   │   ├── 1_build_all_datasets_and_samples_v1_1.py  ← [Stage 1] Manifest builder
│   │   └── 2_build_log_mel_dataset.py           ← [Stage 2] Feature builder
│   │
│   ├── data/processed/                          ← Stage 1 outputs
│   │   ├── manifest_fma_datasets.parquet
│   │   ├── manifest_additional_datasets.parquet
│   │   ├── manifest_fma_all_samples.parquet
│   │   ├── manifest_additional_all_samples.parquet
│   │   ├── manifest_final_samples.parquet
│   │   ├── build_report.txt
│   │   └── build_config.json
│   │
│   ├── cache/logmel_dataset_<sample_length_sec>s/ ← Stage 2 outputs
│   │   ├── config.json
│   │   ├── build_report.txt
│   │   ├── logmel_manifest_all.parquet
│   │   ├── train/
│   │   │   ├── logmel_manifest_train.parquet
│   │   │   └── <genre>/<hash>.npy
│   │   ├── val/
│   │   │   ├── logmel_manifest_val.parquet
│   │   │   └── <genre>/<hash>.npy
│   │   └── test/
│   │       ├── logmel_manifest_test.parquet
│   │       └── <genre>/<hash>.npy
│   │
│   ├── notebooks/
│   │   ├── MelCNN_MGR_Manifest_LogMel_EDA.ipynb ← [Stage 3] EDA notebook
│   │   └── logmel_cnn_v1.py                     ← [Stage 4] Training script
│   │
│   ├── models/
│   │   └── logmel-cnn-v1-<timestamp>/           ← Training run outputs
│   │       ├── logmel_cnn_v1.keras
│   │       ├── best_model_macro_f1.keras
│   │       ├── norm_stats.npz
│   │       └── run_report_logmel_cnn_v1.json
│   │
│   ├── inference_logmel_v20a.py                 ← Inference module
│   └── examples/
│       └── inference_logmel_v20a_example.py     ← Inference CLI example
│
└── docs/
    └── MelCNN-MGR-Production-Like-Pipeline.md   ← This document
```

---

## 13. Design Decisions and Rationale

### 13.1 Why Fixed-Length Segments?

The pipeline standardizes on fixed-length segments, controlled by `settings.data_sampling_settings.sample_length_sec`, rather than the original full source durations. Reasons:

1. **Data multiplication**: A 30s FMA track yields 3 training samples instead of 1, effectively tripling the dataset with real (non-synthetic) content.
2. **Real-world alignment**: Most streaming/radio use cases classify short clips, not full tracks.
3. **Memory efficiency**: Smaller spectrograms from short fixed-length clips reduce GPU memory per batch relative to full 30-second inputs.
4. **Augmentation diversity**: More segments per track means the model sees different parts of the same song as separate training examples.

**Trade-off**: Some genre cues (e.g., song structure, long-term rhythm patterns) may only be apparent over longer durations. The 3-crop inference mode partially mitigates this.

### 13.2 Why Log-Mel over MFCC?

| Aspect | MFCC | Log-Mel |
|--------|------|---------|
| Information | 13 coefficients (DCT-compressed) | 128–192 frequency bands (full resolution) |
| CNN suitability | Very compact; CNN has little spectral detail to work with | Rich 2D representation; CNN can learn local spectral patterns |
| Learnability | Pre-engineered compression; discards information the model might need | Preserves information; lets the model decide what matters |
| Empirical result | Lower accuracy with matched architecture | Higher accuracy once Conv1 kernel design was fixed (v10+) |

The project demonstrated that MFCC's aggressive dimensionality reduction (128 mel bands → 13 coefficients) discards information useful for genre recognition. Log-Mel spectrograms preserve the full mel-scale frequency resolution, giving the CNN more material to learn from.

### 13.3 Why A Manifest-Centric Pipeline?

Traditional audio ML pipelines often embed data discovery, filtering, and splitting logic inside the training script. This project separates these concerns:

- **Manifest builder** (Stage 1) handles all data discovery, quality gating, and split assignment → produces auditable parquet files.
- **Feature builder** (Stage 2) handles only audio → spectrogram conversion.
- **Training script** (Stage 4) consumes pre-built features; has no knowledge of raw audio paths or data filtering.

**Benefits:**
- Each stage can be re-run independently (e.g., retrain without re-extracting features).
- Parquet manifests serve as checkpoints that can be inspected, compared, or version-controlled.
- The EDA notebook (Stage 3) can audit data quality before committing to expensive training.
- Bugs in data selection don't require re-extracting features.

### 13.4 Why Multi-Source Data Fusion?

FMA alone has uneven genre coverage (e.g., fewer Country and Blues tracks). Supplementing with Jamendo, GTZAN, and Dortmund tracks:

- Improves representation for under-served genres
- Introduces audio variety (different recording conditions, artists, musical styles)
- Is controlled via `additional_samples_contribution_ratio` to prevent supplementary data from dominating

The contribution ratio of 39% was chosen to balance diversity against distribution dilution — enough to fill gaps without overwhelming the FMA-centric distribution the model was designed for.

### 13.5 Why Segment-Level Leakage Prevention?

When a long track is split into multiple fixed-length segments, those segments are highly correlated (same artist, same recording, overlapping musical content). If one segment is in training and another is in test, the model can achieve artificially high test accuracy by memorizing track-specific features rather than learning generalizable genre cues.

The manifest builder prevents this by grouping all segments from the same source audio file (identified by stripping the `:segNNNN` suffix from `sample_id`) and assigning the entire group to a single split.

### 13.6 Why SpecAugment?

SpecAugment (frequency + time masking) is a proven augmentation technique from speech recognition that transfers well to music:

- Forces the model to classify from partial spectral information
- Acts as regularization without changing labels or adding synthetic audio
- Particularly effective for preventing overfitting on small-to-medium datasets
- Applied only during training (not validation or test)

### 13.7 Why the Macro-F1 Best Checkpoint?

Under class imbalance, a model can achieve high accuracy by excelling at majority genres while failing on minority ones. Macro-F1 gives equal weight to every genre's F1 score, making it a better optimization target for a balanced genre classifier. The custom `_MacroF1Checkpoint` callback saves the model state that maximizes this metric on validation data, independently of the `val_loss`-based early stopping.

---

*End of document.*
