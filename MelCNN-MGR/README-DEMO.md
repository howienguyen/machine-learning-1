# MelCNN-MGR Demo and Production-Like Pipeline

This document summarizes the current production-like training path, the main preprocessing entry points, the active training entry point, the current inference/application boundary, and the main commands needed to run the demo flow.

## Current Training Path

The current production-like training path is:

```text
download_by_genre_limits.py --> extract_mtg_processed_samples.py

data sources: FMA & additional_datasets -->

MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py -->
MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py -->
MelCNN-MGR/model_training/2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb -->
MelCNN-MGR/model_training/logmel_cnn_v2_1.py
```

Interpretation:

1. optional external-source preparation happens first through genre-limited download and MTG extraction
2. the training corpus is then assembled from FMA plus `additional_datasets`
3. `1_build_all_datasets_and_samples_v1_1.py` is the main manifest-and-sample preprocessing script
4. `2_build_log_mel_dataset.py` is the main log-mel preprocessing script
5. the manifest/log-mel EDA notebook is used to inspect readiness before training
6. `logmel_cnn_v2_1.py` is the main current training entry point
7. `logmel_cnn_v2_1_exp.py` is the experimental sibling for softer regularization tests

## Main Preprocessing Scripts

The main preprocessing scripts are:

1. `MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py`
2. `MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py`

Current intent:

1. `1_build_all_datasets_and_samples_v1_1.py` builds the dataset manifests, sample manifests, and final selected sample manifest
2. `2_build_log_mel_dataset.py` converts the final sample manifest into split-grouped log-mel `.npy` features plus parquet indexes for training

## Active Training Scripts

The main current training scripts are:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py`
3. `MelCNN-MGR/model_training/logmel_cnn_v1.py` and `logmel_cnn_v1_1.py` as legacy baselines

Current intent:

1. `logmel_cnn_v2_1.py` is the primary current training script
2. `logmel_cnn_v2_1_exp.py` is for controlled experimental changes relative to v2.1
3. `v1` / `v1_1` remain useful as baseline references and for the currently documented inference service path

## Run Artifacts

Each `logmel_cnn_v2_1*` run creates a dedicated run directory under `MelCNN-MGR/models/`.

Typical contents:

```text
logmel-cnn-v2_1-YYYYMMDD-HHMMSS/
    best_model_macro_f1.keras
    logmel_cnn_v2_1.keras
    norm_stats.npz
    console_output.txt
    run_report_logmel_cnn_v2_1.json
```

For `logmel_cnn_v2_1_exp.py`, the filenames are the corresponding `v2_1_exp` variants.

Notes:

1. `console_output.txt` captures the console stream for the run after the run directory is created
2. `run_report_*.json` stores structured metadata, metrics, and artifact paths
3. the JSON report includes the path to `console_output.txt`

## Inference and Applications

The current inference and demo/application entry points are:

1. `MelCNN-MGR/inference_logmel_cnn_v1_1.py`
2. `MelCNN-MGR/inference_web_service/app.py`
3. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`

Important boundary:

1. the current inference module and web service are still built around the `v1.1` inference path
2. `MelCNN-MGR/inference_logmel_cnn_v1_1.py` is currently hardcoded to 10-second crops and its own fixed log-mel shape
3. the current `v2.1` training pipeline follows the log-mel dataset configuration, which currently defaults to `sample_length_sec = 15` from `MelCNN-MGR/settings.json`
4. because of that, do not assume a `v2.1` run directory is automatically compatible with the current inference/service modules unless the feature shape matches

Practical meaning:

1. use `v2.1` / `v2.1-exp` for current training experiments
2. use the current inference service path with a compatible `v1.1`-style run directory unless you deliberately update the inference module for `v2.1`

## Command Lines

### 1. Optional external-source preparation

Download genre-limited MTG/Jamendo audio:

```bash
python download_by_genre_limits.py
```

Extract processed MTG samples:

```bash
python extract_mtg_processed_samples.py
```

### 2. Build the manifest and sample inventory

Main script:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py
```

Smallest standalone runs for `1_build_all_datasets_and_samples_v1_1.py`:

Stage 1a, FMA only:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1a \
	--stage1a-sources fma
```

Stage 1a, additional datasets only:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1a \
	--stage1a-sources additional
```

Stage 1a, both sources:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1a \
	--stage1a-sources both
```

Stage 1b, FMA only:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1b \
	--stage1b-sources fma
```

Stage 1b, additional datasets only:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1b \
	--stage1b-sources additional
```

Stage 1b, both sources:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1b \
	--stage1b-sources both
```

Stage 1a plus Stage 1b in one run:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1
```

Stage 2 only from existing sample manifests:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage2
```

Full pipeline in one run:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode both \
	--stage1a-sources both \
	--stage1b-sources both
```

### 3. Build the log-mel dataset indexes/cache

Main script:

```bash
python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py
```

### 4. Review the manifest and downstream readiness

Open and run:

```text
MelCNN-MGR/model_training/2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb
```

### 5. Run model training

Primary current training script:

```bash
python MelCNN-MGR/model_training/logmel_cnn_v2_1.py
```

Experimental variant:

```bash
python MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py
```

Warm-start a brand new v2.1 run from an existing checkpoint:

```bash
python MelCNN-MGR/model_training/logmel_cnn_v2_1.py \
	--pretrained-model MelCNN-MGR/models/logmel-cnn-v1_1-20260311-025046/best_model_macro_f1.keras
```

Backbone-only warm-start when you want to reuse the backbone but rebuild the classifier head:

```bash
python MelCNN-MGR/model_training/logmel_cnn_v2_1.py \
	--pretrained-model MelCNN-MGR/models/logmel-cnn-v1_1-20260311-025046/best_model_macro_f1.keras \
	--pretrained-mode backbone_only
```

Backbone-only warm-start with an initial frozen-head stage:

```bash
python MelCNN-MGR/model_training/logmel_cnn_v2_1.py \
	--pretrained-model MelCNN-MGR/models/logmel-cnn-v1_1-20260311-025046/best_model_macro_f1.keras \
	--pretrained-mode backbone_only \
	--freeze-backbone-epochs 3
```

Notes:

1. this is a fresh new run, not a resume
2. `--pretrained-mode full_model` requires matching log-mel shape, class count, and `genre_classes` order
3. `--pretrained-mode backbone_only` transfers compatible backbone weights and rebuilds the final classifier head for the current genre set
4. `logmel_cnn_v2_1.py` and `logmel_cnn_v2_1_exp.py` write both `console_output.txt` and `run_report_*.json` into the new run directory

### 6. Run direct inference

Current direct inference path is the v1.1 inference module:

```bash
python MelCNN-MGR/inference_logmel_cnn_v1_1.py \
	--run-dir MelCNN-MGR/models/logmel-cnn-v1_1-YYYYMMDD-HHMMSS \
	audio_demo/Blues-Chris\ Stapleton-Tennessee\ Whiskey.mp3
```

Use this with a run directory that is compatible with the current v1.1 inference feature shape.

### 7. Run the inference web service

```bash
python MelCNN-MGR/inference_web_service/app.py \
	--run-dir MelCNN-MGR/models/logmel-cnn-v1_1-YYYYMMDD-HHMMSS \
	--host 127.0.0.1 \
	--port 8000
```

This service currently wraps `inference_logmel_cnn_v1_1.py`, so the same compatibility note applies.

### 8. Run the streaming demo app

> [!IMPORTANT]
> This component **must be run on Windows**. It uses `pyaudiowpatch` to capture system audio via WASAPI Loopback, which is not supported on Linux/macOS.

```bash
# In a Windows CMD or PowerShell terminal:
set MELCNN_INFERENCE_WS_URL=ws://127.0.0.1:8000/ws/stream
set MELCNN_INFERENCE_SEND_INTERVAL_SEC=3
python MelCNN-MGR/demo-app/web_audio_capture_v1.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Practical Run Order

For the current production-like training flow, the practical order is:

1. optional external-source preparation
2. manifest build via `1_build_all_datasets_and_samples_v1_1.py`
3. log-mel build via `2_build_log_mel_dataset.py`
4. EDA in `2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb`
5. training in `logmel_cnn_v2_1.py` as the primary training script, or `logmel_cnn_v2_1_exp.py` for controlled experiments
6. if you need the currently documented web demo/service path, use a compatible `v1.1`-style inference run with `inference_logmel_cnn_v1_1.py` / `inference_web_service/app.py`
7. streaming demo through `demo-app/web_audio_capture_v1.py`

If you want to continue experimentation from an existing trained checkpoint while still producing a new run lineage, use `MelCNN-MGR/model_training/logmel_cnn_v2_1.py` or `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py` with `--pretrained-model ...`.

### Explicit One-Run-Per-Source Commands for Stage 1

#### Stage 1a: FMA only
```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
    --mode stage1a \
    --stage1a-sources fma
```

#### Stage 1a: Additional datasets only
```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
    --mode stage1a \
    --stage1a-sources additional
```

#### Stage 1b: FMA only
```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
    --mode stage1b \
    --stage1b-sources fma
```

#### Stage 1b: Additional datasets only
```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
    --mode stage1b \
    --stage1b-sources additional
```
