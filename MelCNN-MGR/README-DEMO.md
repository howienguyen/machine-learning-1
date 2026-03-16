# MelCNN-MGR Demo and Production-Like Pipeline

This document summarizes the current production-like training path, the main preprocessing entry points, the active training entry point, the current inference/application boundary, and the main commands needed to run the demo flow.

## Current Training Path

The current production-like training path is:

```text
utils/download_by_genre_limits.py --> extract_mtg_processed_samples.py

data sources: FMA & additional_datasets -->

MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py -->
MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py -->
MelCNN-MGR/model_training/2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb -->
MelCNN-MGR/model_training/logmel_cnn_v2_2.py
```

Interpretation:

1. optional external-source preparation happens first through genre-limited download and MTG extraction
2. the training corpus is then assembled from FMA plus `additional_datasets`
3. `1_build_all_datasets_and_samples_v1_1.py` is the main manifest-and-sample preprocessing script
4. `2_build_log_mel_dataset.py` is the main log-mel preprocessing script
5. the manifest/log-mel EDA notebook is used to inspect readiness before training
6. `logmel_cnn_v2_2.py` is the main current training entry point
7. `logmel_cnn_v2_1.py` and `logmel_cnn_v2_1_exp.py` remain useful older v2-family references

## Main Preprocessing Scripts

The main preprocessing scripts are:

1. `MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py`
2. `MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py`

Current intent:

1. `1_build_all_datasets_and_samples_v1_1.py` builds the dataset manifests, sample manifests, and final selected sample manifest
2. `2_build_log_mel_dataset.py` converts the final sample manifest into split-grouped log-mel `.npy` features plus parquet indexes for training

## Active Training Scripts

The main current training scripts are:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`
3. `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py`
4. `MelCNN-MGR/model_training/logmel_cnn_v1.py` and `logmel_cnn_v1_1.py` as legacy baselines

Current intent:

1. `logmel_cnn_v2_2.py` is the primary current training script
2. `logmel_cnn_v2_1.py` and `logmel_cnn_v2_1_exp.py` remain useful comparison points inside the v2 family
3. `v1` / `v1_1` remain useful as baseline references and for the currently documented inference service path

## Run Artifacts

Each v2-family training run creates a dedicated run directory under `MelCNN-MGR/models/`.

`logmel_cnn_v2_2.py` follows the same pattern with `v2_2`-named artifacts, while
`logmel_cnn_v2_1.py` and `logmel_cnn_v2_1_exp.py` keep their corresponding
`v2_1` / `v2_1_exp` names.

Typical `v2_2` contents:

```text
logmel-cnn-v2_2-YYYYMMDD-HHMMSS/
    best_model_macro_f1.keras
	logmel_cnn_v2_2.keras
    norm_stats.npz
    console_output.txt
	run_report_logmel_cnn_v2_2.json
```

For `logmel_cnn_v2_1_exp.py`, the filenames are the corresponding `v2_1_exp` variants.

Notes:

1. `console_output.txt` captures the console stream for the run after the run directory is created
2. `run_report_*.json` stores structured metadata, metrics, and artifact paths
3. the JSON report includes the path to `console_output.txt`

## Inference and Applications

The current inference and demo/application entry points are:

1. `MelCNN-MGR/model_inference/inference_logmel_cnn_v2_x.py`
2. `MelCNN-MGR/Lab/inference_logmel_cnn_v1_1.py` as the legacy direct-inference script
3. `MelCNN-MGR/inference_web_service/app.py`
4. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`

Important boundary:

1. the current web service now wraps the config-driven `v2.1` family inference module
2. the old `v1.1` direct inference script has been relocated to `MelCNN-MGR/Lab/inference_logmel_cnn_v1_1.py`
3. the current `v2.1` training pipeline follows the log-mel dataset configuration, which currently defaults to `sample_length_sec = 15` from `MelCNN-MGR/settings.json`
4. the current service path should be used with a compatible `v2` / `v2.1` / `v2.1-exp` model directory
5. use the relocated legacy `v1.1` script only when you intentionally want the historical fixed-shape v1.1 inference path

Practical meaning:

1. use `v2.1` / `v2.1-exp` for current training experiments
2. use the current inference service path with a compatible v2-family model directory
3. use the relocated `v1.1` script in `MelCNN-MGR/Lab/` only for legacy compatibility checks

## Command Lines

### 1. Optional external-source preparation

Download genre-limited MTG/Jamendo audio:

```bash
python utils/download_by_genre_limits.py
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
python MelCNN-MGR/model_training/logmel_cnn_v2_2.py
```

The full production-like/demo workflow is documented once in `Main Run Profile` below.

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

Current direct inference path is the config-driven v2.x family inference module:

```bash
python MelCNN-MGR/model_inference/inference_logmel_cnn_v2_x.py \
	--model-dir MelCNN-MGR/models/<logmel-v2-family-model-dir> \
	audio_demo/Blues-Chris\ Stapleton-Tennessee\ Whiskey.mp3
```

Companion example script for the current `logmel_cnn_v2_2.py` workflow:

```bash
python MelCNN-MGR/Lab/examples/inference_logmel_cnn_v2_x_example.py
```

That example now auto-discovers the newest `MelCNN-MGR/models/logmel-cnn-v2_2-*` directory and runs the renamed `inference_logmel_cnn_v2_x.py` module against files in `audio_demo/`.

If you need the old fixed-shape v1.1 path, use `MelCNN-MGR/Lab/inference_logmel_cnn_v1_1.py` explicitly.

### 7. Run the inference web service

```bash
python MelCNN-MGR/inference_web_service/app.py \
	--model-dir MelCNN-MGR/models/<logmel-cnn-v2_2-YYYYMMDD-HHMMSS-or-other-compatible-model-dir> \
	--host 127.0.0.1 \
	--port 8000
```

This service currently wraps `model_inference/inference_logmel_cnn_v2_x.py`, so use a compatible v2-family model directory.

### 8. Run the streaming demo app

> [!IMPORTANT]
> This component **must be run on Windows**. It uses `pyaudiowpatch` to capture system audio via WASAPI Loopback, which is not supported on Linux/macOS.

```bash
# In a Windows CMD or PowerShell terminal:
.venvw/Scripts/activate
python MelCNN-MGR/demo-app/web_audio_capture_v1.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Main Run Profile

The current main production-like/demo profile is:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1 \
	--stage1a-sources fma \
	--stage1b-sources fma

python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage1 \
	--stage1a-sources additional \
	--stage1b-sources additional

python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
	--mode stage2

python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py

python MelCNN-MGR/model_training/logmel_cnn_v2_2.py

python MelCNN-MGR/inference_web_service/app.py \
	--model-dir MelCNN-MGR/models/<choose-a-compatible-trained-model-dir>

.venvw/Scripts/activate
python MelCNN-MGR/demo-app/web_audio_capture_v1.py
```

Practical interpretation:

1. run Stage 1 once for FMA only
2. run Stage 1 once for `additional` only
3. run Stage 2 after both source-specific Stage 1 outputs exist
4. build the fixed log-mel dataset cache and parquet indexes
5. train the current primary model with `logmel_cnn_v2_2.py`
6. start the inference web service with the exact model directory you want to serve
7. on Windows, activate `.venvw` and launch the streaming demo app

## Practical Run Order

For the current production-like training flow, the practical order is:

1. optional external-source preparation
2. manifest build via `1_build_all_datasets_and_samples_v1_1.py`
3. log-mel build via `2_build_log_mel_dataset.py`
4. EDA in `2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb`
5. training in `logmel_cnn_v2_2.py` as the primary current training script, with `logmel_cnn_v2_1.py` / `logmel_cnn_v2_1_exp.py` as comparison or warm-start references
6. if you need the currently documented web demo/service path, use a compatible v2-family inference model directory with `model_inference/inference_logmel_cnn_v2_x.py` / `inference_web_service/app.py`
7. streaming demo through `demo-app/web_audio_capture_v1.py`

If you want to continue experimentation from an existing trained checkpoint while still producing a new run lineage, use `MelCNN-MGR/model_training/logmel_cnn_v2_1.py` or `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py` with `--pretrained-model ...`.

## Baselines

Useful baseline and legacy references:

1. `MelCNN-MGR/Lab/build_manifest.py`
2. `MelCNN-MGR/Lab/build_tiny_dataset.py`
3. `MelCNN-MGR/model_training/baseline_mfcc_cnn_v5.ipynb`
4. `MelCNN-MGR/model_training/baseline_logmel_cnn_v1.ipynb`

Practical meaning:

1. `build_manifest.py` and `build_tiny_dataset.py` are useful older Lab utilities for baseline dataset preparation and smoke-test workflows
2. `baseline_mfcc_cnn_v5.ipynb` is the MFCC baseline notebook reference
3. `baseline_logmel_cnn_v1.ipynb` is the earlier log-mel baseline notebook reference
