# MelCNN-MGR Demo and Production-Like Pipeline

This document summarizes the current production-like training path, the inference/application entry points, and the main commands needed to run the demo flow.

## Production-Like Training Pipeline

The intended production-like path is:

```text
download_by_genre_limits.py --> extract_mtg_processed_samples.py

data sources: FMA & additional_datasets -->

MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py -->
MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py -->
MelCNN-MGR/notebooks/2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb -->
MelCNN-MGR/notebooks/logmel_cnn_v1.py
```

Interpretation:

1. optional external-source preparation happens first through genre-limited download and MTG extraction
2. the training corpus is then assembled from FMA plus `additional_datasets`
3. Stage 1 and Stage 2 manifest generation define the production-like sample inventory
4. log-mel cache/index generation materializes the feature layer
5. the manifest/log-mel EDA notebook is used to inspect readiness before training
6. `logmel_cnn_v1.py` is then the training entry point in this documented path

## Inference and Applications

The current inference and demo/application entry points are:

1. `MelCNN-MGR/inference_logmel_cnn_v1_1.py`
2. `MelCNN-MGR/inference_web_service/app.py`
3. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`

Their roles are:

1. `inference_logmel_cnn_v1_1.py` loads the trained v1.1 model and runs direct file or waveform inference
2. `inference_web_service/app.py` exposes the model over HTTP and WebSocket
3. `demo-app/web_audio_capture_v1.py` captures system audio, streams PCM chunks to the inference service, and displays live predictions in a local Flask UI

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

Run the full production-like manifest pipeline:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py --mode both
```

If you need a split run instead:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py --mode stage1
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py --mode stage2
```

### 3. Build the log-mel dataset indexes/cache

```bash
python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py
```

### 4. Review the manifest and downstream readiness

Open and run:

```text
MelCNN-MGR/notebooks/2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb
```

### 5. Run model training

```bash
python MelCNN-MGR/notebooks/logmel_cnn_v1.py
```

Warm-start training from an existing v1.1 or v2 `.keras` checkpoint into a brand new v2 run:

```bash
python MelCNN-MGR/notebooks/logmel_cnn_v2.py \
	--pretrained-model MelCNN-MGR/models/logmel-cnn-v1_1-20260311-025046/best_model_macro_f1.keras
```

Backbone-only warm-start when the genre set changes and you want to replace the classifier head automatically:

```bash
python MelCNN-MGR/notebooks/logmel_cnn_v2.py \
	--pretrained-model MelCNN-MGR/models/logmel-cnn-v1_1-20260311-025046/best_model_macro_f1.keras \
	--pretrained-mode backbone_only
```

Notes:

1. this is a fresh new run, not a resume
2. `logmel_cnn_v2.py` recompiles with a new optimizer and learning-rate schedule
3. `--pretrained-mode full_model` requires matching log-mel shape, class count, and `genre_classes` order
4. `--pretrained-mode backbone_only` transfers compatible backbone weights and rebuilds the final classifier head for the current genre set

### 6. Run direct inference

```bash
python MelCNN-MGR/inference_logmel_cnn_v1_1.py \
	--run-dir MelCNN-MGR/models/logmel-cnn-v1_1-YYYYMMDD-HHMMSS \
	audio_demo/Blues-Chris\ Stapleton-Tennessee\ Whiskey.mp3
```

### 7. Run the inference web service

```bash
python MelCNN-MGR/inference_web_service/app.py \
	--run-dir MelCNN-MGR/models/logmel-cnn-v1_1-YYYYMMDD-HHMMSS \
	--host 127.0.0.1 \
	--port 8000
```

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

For the end-to-end demo and production-like training flow, the practical order is:

1. optional external-source preparation
2. manifest build via `1_build_all_datasets_and_samples.py`
3. log-mel build via `2_build_log_mel_dataset.py`
4. EDA in `2_MelCNN_MGR_Manifest_LogMel_EDA.ipynb`
5. training in `logmel_cnn_v1.py`
6. direct inference or service startup with `inference_logmel_cnn_v1_1.py` / `inference_web_service/app.py`
7. streaming demo through `demo-app/web_audio_capture_v1.py`

If you want to continue experimentation from an existing trained checkpoint while still producing a new run lineage, use `MelCNN-MGR/notebooks/logmel_cnn_v2.py` with `LOGMEL_CNN_PRETRAINED_MODEL=...`.
