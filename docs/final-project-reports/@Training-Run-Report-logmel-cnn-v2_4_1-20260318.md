# Training Run Report — Log-Mel CNN v2.4.1 (TFRecord)

**Run ID:** `logmel-cnn-v2_4-cuda-tf-20260318-104959`  
**Script:** `MelCNN-MGR/model_training/logmel_cnn_v2_4_1_cuda_tf.py`  
**Generated:** 2026-03-18 16:38:03  
**Run directory:** `MelCNN-MGR/demo-models/logmel-cnn-v2_4-cuda-tf-20260318-104959_15s/`  

---

## 1. Executive Summary

This run trained the **Log-Mel CNN v2.4.1** architecture from scratch on a 10-genre music/speech classification task using TFRecord-serialised log-mel spectrograms. Training ran for **85 of a maximum 136 epochs** before standard early stopping triggered (patience = 9, min_delta = 0.002). The model achieved:

| Split | Accuracy | Macro-F1 | Loss |
|-------|----------|----------|------|
| **Train** | 95.70 % | 0.9572 | 0.4369 |
| **Validation** | 86.70 % | 0.8667 | 0.6559 |
| **Test** | **87.07 %** | **0.8695** | 0.6597 |

The train–test macro-F1 gap is **8.77 percentage points**, down from ~17 points in earlier v2.x iterations, reflecting the cumulative effect of graduated spatial dropout, Mixup, and SpecAugment regularisation.

---

## 2. Environment

| Property | Value |
|----------|-------|
| Platform | Linux WSL2 (kernel 6.6.87.2-microsoft-standard) |
| Python / TF | TensorFlow 2.20.0 |
| Hardware | NVIDIA RTX 3090 (CUDA, `/GPU:0`) |
| Random seed | 36 |
| Float precision | float32 |
| JIT compile | Disabled (`jit_compile=False`) |
| TF layout optimiser | Disabled (startup flag) |

---

## 3. Dataset

| Property | Value |
|----------|-------|
| Format | TFRecord (uncompressed, shard-based) |
| Clip length | 15.0 s |
| Sample rate | 22 050 Hz |
| Log-mel shape | 192 mel bins × 1 291 time frames |
| Classes | 10 |
| Genres (sorted) | Blues, Bolero, Classical, Country, Hip-Hop, Jazz, Metal, Pop, Rock, Speech |

### Split sizes

| Split | Samples | Shards |
|-------|---------|--------|
| Train | 15 410 | 16 |
| Validation | 3 309 | 4 |
| Test | 3 309 | 4 |
| **Total** | **22 028** | **24** |

### Per-genre train distribution

| Genre | Train samples | Class weight |
|-------|--------------|-------------|
| Blues | 1 569 | 0.9822 |
| Bolero | 1 566 | 0.9840 |
| Classical | 1 554 | 0.9916 |
| Country | 1 519 | 1.0145 |
| Hip-Hop | 1 567 | 0.9834 |
| Jazz | 1 571 | 0.9809 |
| Metal | 1 367 | 1.1273 |
| Pop | 1 563 | 0.9859 |
| Rock | 1 570 | 0.9815 |
| Speech | 1 564 | 0.9853 |

Class balance ratio (max/min): **1.1492**. Weights capped at 1.5; Metal (the minority class) received the highest weight (1.1273). Class weighting was applied to the pre-Mixup anchor labels.

---

## 4. Feature Extraction

| Parameter | Value |
|-----------|-------|
| Sample rate | 22 050 Hz |
| Mel bins (n_mels) | 192 |
| FFT size (n_fft) | 512 |
| Hop length | 256 samples (~11.6 ms) |
| Clip duration | 15.0 s |
| Log-mel shape | (192, 1 291) |

### Normalisation

**Type:** train-only per-mel-bin standardisation (two-pass streaming over train split).  
Each mel bin $k$ is standardised to zero mean and unit variance:

$$\hat{x}_{k,t} = \frac{x_{k,t} - \mu_k}{\sigma_k + \varepsilon}, \quad \varepsilon = 10^{-6}$$

| Stat | Min | Max |
|------|-----|-----|
| μ (per-bin mean) | 0.000000 | 1.827329 |
| σ (per-bin std) | 0.000001 | 1.657641 |

Statistics were computed only from the training split and applied identically to validation and test splits, preventing information leakage.

---

## 5. Model Architecture

**Model name:** `logmel_cnn_v2_4_1_cuda_tf`  
**Total parameters:** 1 049 002 (≈ 4.00 MB)  
**Architecture family:** 5-block 2-D CNN with global average pooling and bottleneck dense head.

| Layer | Output shape | Parameters |
|-------|-------------|-----------|
| `logmel` (InputLayer) | (None, 192, 1 291, 1) | 0 |
| `conv1` Conv2D (5×5, 32) | (None, 192, 1 291, 32) | 800 |
| `bn1` BatchNorm + `relu1` ReLU | (None, 192, 1 291, 32) | 128 |
| `pool1` MaxPool2D (2×2) | (None, 96, 645, 32) | 0 |
| `conv2` Conv2D (3×3, 64) | (None, 96, 645, 64) | 18 432 |
| `bn2` BatchNorm + `relu2` ReLU | (None, 96, 645, 64) | 256 |
| `sdrop2` SpatialDropout2D (0.05) | (None, 96, 645, 64) | 0 |
| `pool2` MaxPool2D (2×2) | (None, 48, 322, 64) | 0 |
| `conv3` Conv2D (3×3, 128) | (None, 48, 322, 128) | 73 728 |
| `bn3` BatchNorm + `relu3` ReLU | (None, 48, 322, 128) | 512 |
| `sdrop3` SpatialDropout2D (0.10) | (None, 48, 322, 128) | 0 |
| `pool3` MaxPool2D (2×2) | (None, 24, 161, 128) | 0 |
| `conv4` Conv2D (3×3, 256) | (None, 24, 161, 256) | 294 912 |
| `bn4` BatchNorm + `relu4` ReLU | (None, 24, 161, 256) | 1 024 |
| `sdrop4` SpatialDropout2D (0.15) | (None, 24, 161, 256) | 0 |
| `pool4` MaxPool2D (2×2) | (None, 12, 80, 256) | 0 |
| `conv5` Conv2D (3×3, 256) | (None, 12, 80, 256) | 589 824 |
| `bn5` BatchNorm + `relu5` ReLU | (None, 12, 80, 256) | 1 024 |
| `sdrop5` SpatialDropout2D (0.20) | (None, 12, 80, 256) | 0 |
| `pool5` MaxPool2D (2×2) | (None, 6, 40, 256) | 0 |
| `gap` GlobalAveragePooling2D | (None, 256) | 0 |
| `fc_bottleneck` Dense (256, ReLU) | (None, 256) | 65 792 |
| `dropout` Dropout (0.20) | (None, 256) | 0 |
| `fc_out` Dense (10, Softmax) | (None, 10) | 2 570 |

**Trainable:** 1 047 530 params · **Non-trainable (BN):** 1 472 params

Key design choices:
- **Graduated spatial dropout** (0.05 → 0.10 → 0.15 → 0.20 across blocks 2–5) applies progressively stronger channel regularisation at deeper, semantically richer feature maps.
- **256-unit bottleneck** (doubled from v2.1's 128 units) preserves the full backbone feature space before the classification head.
- **Global average pooling** eliminates spatial flattening, reducing overfitting to temporal position.

---

## 6. Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Initialisation | Scratch (no warm-start) |
| Max epochs | 136 |
| Actual epochs completed | **85** |
| Batch size | 48 |
| Optimizer | AdamW |
| Learning rate schedule | Cosine annealing with 5-epoch linear warm-up |
| LR max | 0.0005 |
| LR at epoch 85 | ≈ 0.000166 |
| Weight decay | 0.0001 |
| Gradient clipping | `clipnorm=1.0` |
| Label smoothing | 0.05 |

### Data augmentation

| Technique | Configuration |
|-----------|--------------|
| Mixup | α = 0.3 (applied during training; class weights anchored on pre-Mixup labels) |
| SpecAugment | 1 frequency mask (max 24 mel bins), 1 time mask (max 40 frames) |
| SpatialDropout2D | Graduated 0.05 / 0.10 / 0.15 / 0.20 (blocks 2–5) |
| Final Dropout | 0.20 (applied after FC bottleneck) |

### Early stopping

**Standard early stopping** (active from epoch 66):  
- Monitor: `val_macro_f1`  
- Patience: 9 epochs  
- Min delta: 0.002  
- Restore best weights: enabled  

**Gap-aware early stopping** (configured but disabled for this run):  
- Would activate when train–val F1 gap > 0.10 for ≥ 9 epochs with val not improving, after epoch 60.

### tf.data pipeline

| Setting | Value |
|---------|-------|
| Parallelism mode | Fixed |
| num_parallel_calls | 6 |
| num_parallel_reads | 6 |
| Prefetch | AUTOTUNE |

---

## 7. Training Progression

Training was performed in a single stage (`full_finetune`; all backbone layers trainable from epoch 0). Key milestones are summarised below.

### Selected epoch checkpoints

| Epoch | Val loss | Val acc | Val macro-F1 | Note |
|-------|----------|---------|-------------|------|
| 1 | 2.1956 | 0.230 | 0.1333 | First checkpoint saved |
| 2 | 1.5544 | 0.477 | 0.4091 | |
| 3 | 1.4239 | 0.543 | 0.5004 | |
| 8 | 1.1834 | 0.658 | 0.6585 | |
| 13 | 1.0118 | 0.724 | 0.7245 | |
| 21 | 0.9023 | 0.776 | 0.7724 | |
| 27 | 0.8600 | 0.787 | 0.7833 | |
| 35 | 0.8062 | 0.803 | 0.8053 | Crossed 80 % val F1 |
| 39 | 0.7712 | 0.821 | 0.8209 | |
| 44 | 0.7619 | 0.824 | 0.8245 | |
| 56 | 0.7099 | 0.847 | 0.8471 | |
| 65 | 0.6909 | 0.856 | 0.8561 | |
| 74 | 0.6740 | 0.860 | 0.8601 | |
| 75 | 0.6692 | 0.862 | 0.8619 | |
| 76 | 0.6651 | 0.865 | **0.8653** | Last epoch satisfying min_delta ≥ 0.002 |
| 77 | 0.6637 | 0.865 | 0.8663 | Δ < 0.002 — early-stopping counter starts |
| 83 | 0.6559 | 0.867 | **0.8667** | Best raw checkpoint (ModelCheckpoint) |
| 85 | 0.6688 | 0.864 | 0.8643 | Final completed epoch |

**Early stopping outcome:** After epoch 85 (9 epochs without ≥ 0.002 improvement from epoch 76), early stopping triggered. Weights were **restored from epoch 76** (the last epoch meeting min_delta). The `best_model_macro_f1.keras` artefact reflects epoch 83 (highest raw val_macro_f1 = 0.8667).

### Learning rate curve

The cosine schedule warmed up linearly from ≈ 1.0 × 10⁻⁴ (epoch 1) to 5.0 × 10⁻⁴ (epoch 5), then decayed following a half-cosine to ≈ 1.66 × 10⁻⁴ at epoch 85. No discrete LR drops were applied.

### Training time

| Metric | Value |
|--------|-------|
| Total epochs | 85 |
| Total training time | ≈ 20 273 s (5.63 h) |
| Average per epoch | ≈ 238.5 s |
| Slowest epoch | Epoch 72 (499.7 s) |
| Wall-clock start | 2026-03-18 10:49:59 |
| Wall-clock end | 2026-03-18 16:38:03 |

---

## 8. Evaluation Results

All evaluation metrics are reported on the restored model (epoch 76 weights via EarlyStopping).

### Summary

| Split | Loss | Accuracy | Macro-F1 |
|-------|------|----------|----------|
| Train | 0.4369 | 95.70 % | 0.9572 |
| Validation | 0.6559 | 86.70 % | 0.8667 |
| **Test** | **0.6597** | **87.07 %** | **0.8695** |

**Train–Test macro-F1 gap:** 0.9572 − 0.8695 = **0.0877** (8.77 pp)

### Per-genre test results

| Genre | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Blues | 0.7730 | 0.8030 | 0.7877 | 335 |
| Bolero | 0.9939 | 0.9643 | **0.9789** | 336 |
| Classical | 0.8780 | 0.9851 | 0.9285 | 336 |
| Country | 0.8338 | 0.8262 | 0.8300 | 328 |
| Hip-Hop | 0.8524 | 0.9134 | 0.8818 | 335 |
| Jazz | 0.8514 | 0.8209 | 0.8359 | 335 |
| Metal | 0.9450 | 0.9386 | 0.9418 | 293 |
| Pop | 0.8258 | 0.6586 | **0.7328** | 331 |
| Rock | 0.7878 | 0.8018 | 0.7947 | 338 |
| Speech | 0.9688 | 0.9971 | **0.9827** | 342 |
| **Macro avg** | **0.8810** | **0.8809** | **0.8695** | 3 309 |

### Per-genre validation results

| Genre | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Blues | 0.7577 | 0.7977 | 0.7771 | 341 |
| Bolero | 0.9909 | 0.9733 | **0.9820** | 337 |
| Classical | 0.8533 | 0.9440 | 0.8964 | 339 |
| Country | 0.7840 | 0.7913 | 0.7876 | 321 |
| Hip-Hop | 0.8702 | 0.9320 | 0.9000 | 338 |
| Jazz | 0.8650 | 0.8006 | 0.8315 | 336 |
| Metal | 0.9399 | 0.9172 | 0.9284 | 290 |
| Pop | 0.8467 | 0.6925 | **0.7619** | 335 |
| Rock | 0.7898 | 0.8299 | 0.8093 | 335 |
| Speech | 0.9911 | 0.9941 | **0.9926** | 337 |

### Observations

**Strongest genres (F1 > 0.93 on test):**
- **Bolero** (0.979) and **Speech** (0.983) stand out as by far the most distinctive classes, with near-perfect precision/recall. Their spectro-temporal signatures are highly separable from music genres.
- **Classical** (0.929) and **Metal** (0.942) also exceeded 0.93, benefiting from their characteristic harmonic/timbral density patterns.

**Weakest genres (F1 < 0.80 on test):**
- **Pop** (0.733) is the single hardest class. Its low recall (0.659) indicates frequent misclassification; Pop shares rhythmic and spectral characteristics with Hip-Hop, Rock, and Country.
- **Blues** (0.788) and **Rock** (0.795) cluster in the lower range, consistent with their shared instrumentation and overlapping timbral features.

**Structurally confused pairs** (implied by precision/recall asymmetry):
- Pop: high precision (0.826) but low recall (0.659) — many true-Pop samples are predicted as other genres, but Pop predictions themselves are mostly correct.
- Classical: high recall (0.985) but lower precision (0.878) — the model aggressively predicts Classical, pulling in some non-Classical samples.

---

## 9. Changes from Previous Versions

The run report records the following architectural and training deltas relative to predecessor versions:

### vs. v2.1

| Change | v2.1 | v2.4.1 | Rationale |
|--------|------|--------|-----------|
| Weight decay | 5×10⁻⁴ | **1×10⁻⁴** | Lighter L2 with Mixup + dropout already active |
| SpecAugment masks | 2 | **1** | Avoid over-masking combined with Mixup |
| Spatial dropout | flat 0.10 | **graduated 0.05/0.10/0.15/0.20** | Deeper layers get more regularisation |
| Bottleneck units | 128 | **256** | Preserve full backbone feature space |
| Final dropout | 0.25 | **0.20** | Wider bottleneck absorbs some capacity |
| Gradient clipping | — | **clipnorm=1.0** | Added to AdamW |
| Class-weight strategy | post-Mixup blended | **pre-Mixup anchor** | Avoid diluting minority-class weight signal |
| Gap-aware early stopping | — | **Added** (threshold=0.13, patience=6) | Prevent runaway overfitting |
| Restore best weights | disabled | **enabled** | Final model snaps back to best val epoch |

### vs. v2.2

| Change | Detail |
|--------|--------|
| Normalisation | Train-only per-mel-bin standardisation is now explicit two-pass, saved as `mean_per_bin`/`std_per_bin` artefacts |

### vs. v2.3

| Change | Detail |
|--------|--------|
| Input pipeline | Training now consumes shard-based TFRecords from `3_convert_npy_2_tfrecord.py` instead of per-sample `.npy` loads |

---

## 10. Artefacts

All artefacts are stored in `MelCNN-MGR/demo-models/logmel-cnn-v2_4-cuda-tf-20260318-104959_15s/`.

| File | Description |
|------|-------------|
| `best_model_macro_f1.keras` | ModelCheckpoint — highest raw val_macro_f1 (epoch 83, 0.8667) |
| `logmel_cnn_v2_4_1_cuda_tf.keras` | Final saved model (EarlyStopping-restored weights from epoch 76) |
| `norm_stats.npz` | Per-mel-bin mean/std arrays (shape 192) for inference-time normalisation |
| `normalization_stats.json` | Normalisation metadata (type, n_mels, epsilon) |
| `run_report_logmel_cnn_v2_4_1_cuda_tf.json` | Full structured run report (config, per-epoch LR/time, per-genre eval) |
| `console_output.txt` | Raw training console log (850 lines) |

**Inference note:** To reproduce the test evaluation at inference time, apply the same per-mel-bin standardisation using `norm_stats.npz` before feeding spectrograms to the model. The active production model is referenced in `MelCNN-MGR/settings.json` under `model_inference_settings.model_name`.

---

## 11. References

| # | Resource |
|---|----------|
| 1 | `MelCNN-MGR/model_training/logmel_cnn_v2_4_1_cuda_tf.py` — training script |
| 2 | `MelCNN-MGR/model_training/3_convert_npy_2_tfrecord.py` — TFRecord serialisation |
| 3 | `MelCNN-MGR/model_training/2_build_log_mel_dataset.py` — log-mel feature extraction |
| 4 | `MelCNN-MGR/model_training/1_build_all_datasets_and_samples_v1_1.py` — dataset assembly |
| 5 | `MelCNN-MGR/settings.json` — project-wide settings (genres, sample length) |
| 6 | `docs/Mathematical-Representations-LogMel-CNN-v2_1.md` — formal maths for log-mel and normalisation |
| 7 | `docs/Development Guidelines Gap-Aware Early Stopping.md` — gap-aware stopping design |
| 8 | `dev-logs/2026-03-18-logmel-cnn-v2_4_1-cuda-tf-validation.md` — v2.4.1 validation log |
| 9 | `dev-logs/2026-03-17-tfrecord-conversion-and-logmel-cnn-v2_4-cuda-tf.md` — TFRecord pipeline log |
| 10 | `docs/final-project-reports/Project-Report.md` — parent project report |
