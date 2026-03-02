# Project Proposal: MFCC+CNN vs Log-Mel+CNN on FMA-Medium (Controlled + Optimized)

Mon Mar  2 09:00:00 UTC 2026

Project name: MelCNN MGR (MGR = Music Genre Recognition)

## 1. Project Overview

This project uses the **FMA-medium** dataset (25,000 tracks, 30 seconds each, 16 unbalanced genres) to study a common claim in music genre classification:

> **Log-mel spectrograms + CNNs typically outperform MFCC + CNNs** because log-mel preserves richer time–frequency structure, while MFCC compresses and discards detail before the CNN sees it.

The project is designed in **two goals**:

1. **Goal 1 (Controlled Study):** Compare **MFCC vs log-mel** using the **same CNN architecture** and the **same training setup**, changing only the input representation.
2. **Goal 2 (Best-Result Engineering):** Starting from the **Goal-1 log-mel baseline**, improve/optimize log-mel performance, allowing architecture and training upgrades to achieve the best results possible under a defined experimentation budget.

A key by-product of Goal 1 is a clean, reproducible **log-mel+CNN baseline** that becomes the official starting point for Goal 2.

---

## 2. Motivation and Background

Both approaches perform genre classification by converting audio into a 2D time–frequency matrix and training a neural network to predict a genre distribution.

### (A) MFCC → 2D CNN

* MFCC is derived from log-mel by applying DCT and truncating coefficients.
* This emphasizes coarse spectral envelope / timbre but discards fine detail (harmonics, textures).
* Pros: smaller inputs, faster training, sometimes more stable on small data.
* Cons: lower “information ceiling” because detail is removed before learning.

### (B) Log-mel → 2D CNN

* Log-mel retains more detailed time–frequency structure.
* Pros: often higher accuracy and macro-F1 on mid/large datasets.
* Cons: more compute, more sensitive to normalization/augmentation choices.

**Working hypothesis:** CNNs benefit from local patterns in time–frequency space; log-mel preserves those patterns better than MFCC.

---

## 3. Objectives

### Goal 1 — Controlled Representation Comparison (Same Architecture)

* Reproduce an MFCC+CNN baseline (aligned with the dataset’s intended split).
* Train the *same CNN* on log-mel inputs.
* Quantify the performance gap attributable to representation only.

### Goal 2 — Optimize Log-mel System (Best Achievable Results)

* Use the Goal-1 log-mel+CNN baseline as the official starting baseline.
* Improve performance by applying training upgrades and/or changing architecture.
* Track gains using incremental ablations (each change measured against baseline).

---

## 4. Dataset

**FMA-medium**

* ~25k 30-second tracks
* 16 genres (unbalanced)
* Official metadata includes predefined **train / validation / test splits**

**Core rule:** Use the **official split** for both goals to avoid leakage and ensure comparability.

---

## 5. Methodology

## 5.1 Goal 1: Controlled Experiment Design

### Fixed across both MFCC and log-mel

* Same dataset split (official FMA split)
* Same audio decoding + resampling + duration handling
* Same CNN architecture (shared code path)
* Same training recipe: optimizer, LR schedule, batch size, epochs, early stopping, regularization
* Same augmentation policy (preferably off initially)
* Same normalization approach (train-only statistics)

### Variable (the only change)

* Input feature representation:

  * MFCC matrix
  * Log-mel spectrogram matrix

### Feature extraction (example defaults)

* Sample rate: consistent (e.g., 22050)
* Mono conversion (consistent)
* STFT parameters: consistent (e.g., n_fft=2048, hop_length=512)
* Log-mel: n_mels (e.g., 128), log compression
* MFCC: computed from the same mel configuration, keep n_mfcc (e.g., 20–40)

### Architecture requirement for “same CNN”

Use a CNN that ends with **global average pooling** (or adaptive pooling) so the network can accept different frequency dimensions (MFCC vs log-mel) without changing layers.

### Evaluation metrics

Because classes are unbalanced:

* Accuracy
* Macro-F1 (primary)
* Weighted-F1
* Confusion matrix
* Per-genre precision/recall/F1

Stability:

* Run 3–5 random seeds and report mean ± std.

### Goal-1 Outputs (Baselines)

* **B0 (MFCC-Controlled):** MFCC + CNN_fixed + TrainRecipe_fixed
* **B1 (LogMel-Controlled):** LogMel + CNN_fixed + TrainRecipe_fixed  *(this is the baseline carried into Goal 2)*

---

## 5.2 Goal 2: Optimizing Log-mel + CNN (Architecture Allowed to Change)

Goal 2 starts from **B1** and aims to improve performance systematically. This is treated as an engineering phase with disciplined experimentation.

### Optimization strategy: “upgrade ladder”

Each step introduces one major change, evaluated against B1:

* **B1.1**: B1 + SpecAugment (time/frequency masking)
* **B1.2**: B1.1 + improved LR scheduling (cosine decay / OneCycle)
* **B1.3**: B1.2 + stronger backbone (ResNet-style CNN)
* **B1.4**: B1.3 + mixup / label smoothing (as appropriate)
* **B1.5 (optional)**: multi-crop training or test-time averaging

### Optional model families (if time allows)

* ResNet-like 2D CNN
* CRNN (CNN + recurrent/temporal module)
* Attention pooling over time

### Budget constraint (to keep the project finite)

Define an explicit budget such as:

* maximum number of runs (e.g., 30–50 total experiments), or
* maximum compute hours, or
* a fixed hyperparameter search budget (e.g., 25 Optuna trials)

### Reporting

For each improvement, report:

* ΔMacro-F1 and ΔAccuracy relative to B1
* compute cost / training time
* whether it is kept as part of the final system

---

## 6. Deliverables

### Code and Reproducibility

* Fully reproducible training pipeline with configs for:

  * Goal 1 MFCC baseline (B0)
  * Goal 1 log-mel baseline (B1)
  * Goal 2 optimized variants (B1.1, B1.2, …)
* Feature caching system (avoid recomputing spectrograms)
* Experiment logging (metrics to CSV/JSONL + saved configs)

### Reports

1. **Goal 1 Report (Controlled Comparison)**

   * Baseline definitions (B0, B1)
   * Metrics tables (mean ± std)
   * Confusion matrices and per-genre results
   * Interpretation of differences

2. **Goal 2 Report (Optimization Results)**

   * Upgrade ladder with deltas
   * Best final model performance
   * Ablation summary: what helped, what didn’t

3. **Final Summary (Executive)**

   * “How much better is log-mel than MFCC under a controlled setup?”
   * “How far can optimized log-mel go beyond the baseline?”

---

## 7. Risks and Mitigations

| Risk                       | Impact                | Mitigation                                      |
| -------------------------- | --------------------- | ----------------------------------------------- |
| Leakage via normalization  | Inflated test results | Train-only stats; strict split handling         |
| Unfair comparison (Goal 1) | Invalid conclusions   | Same CNN + same recipe; only features change    |
| Genre imbalance            | Misleading accuracy   | Macro-F1 primary; per-class breakdown           |
| Non-determinism            | Unstable conclusions  | Multi-seed runs; mean ± std                     |
| Compute cost               | Slow iteration        | Feature caching; batch extraction; fixed budget |

---

## 8. Success Criteria

### Goal 1 success

* B0 and B1 are reproducible and comparable
* Clear measurement of representation effect (log-mel vs MFCC) with macro-F1 emphasis

### Goal 2 success

* Demonstrated improvements over B1 with documented deltas
* A best-performing log-mel-based model with a clear “what worked” story

---

## 9. Proposed Repository Structure

```
fma-genre-classification/
  README.md
  configs/
    g1_b0_mfcc_cnnfixed.yaml
    g1_b1_logmel_cnnfixed.yaml
    g2_b1p1_logmel_specaug.yaml
    g2_b1p2_logmel_specaug_cosine.yaml
    g2_b1p3_logmel_resnet.yaml
  data/
    raw/                 # FMA zips or external paths
    features/
      mfcc/
      logmel/
  src/
    data/
      fma_metadata.py
      dataset.py
    features/
      mfcc.py
      logmel.py
    models/
      cnn_fixed.py
      resnet_audio.py
      crnn.py
    train.py
    eval.py
    utils/
      metrics.py
      logging.py
      seed.py
  reports/
    goal1_controlled_comparison.md
    goal2_optimized_logmel.md
    figures/
    results.csv
```

---

## 10. Conclusion

This project is intentionally structured to produce conclusions that are both **scientifically credible** and **practically useful**:

* **Goal 1** isolates the effect of representation (MFCC vs log-mel) using the same CNN architecture.
* **Goal 2** treats the Goal-1 log-mel baseline as a launchpad for optimization, allowing architecture changes while keeping evaluation honest and traceable.

The final output will answer:

1. *How much does log-mel help compared to MFCC under strict controls?*
2. *How good can log-mel+deep learning get on FMA-medium with reasonable optimization effort?*
