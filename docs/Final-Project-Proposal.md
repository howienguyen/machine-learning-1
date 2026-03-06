# Project Proposal: MFCC+CNN vs Log-Mel+CNN on FMA-Medium (Controlled + Optimized)

Mon Mar  2 09:00:00 UTC 2026

Project name: MelCNN MGR (MGR = Music Genre Recognition)

## 1. Project Overview

This project uses the **FMA-medium** dataset (25,000 tracks, 30 seconds each, 16 unbalanced genres) to study a common claim in music genre classification:

> **Log-mel spectrograms + CNNs typically outperform MFCC + CNNs** because log-mel preserves richer time–frequency structure, while MFCC compresses and discards detail before the CNN sees it.

The project is designed in **three goals**:

1. **Goal 1 (Controlled Study):** Compare **MFCC vs log-mel** using the **same CNN architecture** and the **same training setup**, changing only the input representation.
2. **Goal 2 (Best-Result Engineering):** Starting from the **Goal-1 log-mel baseline**, improve/optimize log-mel performance, allowing architecture and training upgrades to achieve the best results possible under a defined experimentation budget.
3. **Goal 3 (Production Robustness):** Address structural limitations in the Goal-1/2 baseline -- fixed-length dependency, class-imbalance bias, and CPU-bound input pipeline -- to move toward a more robust and efficient system.

A key by-product of Goal 1 is a clean, reproducible **log-mel+CNN baseline** that becomes the official starting point for Goals 2 and 3.

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

### (C) Feature Storage: Raw Numerical Arrays vs Image Files

Both MFCC matrices and log-mel spectrograms are fundamentally **2D numerical arrays** — rows are frequency bands, columns are time frames, values are floating-point energy measurements. They are *not* images, even though they can be *visualized* as images.

This distinction determines the caching strategy. This project stores cached features as **NumPy binary files (`.npy`)**, one file per track.

#### Why `.npy` and not image files?

Storing as an image (PNG, JPEG) requires quantizing the float values into pixel integers (0–255 for 8-bit, 0–65535 for 16-bit). This introduces two problems:

1. **Precision loss.** `log1p(mel_power)` values are unbounded floats (e.g., `3.74182`). Mapping to uint8 discards all sub-integer precision. Mapping to uint16 is better but still requires a global or per-track normalization to define the range.
2. **Normalization is entangled with storage.** To fit floats into pixel range, you must choose a normalization scheme *at cache time*. This bakes the normalization into the file — if you later want to change it (e.g., use per-band stats instead of global stats), you must regenerate the entire cache.

With `.npy`, the cache stores the **raw `log1p`-compressed spectrogram** with no normalization applied. The `tf.data` pipeline applies **train-only per-band mean/std normalization at load time**. These two concerns are cleanly separated.

#### Format comparison

| Format | Values | Precision | Notes |
| --- | --- | --- | --- |
| `.npy` (float32) | Raw floats | Full (~7 sig. digits) | Used in this project |
| `.npz` | Raw floats (compressed) | Full | Smaller on disk, slower to load |
| PNG (16-bit grayscale) | Quantized integers | 65,536 levels | Requires float↔int rescaling |
| PNG (8-bit grayscale) | Quantized integers | 256 levels | Too coarse for spectral detail |
| JPEG | Lossy compressed | Poor | Not suitable for training data |
| TFRecord | Raw bytes (any dtype) | Full | Native `tf.data`; no Python callback; preferred for production (→ Goal 3C) |
| HDF5 (`.h5`) | Raw arrays | Full | Good for storing many tracks in one file with metadata |

#### When image files make sense

Image storage is appropriate when using **ImageNet-pretrained backbones** (ResNet, EfficientNet, ViT) that expect 3-channel uint8 input. In that case, the spectrogram is intentionally rendered as an RGB PNG and the quantization is an acceptable trade-off for access to pretrained weights. This is a potential direction in Goal 2 if pretrained vision models are explored.

#### Pipeline summary

```
MP3 file
  -> ffmpeg  -> float32 PCM (raw audio samples)
  -> librosa -> 2D float32 mel power matrix  (128 x T)
  -> log1p() -> 2D float32 log-mel matrix    (128 x 2582)
  -> np.save -> .npy file on disk            [cache]
  -> np.load -> float32 array
  -> (x - mu) / std  -> normalized tensor   [tf.data, train-only stats]
  -> Conv2D input
```

The `.imshow()` calls in the notebooks *visualize* the same `.npy` arrays for human inspection — those rendered images are never saved to disk and play no role in training.

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

### Goal 3 — Production Robustness (Structural Improvements)

Address three known structural limitations carried over from the Goal-1 baseline:

1. **Variable-length input support:** Replace Flatten (which bakes in a fixed time dimension) with GlobalAveragePooling2D or attention pooling so the model can accept tracks of any duration, not just 30 s.
2. **Class-imbalance handling:** Introduce explicit imbalance mitigation during training (e.g., class weights, focal loss, or oversampling) so the model does not bias toward dominant genres.
3. **XPU/GPU-friendly input pipeline:** Migrate from per-track .npy + `tf.numpy_function` to a streaming format (TFRecords or WebDataset) to reduce CPU bottlenecks and improve accelerator utilization.

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
* Log-mel: n_mels (e.g., 128), log compression via `log1p`
* MFCC: computed from the same mel configuration, keep n_mfcc (e.g., 20–40)
* Feature storage: per-track `.npy` files (float32, no normalization baked in); normalization applied at load time by the `tf.data` pipeline — see Section 2C for the full rationale and format comparison.

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

The phase 2 philosophy: stop treating *consistency* as sacred — optimize the whole system like an engineer, not a scientist doing a controlled ablation.

Each step introduces one major change, evaluated against B1:

* **B1.1**: B1 + SpecAugment (time masking + frequency masking)
  *Why*: large robustness gain without changing labels; particularly effective for log-mel.

* **B1.2**: B1.1 + improved optimizer + LR schedule
  Preferred options (both are strong):
  * **AdamW** (Adam + weight decay) + cosine decay — fast convergence, better generalization
  * **SGD momentum=0.9** + cosine decay — often strong final accuracy for CNNs
  Add: **warmup** for the first 1–5% of steps (prevents early instability).
  Add: **early stopping on val Macro-F1** rather than val loss — aligns stopping criterion with the imbalance-aware goal.

* **B1.3**: B1.2 + regularization
  * Weight decay / L2 (cheap and effective)
  * Dropout after dense layers or late conv blocks

* **B1.4**: B1.3 + mixup / label smoothing (as appropriate)

* **B1.5+ (future work):** Architecture upgrades, pretrained embeddings, Transformers
  *Note: architecture changes are deferred to future work. In the current scope, focus solidifies around training recipe (optimizer + LR + regularization + augmentation) and data strategy (random crops). Architecture upgrades, when time/budget allows, will be cleaner to interpret if training recipe is already locked in.*

### Optional model families (if time allows)

* **ResNet-like 2D CNN** with BatchNorm + ReLU + strided conv blocks + GlobalAveragePooling2D
  *This single upgrade often yields a bigger jump than swapping between MFCC and log-mel.*

* **CRNN** (CNN front-end + GRU/LSTM back-end): good for capturing longer rhythmic and temporal structure.

* **Attention pooling over time**: learned weighting of time steps; sits between GAP and full self-attention cost.

* **Audio Transformers** on log-mel patches: strong but heavier; worth exploring if compute allows.

* **Pretrained audio embeddings** (high leverage, especially if training data is limited):
  * VGGish / YAMNet-style embeddings
  * OpenL3 / PANNs / HTSAT / AST-style features
  * Workflow: freeze pretrained encoder, train a small classifier head on top
  * *Why*: genre classification benefits greatly from representations learned on large audio corpora; the pretrained model provides a rich vocabulary of audio concepts that would take many FMA epochs to discover from scratch.

### Recommended execution order (current scope)

This sequence gives consistent, interpretable gains with minimal wasted experiments:

1. Keep log-mel input (best information content as starting point)
2. Add SpecAugment + AdamW/SGD-cosine + LR warmup + early stopping on Macro-F1
3. Switch to random 10–15s crops during training + multi-crop voting at eval

These three steps form the **current phase 2 scope**. They target training recipe and data strategy — areas where improvements are measurable and reproducible.

#### Future work (out of scope for now):

4. Upgrade to ResNet-ish CNN with GlobalAveragePooling2D
5. Pretrained audio embeddings or Transformer back-ends

Architecture and model family changes will be revisited in a follow-up phase once the optimized training recipe is stable and understood. This keeps comparisons clean: improvement from architecture is easier to isolate when training is already locked in.

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

## 5.3 Goal 3: Production Robustness

Goal 3 can be pursued in parallel with or after Goal 2. Each improvement is evaluated independently against the best available baseline (B1 or the current Goal-2 best).

### 3A — Variable-length input (remove Flatten)

The current CNN uses Flatten → Dense, which requires every input to have exactly the same time dimension (2582 frames = 30 s at sr=22050, hop=256). This breaks on shorter or longer clips.

Two complementary strategies:

**3A-i: Random crop training policy** (simple, strong, low cost)

* Train on **random 10–15 s crops** from each 30 s track; new crop sampled each epoch.
* Evaluate with:
  * single centre crop (fast baseline), and/or
  * multi-crop voting — e.g. 3–5 crops averaged (higher accuracy, slower).
* Benefits: acts as augmentation, reduces over-reliance on one lucky segment, lets the model learn genre cues that appear intermittently, and naturally decouples training from the 30 s constraint.

**3A-ii: Architecture-level variable-length support**

* Replace `Flatten` with `GlobalAveragePooling2D` (or `GlobalMaxPooling2D`, or learned temporal attention pooling over the time axis).
* The model then accepts any feature width at inference time without padding artifacts.
* Re-train on the same split and compare macro-F1 to B1.
* Optionally test on variable-length clips to confirm length-agnosticism.

Both 3A-i and 3A-ii can be applied together for maximum benefit.

### 3B — Class-imbalance mitigation

FMA-medium has 16 naturally imbalanced genres. The current pipeline uses plain `categorical_crossentropy` with uniform sample weights, which can bias the model toward majority genres.

Options to evaluate (one at a time, ablation style):

* **Class weights:** Inverse-frequency weights passed to `model.fit(class_weight=...)`.
* **Focal loss:** Down-weights well-classified examples, emphasizing hard/minority classes.
* **Oversampling / undersampling:** Balanced sampling within `tf.data` pipeline.

Primary metric: per-genre recall balance and macro-F1 improvement.

### 3C — Accelerator-friendly input pipeline

The current pipeline loads per-track `.npy` files via `tf.numpy_function`, which pins work to CPU and prevents XPU/GPU overlap.

* Convert cached features to **TFRecord shards** (or WebDataset tar archives) so `tf.data` can prefetch and decode without Python callbacks.
* Benchmark throughput (samples/s) and accelerator utilization before vs. after.
* Ensure the new format still supports per-band normalization with train-only statistics.

### Goal-3 Reporting

For each improvement (3A, 3B, 3C), report:

* ΔMacro-F1 and ΔAccuracy relative to baseline
* Qualitative benefit (e.g., variable-length support, per-genre recall balance, throughput)
* Whether it is adopted into the final system

---

## 6. Deliverables

### Code and Reproducibility

* Fully reproducible training pipeline with configs for:

  * Goal 1 MFCC baseline (B0)
  * Goal 1 log-mel baseline (B1)
  * Goal 2 optimized variants (B1.1, B1.2, …)
  * Goal 3 robustness variants (3A, 3B, 3C)
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

3. **Goal 3 Report (Production Robustness)**

   * Variable-length support results (3A)
   * Class-imbalance handling results (3B)
   * Input pipeline throughput comparison (3C)
   * Which improvements are adopted into the final system

4. **Final Summary (Executive)**

   * "How much better is log-mel than MFCC under a controlled setup?"
   * "How far can optimized log-mel go beyond the baseline?"
   * "What structural improvements make the system more robust and efficient?"

---

## 7. Risks and Mitigations

| Risk                       | Impact                | Mitigation                                      |
| -------------------------- | --------------------- | ----------------------------------------------- |
| Leakage via normalization  | Inflated test results | Train-only stats; strict split handling         |
| Unfair comparison (Goal 1) | Invalid conclusions   | Same CNN + same recipe; only features change    |
| Genre imbalance            | Misleading accuracy   | Macro-F1 primary; per-class breakdown           |
| Non-determinism            | Unstable conclusions  | Multi-seed runs; mean ± std                     |
| Compute cost               | Slow iteration        | Feature caching; batch extraction; fixed budget |
| Validation overfitting     | Phases 2–3 look better than they are | Hold test set untouched until final eval; tune only against val; track per-genre confusion to understand *what changed* |

---

## 8. Success Criteria

### Goal 1 success

* B0 and B1 are reproducible and comparable
* Clear measurement of representation effect (log-mel vs MFCC) with macro-F1 emphasis

### Goal 2 success

* Demonstrated improvements over B1 with documented deltas
* A best-performing log-mel-based model with a clear “what worked” story
### Goal 3 success

* Model accepts variable-length inputs (not locked to 30 s)
* Explicit imbalance handling improves per-genre recall balance and macro-F1
* Input pipeline throughput measurably improves with TFRecords/WebDataset
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
    g3_3a_gap_pooling.yaml
    g3_3b_class_weights.yaml
    g3_3c_tfrecords.yaml
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
* **Goal 3** tackles structural limitations (fixed-length input, class imbalance, CPU-bound pipeline) to produce a more robust and efficient system.

The final output will answer:

1. *How much does log-mel help compared to MFCC under strict controls?*
2. *How good can log-mel+deep learning get on FMA-medium with reasonable optimization effort?*
3. *What structural improvements make the system production-ready and more efficient?*

---

See the companion reading guide for interpreting baseline comparisons: [@How-To-Read-MFCC-vs-LogMel-Guidelines](./@How-To-Read-MFCC-vs-LogMel-Guidelines)

---

## Appendix A: Baseline Comparison Guidelines (MFCC+CNN vs Log-mel+CNN)

The goal is a clean ablation: **same pipeline and model, different input representation**. To make that comparison trustworthy, lock down two things: **reproducibility** and **imbalance-aware evaluation**.

### A.1) Seed everything (reproducible comparison)

#### What to do

Set a single `SEED` and apply it consistently to:

* **Python** RNG (`random`)
* **NumPy** RNG (`numpy.random`)
* **TensorFlow** RNG (`tf.random`)
* **Dataset shuffling**: make it deterministic with
  `ds.shuffle(buffer, seed=SEED, reshuffle_each_iteration=True)`

#### Why it matters

Deep learning training is naturally noisy because of:

* random weight initialization
* random minibatch order (shuffle)
* nondeterministic kernels (sometimes)

If you don't seed, two runs can differ enough that you might "discover" a fake winner just due to randomness. Seeding turns your experiment from:

> "MFCC won this time"
> into
> "Under the same conditions, MFCC vs log-mel differ by X in a repeatable way."

That's the whole spirit of a baseline: **stable and comparable**.

#### Why deterministic shuffle matters (even if both models shuffle)

You want:

* **train**: shuffle ON (but repeatable)
* **val/test**: shuffle OFF

Shuffle is good for learning, but if it's not seeded, it becomes another uncontrolled variable. Seeded shuffle gives you the best of both worlds:

* SGD sees mixed batches (better optimization)
* both representations see effectively the same training dynamics across runs

### A.2) Report metrics that respect class imbalance

FMA-style genre datasets are **long-tailed**: some genres dominate (Rock/Electronic/Experimental), others are rare. That makes "plain accuracy" misleading.

#### What to report (make this the official baseline output)

1. **Accuracy**
   * Why: simple, intuitive "overall correctness"
   * Limitation: can look great even when the model ignores rare genres

2. **Macro-F1**
   * What it is: compute F1 per class, then average equally across classes
   * Why: forces the model to "care" about small genres; each genre counts the same
   * Interpretation: better indicator of "balanced genre competence"

3. **Confusion matrix + per-class F1**
   * Confusion matrix: shows *which genres get confused with which* (e.g., Rock vs Pop, Electronic vs Experimental)
   * Per-class F1: tells you where the model is strong/weak and whether improvements come from minority classes or just the big ones

#### Why this is essential for your comparison

If MFCC and log-mel differ mainly in how they handle minority genres, accuracy might not show it. Macro-F1 and per-class F1 will.

### A.3) What you should expect (typical pattern)

* **Log-mel + CNN often leads on Macro-F1**, especially improving minority/texture-heavy genres.
  Reason: log-mel retains richer time–frequency detail, so the CNN can pick up timbre/texture cues that matter for genre.

* **MFCC + CNN can be surprisingly close** when:
  * the model is small (limited capacity)
  * training is modest
  * genre cues are dominated by coarse spectral shape rather than fine detail

  Reason: MFCC is a strong compression/regularization that sometimes generalizes well when data or model capacity is constrained.

So: **accuracy may look similar**, but **macro-F1 and per-class F1 often reveal the real difference**.

### A.4) Implementation status

Both baseline notebooks apply these two rules:

| Guideline | Where | Status |
| --- | --- | --- |
| `SEED = 42` with `random.seed`, `np.random.seed`, `tf.random.set_seed` | Config cell (Section 2) in both notebooks | ✅ Done |
| `ds.shuffle(…, seed=SEED, reshuffle_each_iteration=True)` for train; shuffle OFF for val/test | `make_dataset()` in both notebooks | ✅ Done |
| Accuracy + Macro-F1 (top-line per split) | `eval_dataset()` in both notebooks | ✅ Done |
| Per-class F1 via `classification_report` | `eval_dataset()` in both notebooks | ✅ Done |
| Confusion matrix (test set) | Section 9b in both notebooks | ✅ Done |
| All metrics persisted in JSON run report | `evaluation.splits.{train,validation,test}` in run report | ✅ Done |

Notebooks:
* `MelCNN-MGR/notebooks/baseline_mfcc_cnn_v5.ipynb` — MFCC baseline (B0)
* `MelCNN-MGR/notebooks/baseline_logmel_cnn_v1.ipynb` — Log-mel baseline (B1)
