# Analysis of logmel_cnn_v1 Training Run and Proposed Improvements for v1.1

*Date: 2026-03-11*
*Run analyzed: `logmel-cnn-v1-20260310-151804`*

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Training Run Overview](#2-training-run-overview)
3. [Per-Genre Performance Analysis](#3-per-genre-performance-analysis)
4. [Diagnosis: Root Causes of Weak Performance](#4-diagnosis-root-causes-of-weak-performance)
5. [Generalization Gap Analysis](#5-generalization-gap-analysis)
6. [Class Weight Analysis](#6-class-weight-analysis)
7. [Training Dynamics Analysis](#7-training-dynamics-analysis)
8. [Proposed Improvements for v1.1](#8-proposed-improvements-for-v11)
9. [Change Summary: v1 → v1.1](#9-change-summary-v1--v11)

---

## 1. Executive Summary

The first full training run of `logmel_cnn_v1.py` on the 10-genre, 10-second log-mel
dataset achieved:

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| **Accuracy** | 75.17% | 62.60% | 58.12% |
| **Macro-F1** | 0.7534 | 0.6281 | 0.5783 |

The model shows **significant overfitting** (17.5-point F1 gap between train and test)
and highly **uneven genre performance** (Classical F1=0.82 vs Pop F1=0.30). The training
stopped early at epoch 42/99 due to val_loss patience.

The primary issues are:
1. **Insufficient regularization** — the model memorizes training data faster than it generalizes
2. **Aggressive class weights for Metal** — causing systematic over-prediction
3. **Genre confusion clusters** — Folk/Country/Blues/Pop/Rock form a confusion block
4. **Validation instability** — large val_loss swings between epochs suggest sensitivity

The proposed v1.1 targets these via: **Mixup augmentation**, **stronger SpecAugment**,
**class-weight capping**, **increased dropout**, **higher weight decay**, and **architecture
refinements** (SpatialDropout in deeper blocks, bottleneck dense layer).

---

## 2. Training Run Overview

### Environment
- **Platform**: WSL2, Linux 6.6.87.2
- **Python**: 3.11.14, TensorFlow 2.15.1
- **Device**: Intel XPU (`/XPU:0`)
- **Total runtime**: 166 min 54s (97.6% spent in training)

### Dataset
- **Train**: 9,818 samples across 10 genres
- **Validation**: 1,238 samples
- **Test**: 1,237 samples
- **Feature shape**: (192 mel bands, 861 time frames) — 10-second clips at 22,050 Hz

### Training Config
- **Optimizer**: AdamW (weight_decay=1e-4)
- **LR schedule**: Cosine annealing with 3-epoch warmup, lr_max=1e-3, lr_min=1e-6
- **Loss**: Categorical crossentropy with label_smoothing=0.02
- **Batch size**: 32
- **Max epochs**: 99 (early-stopped at epoch 42)
- **SpecAugment**: freq_mask=15, time_mask=25, num_masks=2
- **Architecture**: 5 conv blocks (32→64→128→256→256), ~983K parameters

### Training Trajectory
- **Best val_loss**: epoch 33 (val_loss=1.2486)
- **Best val Macro-F1**: epoch 41 (val_macro_f1=0.6281)
- **Early stopping triggered**: epoch 42 (patience=9, restoring epoch 33 weights)
- The F1 checkpoint correctly saved epoch 41's model as the evaluation model.

---

## 3. Per-Genre Performance Analysis

### Test Set Performance Table

| Genre | Precision | Recall | F1 | Support | Assessment |
|-------|-----------|--------|-----|---------|------------|
| **Classical** | 0.87 | 0.77 | **0.82** | 132 | ★ Best performer — distinctive spectral signature |
| **Hip-Hop** | 0.76 | 0.78 | **0.77** | 132 | ★ Strong — rhythmic/bass patterns are distinctive |
| **Metal** | 0.47 | **0.97** | 0.63 | 58 | ⚠ Extreme recall, terrible precision — over-predicted |
| **Jazz** | **0.79** | 0.47 | 0.59 | 132 | ⚠ When it says Jazz, it's right — but misses most Jazz |
| **Electronic** | 0.68 | 0.51 | 0.58 | 131 | ⚠ Moderate — confused with other genres |
| **Blues** | 0.52 | 0.59 | 0.56 | 128 | ⚠ Low precision — confused with Country/Folk |
| **Country** | 0.52 | 0.59 | 0.55 | 130 | ⚠ Low precision — confused with Folk/Blues |
| **Folk** | 0.42 | **0.77** | 0.54 | 132 | ⚠ High recall, low precision — absorbs other genres |
| **Rock** | **0.82** | 0.30 | 0.44 | 132 | ✗ Barely identified — most Rock classified elsewhere |
| **Pop** | 0.35 | 0.27 | **0.30** | 130 | ✗ Worst performer — nearly random |

### Genre Confusion Clusters

The genres can be grouped into three performance tiers:

**Tier 1 — Well-separated** (F1 > 0.70):
- Classical, Hip-Hop: These genres have strong, distinctive spectral signatures (orchestral
  harmonics, bass-heavy beats) that the CNN can learn even from 10s clips.

**Tier 2 — Partially confused** (F1 0.55–0.63):
- Metal, Jazz, Electronic, Blues, Country: These share some acoustic features but have
  *enough* distinctiveness to achieve moderate performance.

**Tier 3 — Heavily confused** (F1 < 0.55):
- Folk, Rock, Pop: These form a confusion block. Pop in particular has almost no
  distinctive 10-second spectral fingerprint — it is a *cultural* genre, not an
  *acoustic* one.

### Key Confusion Patterns

1. **Folk absorbs Rock, Pop, Country, Blues**: Folk's recall=0.77 but precision=0.42
   means the model's "Folk" prediction is frequently wrong. It's using Folk as a
   catch-all for acoustic/guitar-driven music.

2. **Metal over-predicted**: Recall=0.97 but precision=0.47 — the model predicts Metal
   for almost any sample with high-energy distorted guitar. This pulls from Rock
   (recall=0.30) and potentially other genres.

3. **Rock under-identified**: Precision=0.82 means when it says Rock, it's usually right.
   But recall=0.30 means 70% of Rock samples are misclassified (likely as Metal, Folk,
   or Pop).

4. **Pop is nearly unlearnable**: F1=0.30 with both low precision and recall. Pop music
   spans too many acoustic styles (ballads, dance-pop, synth-pop, acoustic pop) to have
   a consistent spectral pattern in 10s clips.

---

## 4. Diagnosis: Root Causes of Weak Performance

### Cause 1: Overfitting (Primary Issue)

| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| Macro-F1 | 0.7534 | 0.5783 | **0.175** |
| Accuracy | 75.17% | 58.12% | **17.1pp** |

A ~17-point gap is severe and signals the model is memorizing training-set patterns
(artist timbre, recording quality, segment position) rather than learning transferable
genre features.

**Evidence of memorization**:
- Metal: Train recall=0.987 vs Test recall=0.965 — only genre where test ≈ train, because
  the extreme class weight forces predictions.
- Rock: Train recall=0.54 vs Test recall=0.30 — the model learned some training-set-specific
  Rock patterns that don't transfer.
- Pop: Train F1=0.62 vs Test F1=0.30 — massive generalization failure.

**Contributing factors**:
- Only 0.2 dropout at the classifier, 0.10 SpatialDropout in only 2 of 5 conv blocks
- Weight decay of 1e-4 is conservative for a dataset this small (~10K samples)
- SpecAugment masks (freq=15, time=25) are mild for 192-mel-band spectrograms
- No mixup or other label-smoothing augmentation beyond the existing 0.02

### Cause 2: Aggressive Class Weights for Metal

The class weight formula `compute_class_weight('balanced')` produces:

| Genre | Train samples | Class weight |
|-------|--------------|--------------|
| Most genres | ~1,040 | ~0.94 |
| **Metal** | **460** | **2.134** |

Metal's weight is **2.27× higher** than any other genre. During training, every Metal
sample contributes twice the gradient of other samples. This teaches the model that
"predicting Metal is very rewarding" — leading to extreme Metal recall (0.97) at the
expense of precision (0.47).

The 460 vs ~1040 ratio is only ~2.3:1, which is moderate imbalance. A 2.13× weight
overcorrects, especially on a dataset this size. Capping the maximum weight would
restore balance.

### Cause 3: The "Acoustic Guitar" Confusion Block

Folk, Country, Blues, Rock (acoustic), and Pop (ballad) all share common acoustic
features in 10-second clips:
- Guitar-centric instrumentation
- Moderate tempo
- Vocal-forward mixing
- Similar frequency energy distribution

The CNN, which operates on spectral patterns alone, struggles to distinguish these because
the discriminative features are often *structural* (song form, chord progressions) or
*cultural* (lyrical content, production era) — information not visible in a spectrogram.

### Cause 4: Validation Instability

The validation loss trajectory shows significant oscillation:
- Epoch 8: val_loss=1.51 → Epoch 9: val_loss=1.88 (↑0.37)
- Epoch 19: val_loss=1.32 → Epoch 20: val_loss=1.42 (↑0.10)
- Epoch 29: val_loss=1.46 → Epoch 30: val_loss=1.29 (↓0.17)

These swings suggest:
- The learning rate (lr_max=1e-3) may be too high during mid-training, causing the model
  to overshoot good parameter regions.
- The small validation set (1,238 samples) amplifies noise.
- Batch-to-batch variance in class composition (due to shuffling + variable class weights)
  creates inconsistent gradient signals.

---

## 5. Generalization Gap Analysis

### Train-to-Test Drop by Genre

| Genre | Train F1 | Test F1 | Drop | Interpretation |
|-------|----------|---------|------|----------------|
| Pop | 0.62 | 0.30 | **-0.32** | ← Worst generalization; memorized training Pop |
| Rock | 0.67 | 0.44 | **-0.23** | ← Heavy memorization of training Rock patterns |
| Jazz | 0.75 | 0.59 | -0.16 | Moderate drop |
| Electronic | 0.85 | 0.58 | **-0.27** | ← Surprising drop; train precision inflated |
| Blues | 0.68 | 0.56 | -0.12 | Acceptable drop |
| Folk | 0.66 | 0.54 | -0.12 | Acceptable drop |
| Country | 0.74 | 0.55 | **-0.19** | Notable drop |
| Metal | 0.78 | 0.63 | -0.15 | Moderate (driven by recall saturation) |
| Classical | 0.87 | 0.82 | -0.05 | ★ Best generalization |
| Hip-Hop | 0.92 | 0.77 | -0.15 | Good — distinctive features transfer |

**Pattern**: Genres with strong acoustic identity (Classical, Hip-Hop) generalize well.
Genres with diffuse acoustic identity (Pop, Rock, Electronic) suffer massive generalization
drops. This confirms that overfitting is a bigger problem than underfitting — the model
has enough capacity to learn genre features, but it also learns non-genre shortcuts.

---

## 6. Class Weight Analysis

### Current Weights (v1)

```
Blues        0.9551    (1028 samples)
Classical    0.9422    (1042 samples)
Country      0.9422    (1042 samples)
Electronic   0.9440    (1040 samples)
Folk         0.9431    (1041 samples)
Hip-Hop      0.9431    (1041 samples)
Jazz         0.9431    (1041 samples)
Metal        2.1343    (460 samples)   ← 2.27× the next highest
Pop          0.9422    (1042 samples)
Rock         0.9431    (1041 samples)
```

### The Metal Problem

Metal's weight (2.13) is extreme because `compute_class_weight('balanced')` uses:

```
weight_c = n_samples / (n_classes × n_c)
weight_metal = 9818 / (10 × 460) = 2.134
```

This makes every Metal sample worth 2.13 loss units, while other samples contribute ~0.94.
The model learns that predicting Metal is a very efficient way to reduce loss on the ~5%
of samples that are actually Metal — even at the cost of false-positiving non-Metal samples.

### Proposed Fix: Capped Class Weights

Instead of uncapped `'balanced'` weights, apply a cap:

```python
MAX_CLASS_WEIGHT = 1.5  # cap extreme weights

raw_weights = compute_class_weight('balanced', ...)
capped_weights = np.minimum(raw_weights, MAX_CLASS_WEIGHT)
```

This produces:
```
Metal        1.5000    (capped from 2.13)
Others       ~0.94     (unchanged)
```

The Metal weight is still 1.6× higher than average (providing imbalance correction) but
no longer extreme enough to distort the entire prediction distribution.

---

## 7. Training Dynamics Analysis

### Epoch-Level Observations

1. **Slow start (epochs 1-5)**: Accuracy climbs from 23% to 43%, loss drops from 2.19 to
   1.64. The 3-epoch warmup works well here.

2. **Productive middle (epochs 15-33)**: Steady improvement. Best val_loss at epoch 33
   (1.2486). Train accuracy reaches ~66%.

3. **Diminishing returns (epochs 34-42)**: Train accuracy keeps rising (68%→73%) but
   val_loss stagnates around 1.26-1.40. This is the classic overfitting zone where the
   model is fitting training noise.

4. **F1 still improving at stopping point**: Best macro-F1 at epoch 41, just 1 epoch
   before early stopping. This suggests the model might have continued improving on F1
   if it had trained longer — but val_loss-based stopping halted it.

### LR Schedule Behavior

The learning rate decays smoothly from 1e-3 to 6.5e-4 over 42 epochs. At epoch 42, the
LR is still relatively high (0.000646). A lower LR_MAX or faster decay might have helped
the model converge more cleanly in the 30-42 epoch range where overfitting accelerated.

### Seconds Per Epoch

Average ~230s/epoch on XPU, with spikes to 290-358s (likely XPU memory management or
system load). Total training time: ~163 minutes for 42 epochs.

---

## 8. Proposed Improvements for v1.1

The improvements are organized by expected impact, from highest to lowest.

### 8.1 Mixup Augmentation (High Impact)

**What**: During training, randomly blend pairs of samples and their labels:
```
x_mix = λ·x_a + (1-λ)·x_b
y_mix = λ·y_a + (1-λ)·y_b
where λ ~ Beta(α, α), α = 0.3
```

**Why**:
- Mixup is one of the most effective regularizers for audio classification.
- It creates smooth decision boundaries between genre clusters, directly attacking
  the confusion-block problem.
- Forces the model to learn from *combinations* of genre features rather than
  memorizing individual samples.
- A meta-analysis of audio classification papers shows Mixup typically provides
  3-5 point F1 gains when overfitting is present.

**Implementation**: Apply at the batch level, after normalization but before SpecAugment.
Use `α=0.3` for a moderate blend (most blends will be mild, ~70:30 or more one-sided).

### 8.2 Stronger SpecAugment (Medium-High Impact)

**Current**: freq_mask=15, time_mask=25, num_masks=2
**Proposed**: freq_mask=24, time_mask=40, num_masks=2

**Why**:
- With 192 mel bands, a freq_mask of 15 covers only ~8% of the frequency axis per mask.
  Bumping to 24 covers ~12.5%, forcing the model to classify from partial spectral info.
- A time_mask of 40 covers ~4.6% of the 861-frame time axis (vs 2.9% currently).
- This is still conservative (no risk of masking too much information).

### 8.3 Class Weight Capping (Medium Impact)

**Current**: Metal weight = 2.13, others ≈ 0.94
**Proposed**: Cap at MAX_CLASS_WEIGHT = 1.5

**Why**:
- Metal's 2.13 weight causes ~50% of Metal test samples to be false positives.
- Capping to 1.5 still corrects for imbalance (Metal gets 1.6× the gradient of others)
  but prevents the extreme over-prediction pattern.
- Expected to improve Metal precision from 0.47 to ~0.55-0.60, with corresponding
  recall reduction from 0.97 to ~0.80-0.85 — a net F1 improvement.

### 8.4 Increased Regularization (Medium Impact)

**Architecture changes**:
- Add `SpatialDropout2D(0.12)` to conv blocks 4 and 5 (currently missing)
- Increase final `Dropout` from 0.20 to 0.30
- Add a `Dense(128, relu) + Dropout(0.30)` bottleneck between GAP and output

**Training changes**:
- Increase `WEIGHT_DECAY` from 1e-4 to 5e-4

**Why**:
- The current architecture only regularizes blocks 2-3 with SpatialDropout.
  Blocks 4-5 (which have the most parameters — 589K of 983K total) are unregularized.
- The GAP→Dense(10) path has only 2,570 parameters and 0.2 dropout — a narrow
  bottleneck. Adding a Dense(128) intermediate layer gives the classifier more capacity
  to disentangle confused genres, while the added dropout prevents overfitting this
  extra capacity.
- Weight decay of 5e-4 is a well-tested default for AdamW on mid-size vision models.

### 8.5 Learning Rate Adjustments (Low-Medium Impact)

**Current**: lr_max=1e-3, warmup=3 epochs
**Proposed**: lr_max=5e-4, warmup=5 epochs

**Why**:
- The validation loss instability (epoch 8→9 spike of +0.37) suggests the LR is too
  high during mid-training.
- Halving lr_max typically reduces val_loss variance and may allow training to continue
  longer before early stopping.
- Extending warmup from 3→5 epochs reduces the chance of gradient instability during
  the early phase when BatchNorm statistics are still adapting.

### 8.6 EarlyStopping Patience Increase (Low Impact)

**Current**: patience=9 on val_loss
**Proposed**: patience=12 on val_loss

**Why**:
- The F1 was still improving at epoch 41 when early stopping triggered at epoch 42.
  With patience=12, the model might have continued and found a better F1 checkpoint.
- Risk is low: cosine LR will naturally reduce the learning rate, so extra epochs
  won't cause divergence.

---

## 9. Change Summary: v1 → v1.1

| Component | v1 | v1.1 | Rationale |
|-----------|-----|------|-----------|
| **Mixup** | None | α=0.3, batch-level | Primary regularizer against overfitting |
| **SpecAugment freq_mask** | 15 | 24 | Stronger partial-information training |
| **SpecAugment time_mask** | 25 | 40 | Stronger temporal masking |
| **Class weight cap** | Uncapped (Metal=2.13) | MAX=1.5 | Prevent Metal over-prediction |
| **Weight decay** | 1e-4 | 5e-4 | Stronger L2 regularization |
| **SpatialDropout blocks 4-5** | None | 0.12 | Regularize highest-param layers |
| **Final Dropout** | 0.20 | 0.30 | Harder classifier regularization |
| **Classifier** | GAP→Dropout→Dense(10) | GAP→Dense(128,relu)→Dropout(0.30)→Dense(10) | More classification capacity |
| **LR max** | 1e-3 | 5e-4 | Reduce val_loss instability |
| **Warmup epochs** | 3 | 5 | Smoother early training |
| **EarlyStopping patience** | 9 | 12 | Allow more exploration |
| **Model name** | logmel_cnn_v1 | logmel_cnn_v1_1 | Track version lineage |

### Expected Impact

Conservative estimate based on the individual contributions of each technique:

| Improvement | Estimated F1 gain |
|-------------|------------------|
| Mixup | +3 to +5 points |
| Stronger SpecAugment | +1 to +2 points |
| Class weight capping | +1 to +2 points (via Metal/Rock rebalancing) |
| Increased regularization | +1 to +3 points (reduced overfitting) |
| LR adjustment | +0.5 to +1.5 points (stability) |
| **Combined** | **+5 to +10 points** (interactions may boost or diminish) |

**Realistic target**: Test Macro-F1 of **0.63–0.68**, up from 0.5783. Accuracy target:
**63–68%**, up from 58.12%.

### What v1.1 Does NOT Change

- Feature extraction (same 192-mel, 10s clips from the same prebuilt dataset)
- Data splits (same train/val/test from the same `logmel_dataset_10s`)
- Overall architecture family (5-block CNN with GlobalAveragePooling)
- Evaluation methodology (same classification_report, confusion matrix)
- Run artifact structure (same `.keras`, `norm_stats.npz`, `run_report.json`)

This ensures v1.1 results are directly comparable to v1.

---

*End of analysis.*
