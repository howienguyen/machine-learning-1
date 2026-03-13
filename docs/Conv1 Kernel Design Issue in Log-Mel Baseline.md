## Document: Conv1 Kernel Design Issue in Log-Mel Baseline

**Date:** Friday, March 6, 2026 3:00:50 PM

### Context and Original Design Goal

In this project, two baseline pipelines were implemented for music genre classification:

1. **MFCC + CNN** (MelCNN-MGR/model_training/baseline_mfcc_cnn_v5.ipynb)
2. **Log-Mel Spectrogram + CNN** (MelCNN-MGR/model_training/baseline_logmel_cnn_v1.ipynb)

Both pipelines operate on the same audio dataset and are intended to serve as **baseline models** for later improvements.

The original goal when designing these baselines was to achieve **maximum experimental consistency** between the two approaches. Specifically, the intention was to isolate the effect of **input representation** while keeping all other factors as similar as possible.

Therefore the following aspects were intentionally aligned between the two notebooks:

* Same dataset split
* Same training configuration
* Same CNN architecture
* Same optimizer and training schedule
* Same batch size and epochs
* Same input duration (30 seconds)
* Same frame parameters (FFT, hop length, etc.)

To preserve this architectural symmetry, the **first convolutional layer (Conv1)** was defined using the same rule in both models:

```
kernel = (freq_bins, 10)
```

Where:

* `freq_bins = 13` for MFCC
* `freq_bins = 128` for Log-Mel

Thus:

| Representation | Input Shape | Conv1 Kernel |
| -------------- | ----------- | ------------ |
| MFCC           | (13, T, 1)  | (13, 10)     |
| Log-Mel        | (128, T, 1) | (128, 10)    |

At first glance this appears fair and symmetric.

However, this design unintentionally introduced a **representation mismatch**, which negatively affected the Log-Mel model.

---

# The Problem

### Conv1 Spanning the Entire Frequency Axis

In the Log-Mel baseline, the first convolution layer uses:

```
kernel = (128, 10)
```

This means the convolution filter spans the **entire frequency axis**.

Effectively, each filter performs:

[
y_t = \sum_{f=1}^{128} \sum_{\tau=1}^{10} W_{f,\tau} X_{f,t+\tau}
]

Where:

* (X) is the log-mel spectrogram
* (W) is the convolution kernel

This forces the model to learn a **global spectral template across all frequencies** in the very first layer.

---

# Why This Is Problematic for Log-Mel

### 1. Log-Mel is a Local Time-Frequency Representation

A log-mel spectrogram is essentially a **time-frequency image**.

Important properties:

* Local frequency patterns matter
* Harmonic stacks appear as vertical structures
* Percussion appears as broadband bursts
* Timbre appears as localized textures

CNNs normally learn these patterns **hierarchically**:

| Layer  | Function                     |
| ------ | ---------------------------- |
| Conv1  | detect small local patterns  |
| Conv2  | combine patterns into motifs |
| Conv3+ | detect global structure      |

Using `(128, 10)` bypasses this hierarchy and forces Conv1 to learn **global patterns immediately**.

---

### 2. Parameter Explosion

Conv1 parameter count increases dramatically:

| Representation | Kernel Size | Parameters per Filter |
| -------------- | ----------- | --------------------- |
| MFCC           | 13 × 10     | 130                   |
| Log-Mel        | 128 × 10    | 1280                  |

That is **~10× more parameters in the first layer**.

This increases:

* optimization difficulty
* overfitting risk
* sensitivity to spectral variation

---

### 3. MFCC Already Performs Spectral Compression

MFCC features are derived from log-mel spectrograms using a **Discrete Cosine Transform (DCT)**:

[
c_t = D \cdot e_t
]

Where:

* (e_t) = log-mel vector
* (D) = DCT matrix
* (c_t) = MFCC vector

Keeping only the first 13 coefficients effectively retains the **smooth spectral envelope**.

Thus the MFCC representation is already **compressed and globalized**.

Because of this, using a kernel spanning all MFCC coefficients `(13,10)` is not problematic.

The MFCC space already represents global spectral structure.

---

# Resulting Behavior

Because of the architecture mismatch:

* The Log-Mel model is forced to solve a **harder learning problem**
* The CNN cannot exploit **local time-frequency structure**
* The optimization becomes unstable

This leads to the unexpected outcome:

```
MFCC baseline accuracy > Log-Mel baseline accuracy
```

Even though Log-Mel generally contains **more detailed information** and should typically outperform MFCC in modern CNN pipelines.

Thus the issue is not the representation itself but the **architecture applied to that representation**.

---

# Root Cause

The root cause is the assumption that **architectural symmetry equals fairness**.

In reality there are two types of fairness:

### Ablation Fairness

Keep the model identical and change only the input.

### Representation Fairness

Allow minimal architectural adjustments so each representation is modeled appropriately.

The original design prioritized **ablation fairness**, but inadvertently violated **representation fairness**.

---

# Recommended Solutions

## Solution 1 — Use Local Convolution Kernels (Recommended)

For Log-Mel inputs, Conv1 should detect **local patterns**, not global ones.

Recommended kernels:

```
Conv2D(filters, (3,3))
Conv2D(filters, (5,5))
Conv2D(filters, (7,3))
```

Example:

```
Conv2D(32, (5,5), padding="same", activation="relu")
MaxPool2D((2,2))
```

Benefits:

* captures local spectrogram textures
* reduces parameters
* aligns with CNN design principles
* improves training stability

---

## Solution 2 — Add Pooling Layers Early

Pooling helps aggregate local frequency information gradually.

Example structure:

```
Conv2D(32, (5,5))
MaxPool2D((2,2))

Conv2D(64, (3,3))
MaxPool2D((2,2))
```

This allows the model to learn:

1. local spectral features
2. mid-level motifs
3. global genre patterns

---

## Solution 3 — Use Global Pooling Instead of Flatten

Replacing `Flatten()` with:

```
GlobalAveragePooling2D()
```

reduces parameter count and improves generalization.

---

# Recommended Baseline Architecture for Log-Mel

Example improved baseline:

```
Input: (128, T, 1)

Conv2D(32, (5,5), padding="same")
MaxPool2D((2,2))

Conv2D(64, (3,3), padding="same")
MaxPool2D((2,2))

Conv2D(128, (3,3), padding="same")

GlobalAveragePooling2D()

Dense(num_classes, softmax)
```

This design aligns with standard audio CNN architectures.

---

# Key Lesson

This issue highlights an important methodological principle:

> **Input representations define the geometry of the learning problem.
> The model architecture must align with that representation.**

Even when two representations encode the same underlying data, they may require **different modeling strategies**.

Applying the same operations blindly can produce misleading conclusions about representation quality.

---

## Lesson Learned

A key lesson from this investigation is that **machine learning operations must be aligned with the structure of the representation they operate on**. Applying the same method to different representations is not always meaningful, even when those representations originate from the same underlying data.

An intuitive way to understand this is through an analogy with mathematics.

In mathematics, operations are not universally applicable to every object. They are defined on **specific spaces with specific structures**. For example:

* Vector addition is defined in a **vector space**.
* A dot product requires an **inner product space**.
* Matrix multiplication requires **compatible dimensions**.
* Differentiation requires a notion of **smoothness**.

If an operation is applied to an object outside the space where it is defined, the computation may still produce a numerical result, but that result does not necessarily carry meaningful interpretation.

A similar principle applies in machine learning. Before a model can operate on data, the data is transformed into a representation:

[
\phi(x) \in \mathcal{Z}
]

Here, (x) is the original input (e.g., audio), and (\phi(x)) is the representation used by the model. Different representations produce different **feature spaces**, and model architectures implicitly assume certain structures in those spaces.

In this project, two representations were used:

**Log-Mel Spectrogram**

The log-mel spectrogram forms a **time–frequency representation** that behaves much like a two-dimensional image. In this space:

* Nearby frequencies are related.
* Local spectral patterns are meaningful.
* CNNs can effectively learn using **local convolution kernels** that detect small motifs and build hierarchical features over layers.

**MFCC**

MFCC features, on the other hand, are obtained through a transformation of the log-mel spectrum using a Discrete Cosine Transform (DCT):

[
c_t = D,e_t
]

where (e_t) represents the log-mel spectrum and (D) is the DCT matrix. The resulting MFCC coefficients are **global mixtures of the entire spectrum**, and the representation mainly captures the **smooth spectral envelope**.

Because of this transformation:

* The MFCC “frequency axis” is not a literal frequency axis.
* Each coefficient summarizes global spectral structure.
* Spatial locality across coefficients is less meaningful.

This difference leads to an important architectural implication.

When a convolution kernel spans the entire frequency dimension:

* **In MFCC space**: spanning all 13 coefficients effectively reads the whole compressed spectral summary. This is reasonable because MFCCs already encode global information.

* **In log-mel space**: spanning all 128 mel bins forces the network to match a global spectral template in the very first layer. This bypasses the natural hierarchical learning process of CNNs and prevents the model from exploiting local time–frequency patterns.

Thus, although the same convolution rule was applied in both baselines to maintain architectural symmetry, it actually imposed a mismatch for the log-mel representation.

The deeper lesson can be summarized as follows:

> The representation determines the structure of the feature space.
> Model operations must respect that structure.

Or, more intuitively:

> The same data can have different “types” depending on how it is represented.
> Applying the wrong operator to the wrong type may still produce a result, but not the result you intended.

This insight highlights an important methodological principle for machine learning experiments: **representation choice and model architecture are tightly coupled**. When the representation changes, the assumptions built into the model must be reconsidered.


---

# Summary

The Log-Mel baseline originally used a `(128,10)` convolution kernel to maintain architectural symmetry with the MFCC model.

While this approach satisfied **ablation fairness**, it created an architectural mismatch that prevented the CNN from effectively learning from the log-mel representation.

The solution is to adopt **local convolution kernels and hierarchical feature extraction**, which better aligns the model with the structure of spectrogram data.

With these adjustments, the Log-Mel model is expected to outperform the MFCC baseline, as commonly observed in modern audio classification systems.


---
---

# Development Plan: baseline_logmel_cnn_v10.ipynb

**Date:** Friday, March 6, 2026 3:19:50 PM
**Parent:** `MelCNN-MGR/model_training/baseline_logmel_cnn_v1.ipynb`  
**Reference:** `docs/Conv1 Kernel Design Issue in Log-Mel Baseline.md`

---

## Motivation

`baseline_logmel_cnn_v1.ipynb` uses a `(128, 10)` Conv1 kernel that spans the entire frequency axis, collapsing all 128 mel bands in a single layer. This prevents the CNN from learning local spectro-temporal patterns hierarchically and produces worse results than the MFCC baseline — despite log-mel being a richer representation.

**v10 fixes the architectural mismatch** by replacing the global-frequency Conv1 with a proper local-convolution architecture that respects the 2D time-frequency structure of log-mel spectrograms.

---

## Scope of Changes (v1 → v10)

Only the **model architecture** (Section 6) changes. Everything else is carried over from v1 unchanged:

| Aspect | Change? | Notes |
|--------|---------|-------|
| Imports | No | Same as v1 |
| Configuration & hyperparameters | Minimal | Same SUBSET, SEED, audio params. Only EPOCHS/LR may be tuned |
| Device setup | No | Same CUDA → XPU → CPU fallback |
| Manifest loading | No | Same split parquets |
| Log-mel extraction & caching | No | Same 128-band log-mel, same cache |
| Preprocessing & tf.data pipeline | No | Same per-band z-normalization |
| **Model architecture (Section 6)** | **YES** | **Core change — see below** |
| Compile & train (Section 7) | Minor | Optimizer/LR may change for new architecture |
| Training history plot | No | Same visualization |
| Evaluation | No | Same eval code |
| Inference | No | Same single-sample inference |
| Runtime summary | No | Same timing display |

---

## New Model Architecture (Section 6)

### Design Principles

1. **Local convolution kernels** — small (3×3), (5×5) or (7×3) filters to detect local time-frequency patterns
2. **Hierarchical feature extraction** — local → mid-level → global, across multiple conv+pool stages
3. **MaxPool2D for gradual spatial reduction** — instead of aggressive stride-based downsampling
4. **GlobalAveragePooling2D** — replaces Flatten to reduce parameters and improve generalization
5. **BatchNormalization** — stabilize training with deeper architecture
6. **Dropout** — regularize before the classifier head

### Proposed Architecture

```
Input: (128, 2582, 1)

# Block 1 — local spectro-temporal features
Conv2D(32, (5, 5), padding="same", activation="relu")
BatchNormalization()
MaxPool2D((2, 4))                            # → (64, 645, 32)

# Block 2 — mid-level motifs
Conv2D(64, (3, 3), padding="same", activation="relu")
BatchNormalization()
MaxPool2D((2, 4))                            # → (32, 161, 64)

# Block 3 — higher-level patterns
Conv2D(128, (3, 3), padding="same", activation="relu")
BatchNormalization()
MaxPool2D((2, 4))                            # → (16, 40, 128)

# Block 4 — global structure
Conv2D(128, (3, 3), padding="same", activation="relu")
BatchNormalization()
MaxPool2D((2, 2))                            # → (8, 20, 128)

# Classifier head
GlobalAveragePooling2D()                     # → (128,)
Dropout(0.3)
Dense(N_CLASSES, activation="softmax")
```

### Parameter Comparison

| Model | Conv1 params/filter | Total params (approx) | Conv1 output freq dim |
|-------|--------------------|-----------------------|----------------------|
| v1 (128,10) kernel | 1,280 | ~175K | 1 (collapsed) |
| v10 (5,5) kernel | 25 | ~250K | 128 (preserved) |

v10 has moderately more total parameters but **distributes them across layers** instead of front-loading them into Conv1. The effective capacity per layer is more balanced.

---

## Training Configuration Changes

| Parameter | v1 | v10 | Rationale |
|-----------|-----|------|-----------|
| Optimizer | SGD lr=1e-3 | Adam lr=1e-3 | Adam converges faster for deeper architectures |
| Epochs | 20 | 30 | More layers need more epochs; early stopping will guard |
| Batch size | 16 | 16 | Unchanged |
| Loss | categorical_crossentropy | categorical_crossentropy | Unchanged |
| Callbacks | None | EarlyStopping(patience=5) + ReduceLROnPlateau | Prevent overfitting, adaptive LR |

---

## Implementation Steps

### Step 1: Copy v1 notebook as v10 skeleton
- Duplicate `baseline_logmel_cnn_v1.ipynb` → `baseline_logmel_cnn_v10.ipynb`
- Update title and version references in markdown cells

### Step 2: Update Section 1 — Header markdown
- Change title to: `# FMA Baseline - Log-Mel → 2D CNN (Version 10)`
- Add description noting this is the representation-fair architecture
- Reference `docs/Conv1 Kernel Design Issue in Log-Mel Baseline.md`

### Step 3: Update Section 5 — Configuration
- Change `EPOCHS = 30`
- Keep all other config identical (same SUBSET, SEED, audio params, cache dirs)

### Step 4: Replace Section 6 — Model Architecture (CORE CHANGE)
- Replace the `build_model()` function with the new local-convolution architecture
- Add architecture rationale in a markdown cell before the code cell

### Step 5: Update Section 7 — Compile & Train
- Switch optimizer from SGD to Adam (lr=1e-3)
- Add EarlyStopping and ReduceLROnPlateau callbacks
- Update run directory naming to `logmel-cnn-v10-{timestamp}`
- Update run report metadata to reflect v10

### Step 6: Update reporting / metadata
- Update model name string to `logmel_2dcnn_v10`
- Update plot titles to reference v10
- Update run report to include architecture rationale

### Step 7: Verify and validate
- Confirm the model builds and `model.summary()` shows expected shapes
- Confirm training runs without errors
- Compare test accuracy and macro-F1 against v1 results

---

## Expected Outcomes

- **Higher test accuracy** than v1 (the (128,10) baseline), likely by 5–15+ percentage points
- **Higher Macro-F1** — local kernels capture genre-distinguishing spectro-temporal textures
- **Log-mel outperforms MFCC** — as expected in modern audio CNN literature
- **More stable training** — BatchNorm + Adam + smaller kernels = smoother optimization

---

## Files Affected

| File | Action |
|------|--------|
| `MelCNN-MGR/model_training/baseline_logmel_cnn_v10.ipynb` | **CREATE** — new notebook |
| `docs/@MelCNN-MGR-Todos.md` | **UPDATE** — add v10 task entry |

No changes to preprocessing, manifest, or cache infrastructure.
