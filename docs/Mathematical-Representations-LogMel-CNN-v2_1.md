# Mathematical Representations — Log-Mel CNN v2.1

> Reference implementation: `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`

---

## Table of Contents

1. [Feature Extraction Pipeline](#1-feature-extraction-pipeline)
2. [Input Normalization](#2-input-normalization)
3. [CNN Architecture — Forward Pass](#3-cnn-architecture--forward-pass)
4. [Data Augmentation](#4-data-augmentation)
5. [Loss Function & Class Weighting](#5-loss-function--class-weighting)
6. [Optimizer — AdamW with Cosine Annealing & Warmup](#6-optimizer--adamw-with-cosine-annealing--warmup)
7. [Regularization Techniques](#7-regularization-techniques)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Hyperparameter Summary](#9-hyperparameter-summary)
10. [Parameter Count Breakdown](#10-parameter-count-breakdown)
11. [Compact Mathematical Representations](#11-compact-mathematical-representations)

---

## 1. Feature Extraction Pipeline

Audio waveforms are converted to log-mel spectrograms upstream by `preprocessing/2_build_log_mel_dataset.py`. The pipeline has three stages.

### 1.1 Short-Time Fourier Transform (STFT)

Given a discrete-time audio signal $x[n]$ sampled at $f_s = 22{,}050$ Hz and a window function $w[n]$ of length $N_{\text{FFT}} = 512$, the STFT at frame $m$ and frequency bin $k$ is:

$$
X[m, k] = \sum_{n=0}^{N_{\text{FFT}}-1} x[n + m \cdot H] \, w[n] \, e^{-j\frac{2\pi k n}{N_{\text{FFT}}}}
$$

where $H = 256$ is the hop length (stride between consecutive frames) and $k \in \{0, 1, \ldots, N_{\text{FFT}}/2\}$.

The power spectrogram is:

$$
S_{\text{power}}[m, k] = |X[m, k]|^2
$$

### 1.2 Mel Filterbank

The power spectrogram is mapped to the mel scale using a bank of $M = 192$ triangular filters. Each filter $\mathbf{h}_i$ ($i = 1, \ldots, M$) spans a frequency range defined by the mel scale:

$$
\text{mel}(f) = 2595 \cdot \log_{10}\!\left(1 + \frac{f}{700}\right)
$$

The mel spectrogram is the matrix product:

$$
S_{\text{mel}}[i, m] = \sum_{k=0}^{N_{\text{FFT}}/2} h_i[k] \cdot S_{\text{power}}[m, k], \quad i = 1, \ldots, M
$$

or in matrix form:

$$
\mathbf{S}_{\text{mel}} = \mathbf{H} \, \mathbf{S}_{\text{power}}
$$

where $\mathbf{H} \in \mathbb{R}^{M \times (N_{\text{FFT}}/2 + 1)}$ is the mel filterbank matrix.

### 1.3 Logarithmic Compression

The log-mel spectrogram applies a $\log(1+x)$ compression to handle the wide dynamic range of audio power:

$$
\mathbf{X}_{\text{logmel}}[i, m] = \ln\!\left(1 + S_{\text{mel}}[i, m]\right)
$$

This is equivalent to `np.log1p(spec)` in the codebase. The resulting feature matrix has shape:

$$
\mathbf{X}_{\text{logmel}} \in \mathbb{R}^{M \times T}
$$

where $T = \left\lfloor \frac{f_s \cdot d}{H} \right\rfloor$ and $d$ is the clip duration in seconds.

**Default configuration:**

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Sample rate | $f_s$ | 22,050 Hz |
| FFT size | $N_{\text{FFT}}$ | 512 |
| Hop length | $H$ | 256 |
| Mel bands | $M$ | 192 |
| Clip duration | $d$ | 15 s |
| Time frames | $T$ | $\lfloor 22050 \times 15 / 256 \rfloor = 1291$ |
| Feature shape | — | $(192, 1291)$ |

---

## 2. Input Normalization

Training-set statistics are computed per mel band using a streaming pass over all training samples. For mel band $i$:

$$
\mu_i = \frac{1}{N_{\text{total}}} \sum_{n=1}^{N_{\text{train}}} \sum_{m=1}^{T} \mathbf{X}_{\text{logmel}}^{(n)}[i, m]
$$

$$
\sigma_i = \sqrt{\frac{1}{N_{\text{total}}} \sum_{n=1}^{N_{\text{train}}} \sum_{m=1}^{T} \left(\mathbf{X}_{\text{logmel}}^{(n)}[i, m]\right)^2 - \mu_i^2 + \epsilon}
$$

where $N_{\text{total}}$ is the total number of time frames across all training samples and $\epsilon = 10^{-12}$.

Each input is standardized (z-score normalization) before being fed to the network:

$$
\hat{\mathbf{X}}[i, m] = \frac{\mathbf{X}_{\text{logmel}}[i, m] - \mu_i}{\sigma_i}
$$

The statistics $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are stored in `norm_stats.npz` and reused at inference time.

---

## 3. CNN Architecture — Forward Pass

The model input is $\hat{\mathbf{X}} \in \mathbb{R}^{M \times T \times 1}$ (height × width × channels). The architecture consists of 5 convolutional blocks followed by a classifier head.

### 3.1 Convolutional Block

Each block $b$ applies the sequence: **Conv2D → BatchNorm → ReLU → [SpatialDropout2D] → MaxPool2D**.

#### 3.1.1 2D Convolution (no bias)

For block $b$ with $C_{\text{in}}$ input channels and $C_{\text{out}}$ output channels using kernel size $(k_h, k_w)$ with "same" padding:

$$
\mathbf{Z}^{(b)}[c, i, j] = \sum_{c'=1}^{C_{\text{in}}} \sum_{p=0}^{k_h-1} \sum_{q=0}^{k_w-1} \mathbf{W}^{(b)}[c, c', p, q] \cdot \mathbf{A}^{(b-1)}[c', i+p-\lfloor k_h/2\rfloor, j+q-\lfloor k_w/2\rfloor]
$$

where $\mathbf{W}^{(b)}$ is the learned kernel tensor and bias is omitted (`use_bias=False`) since BatchNorm provides a learnable shift.

#### 3.1.2 Batch Normalization

For each channel $c$ in a mini-batch of size $B$:

$$
\hat{z}_{c} = \frac{z_c - \mathbb{E}[z_c]}{\sqrt{\text{Var}[z_c] + \epsilon}} \cdot \gamma_c + \beta_c
$$

where $\gamma_c$ and $\beta_c$ are learnable scale and shift parameters per channel, and $\epsilon = 10^{-3}$ (Keras default). During training, $\mathbb{E}[z_c]$ and $\text{Var}[z_c]$ are computed over the current mini-batch; during inference, exponential moving averages are used.

#### 3.1.3 ReLU Activation

$$
\text{ReLU}(z) = \max(0, z)
$$

#### 3.1.4 Spatial Dropout 2D

Drops entire feature maps (channels) rather than individual elements. During training, for each channel $c$, a Bernoulli mask $m_c \sim \text{Bernoulli}(1 - p)$ is sampled:

$$
\tilde{\mathbf{A}}^{(b)}[c, :, :] = \frac{m_c}{1-p} \cdot \mathbf{A}^{(b)}[c, :, :]
$$

The scaling by $\frac{1}{1-p}$ ensures the expected value is preserved (inverted dropout).

#### 3.1.5 Max Pooling 2D

With pool size $(2, 2)$ and stride $(2, 2)$:

$$
\mathbf{P}^{(b)}[c, i, j] = \max_{0 \le p, q < 2} \mathbf{A}^{(b)}[c, 2i+p, 2j+q]
$$

This halves both spatial dimensions at each block.

### 3.2 Block-by-Block Architecture

| Block | Filters $C_{\text{out}}$ | Kernel | Spatial Dropout $p$ | Output Shape |
|-------|--------------------------|--------|---------------------|--------------|
| 1 | 32 | $(5, 5)$ | — | $(96, 645, 32)$ |
| 2 | 64 | $(3, 3)$ | 0.05 | $(48, 322, 64)$ |
| 3 | 128 | $(3, 3)$ | 0.10 | $(24, 161, 128)$ |
| 4 | 256 | $(3, 3)$ | 0.15 | $(12, 80, 256)$ |
| 5 | 256 | $(3, 3)$ | 0.20 | $(6, 40, 256)$ |

The graduated spatial dropout schedule ($0.05 \to 0.20$) concentrates regularization in deeper layers where features are more abstract and prone to co-adaptation.

### 3.3 Classifier Head

#### 3.3.1 Global Average Pooling (GAP)

Collapses the spatial dimensions by averaging over all spatial positions:

$$
\mathbf{g}[c] = \frac{1}{H' \times W'} \sum_{i=1}^{H'} \sum_{j=1}^{W'} \mathbf{A}^{(5)}[c, i, j]
$$

For block 5 output of shape $(6, 40, 256)$: $\mathbf{g} \in \mathbb{R}^{256}$.

#### 3.3.2 Dense Bottleneck (256 units, ReLU)

$$
\mathbf{h} = \text{ReLU}\!\left(\mathbf{W}_{\text{fc1}} \, \mathbf{g} + \mathbf{b}_{\text{fc1}}\right)
$$

where $\mathbf{W}_{\text{fc1}} \in \mathbb{R}^{256 \times 256}$ and $\mathbf{b}_{\text{fc1}} \in \mathbb{R}^{256}$. The 256-unit width matches the backbone output dimension, preserving the full feature space (no compression bottleneck).

#### 3.3.3 Dropout

Standard element-wise dropout with $p = 0.20$:

$$
\tilde{h}_k = \frac{m_k}{1-p} \cdot h_k, \quad m_k \sim \text{Bernoulli}(1-p)
$$

#### 3.3.4 Softmax Output

For $K = 10$ genre classes:

$$
\hat{y}_k = \text{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}, \quad \mathbf{z} = \mathbf{W}_{\text{fc2}} \, \tilde{\mathbf{h}} + \mathbf{b}_{\text{fc2}}
$$

where $\mathbf{W}_{\text{fc2}} \in \mathbb{R}^{K \times 256}$ and $\hat{\mathbf{y}} \in [0, 1]^K$ with $\sum_k \hat{y}_k = 1$.

### 3.4 Complete Forward Pass Summary

$$
\hat{\mathbf{X}} \xrightarrow{\text{Block}_1} \xrightarrow{\text{Block}_2} \xrightarrow{\text{Block}_3} \xrightarrow{\text{Block}_4} \xrightarrow{\text{Block}_5} \xrightarrow{\text{GAP}} \mathbf{g} \xrightarrow{\text{FC}(256)} \xrightarrow{\text{Dropout}} \xrightarrow{\text{FC}(K)} \xrightarrow{\text{softmax}} \hat{\mathbf{y}}
$$

---

## 4. Data Augmentation

Two augmentation methods are applied during training only, in the order: **Mixup → SpecAugment**.

### 4.1 Mixup

For each mini-batch, samples are paired by random shuffling. A mixing coefficient $\lambda$ is drawn from a Beta distribution (via the ratio of two Gamma draws):

$$
\lambda_a \sim \text{Gamma}(\alpha, 1), \quad \lambda_b \sim \text{Gamma}(\alpha, 1), \quad \lambda = \frac{\lambda_a}{\lambda_a + \lambda_b}
$$

where $\alpha = 0.3$. The coefficient is then rectified to ensure the anchor dominates: $\lambda \leftarrow \max(\lambda, 1 - \lambda)$, so $\lambda \in [0.5, 1]$.

Given an anchor sample $(\mathbf{X}_i, \mathbf{y}_i)$ and a shuffled partner $(\mathbf{X}_j, \mathbf{y}_j)$:

$$
\tilde{\mathbf{X}} = \lambda \, \mathbf{X}_i + (1 - \lambda) \, \mathbf{X}_j
$$

$$
\tilde{\mathbf{y}} = \lambda \, \mathbf{y}_i + (1 - \lambda) \, \mathbf{y}_j
$$

where $\mathbf{y}_i, \mathbf{y}_j \in \{0, 1\}^K$ are one-hot label vectors and $\tilde{\mathbf{y}}$ becomes a soft label.

**Per-sample $\lambda$**: Each pair in the batch draws its own $\lambda$ independently (not a shared batch-level value).

### 4.2 Pre-Mixup Anchor-Based Class Weighting

When class imbalance is detected (max/min class ratio $> 1.05$), per-sample loss weights are computed from the anchor sample's class **before** mixing:

$$
w_i = \mathbf{c}^{\top} \mathbf{y}_i
$$

where $\mathbf{c} \in \mathbb{R}^K$ is the class weight vector. This preserves the class-balance signal that would otherwise be diluted by soft Mixup labels.

Class weights are computed via sklearn's "balanced" formula and capped:

$$
c_k^{\text{raw}} = \frac{N_{\text{train}}}{K \cdot N_k}, \quad c_k = \min\!\left(c_k^{\text{raw}},\; w_{\max}\right)
$$

where $N_k$ is the count of class $k$ in the training set and $w_{\max} = 1.5$.

### 4.3 SpecAugment

After Mixup, SpecAugment applies random masking along frequency and time axes. For each sample, $N_{\text{masks}} = 1$ mask is applied per axis.

**Frequency masking**: A contiguous band of $f$ mel bands is zeroed out:

$$
f \sim \text{Uniform}(0, F), \quad f_0 \sim \text{Uniform}(0, M - f)
$$
$$
\tilde{\mathbf{X}}[i, :] = 0, \quad \forall\; i \in [f_0, f_0 + f)
$$

where $F = 24$ (maximum frequency mask width).

**Time masking**: A contiguous span of $t$ frames is zeroed out:

$$
t \sim \text{Uniform}(0, T_{\text{mask}}), \quad t_0 \sim \text{Uniform}(0, T - t)
$$
$$
\tilde{\mathbf{X}}[:, j] = 0, \quad \forall\; j \in [t_0, t_0 + t)
$$

where $T_{\text{mask}} = 40$ (maximum time mask width).

---

## 5. Loss Function & Class Weighting

### 5.1 Categorical Cross-Entropy

With label smoothing $\varepsilon = 0$ (disabled, since Mixup already produces soft labels):

$$
\mathcal{L}_{\text{CE}}(\tilde{\mathbf{y}}, \hat{\mathbf{y}}) = -\sum_{k=1}^{K} \tilde{y}_k \, \log(\hat{y}_k)
$$

where $\tilde{\mathbf{y}}$ is the (possibly Mixup-blended) target and $\hat{\mathbf{y}}$ is the softmax output.

### 5.2 Weighted Loss

When class weights are active, the per-sample loss is scaled by the anchor weight:

$$
\mathcal{L}_{\text{weighted}}^{(i)} = w_i \cdot \mathcal{L}_{\text{CE}}(\tilde{\mathbf{y}}_i, \hat{\mathbf{y}}_i)
$$

The batch loss is:

$$
\mathcal{L}_{\text{batch}} = \frac{1}{B} \sum_{i=1}^{B} \mathcal{L}_{\text{weighted}}^{(i)}
$$

---

## 6. Optimizer — AdamW with Cosine Annealing & Warmup

### 6.1 AdamW Update Rule

AdamW decouples weight decay from the gradient-based update. At step $t$:

$$
\mathbf{m}_t = \beta_1 \, \mathbf{m}_{t-1} + (1 - \beta_1) \, \mathbf{g}_t
$$
$$
\mathbf{v}_t = \beta_2 \, \mathbf{v}_{t-1} + (1 - \beta_2) \, \mathbf{g}_t^2
$$
$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$
$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta_t \left( \frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda_{\text{wd}} \, \boldsymbol{\theta}_{t-1} \right)
$$

where:
- $\beta_1 = 0.9$, $\beta_2 = 0.999$ (Adam defaults)
- $\epsilon = 10^{-7}$
- $\lambda_{\text{wd}} = 10^{-4}$ (decoupled weight decay)
- $\eta_t$ is the scheduled learning rate

### 6.2 Gradient Clipping

Before the optimizer step, gradients are clipped by global norm:

$$
\hat{\mathbf{g}} = \begin{cases}
\mathbf{g} & \text{if } \|\mathbf{g}\|_2 \le c \\[4pt]
\displaystyle \frac{c}{\|\mathbf{g}\|_2} \cdot \mathbf{g} & \text{otherwise}
\end{cases}
$$

where $c = 1.0$ (`clipnorm=1.0`).

### 6.3 Cosine Annealing with Linear Warmup

The learning rate schedule consists of two phases:

**Phase 1 — Linear warmup** ($t < t_w$):

$$
\eta_t = \eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot \frac{t}{t_w}
$$

**Phase 2 — Cosine annealing** ($t \ge t_w$):

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\!\left(\pi \cdot \frac{t - t_w}{t_{\text{total}} - t_w}\right)\right)
$$

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Max learning rate | $\eta_{\max}$ | $5 \times 10^{-4}$ |
| Min learning rate | $\eta_{\min}$ | $10^{-6}$ |
| Warmup epochs | — | 5 |
| Total epochs | — | 102 |

---

## 7. Regularization Techniques

The v2.1 model employs multiple complementary regularization strategies:

### 7.1 Summary

| Method | Mathematical Effect | Configuration |
|--------|-------------------|---------------|
| **Mixup** | Convex combination of input-label pairs; creates virtual training examples in feature space | $\alpha = 0.3$, anchor-dominant ($\lambda \ge 0.5$) |
| **SpecAugment** | Zeroes contiguous frequency/time bands; forces robustness to partial information loss | $F=24$, $T=40$, 1 mask each |
| **Graduated Spatial Dropout** | Drops entire feature maps; prevents channel co-adaptation | $p \in \{0.05, 0.10, 0.15, 0.20\}$ across blocks 2–5 |
| **Final Dropout** | Standard element-wise dropout on the bottleneck | $p = 0.20$ |
| **Weight Decay (L2)** | Decoupled L2 penalty shrinks weights toward zero | $\lambda_{\text{wd}} = 10^{-4}$ |
| **Batch Normalization** | Normalizes hidden activations; provides implicit regularization via mini-batch noise | $\epsilon = 10^{-3}$ |
| **Early Stopping** | Halts training when val Macro-F1 plateaus | patience = 20 epochs |

### 7.2 Effective Regularization Strength

The total regularization can be conceptualized as an effective penalty:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \underbrace{\frac{\lambda_{\text{wd}}}{2} \|\boldsymbol{\theta}\|_2^2}_{\text{weight decay}} + \underbrace{\text{implicit}(\text{BN}, \text{Dropout}, \text{Mixup}, \text{SpecAug})}_{\text{stochastic regularizers}}
$$

The v2.1 design principle: with Mixup and graduated dropout both active, the explicit L2 penalty is reduced ($10^{-4}$ vs $5 \times 10^{-4}$ in v2) to avoid over-regularization.

---

## 8. Evaluation Metrics

### 8.1 Accuracy

$$
\text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\!\left[\hat{y}_i = y_i\right]
$$

where $\hat{y}_i = \arg\max_k \hat{\mathbf{y}}_i^{(k)}$.

### 8.2 Macro-F1 (Primary Metric)

For each class $k$:

$$
\text{Precision}_k = \frac{TP_k}{TP_k + FP_k}, \quad \text{Recall}_k = \frac{TP_k}{TP_k + FN_k}
$$

$$
F_{1,k} = 2 \cdot \frac{\text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}
$$

The macro-averaged F1 score (treating all classes equally regardless of support):

$$
\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^{K} F_{1,k}
$$

This is the **primary metric** for model checkpointing and early stopping, chosen because it penalizes poor performance on minority classes more than accuracy does.

### 8.3 Confusion Matrix

The confusion matrix $\mathbf{C} \in \mathbb{N}^{K \times K}$ where:

$$
C_{ij} = \#\{n : y_n = i \text{ and } \hat{y}_n = j\}
$$

Diagonal entries $C_{ii}$ are correct predictions; off-diagonal entries reveal systematic misclassification patterns (e.g., Rock↔Metal, Pop↔Hip-Hop confusion).

---

## 9. Hyperparameter Summary

| Category | Hyperparameter | Symbol / Key | Value |
|----------|---------------|-------------|-------|
| **Feature extraction** | Sample rate | $f_s$ | 22,050 Hz |
| | FFT size | $N_{\text{FFT}}$ | 512 |
| | Hop length | $H$ | 256 |
| | Mel bands | $M$ | 192 |
| | Clip duration | $d$ | 15 s |
| **Architecture** | Conv blocks | — | 5 |
| | Filter progression | — | 32 → 64 → 128 → 256 → 256 |
| | Bottleneck units | — | 256 |
| | Output classes | $K$ | 10 |
| **Optimization** | Max learning rate | $\eta_{\max}$ | $5 \times 10^{-4}$ |
| | Min learning rate | $\eta_{\min}$ | $10^{-6}$ |
| | Weight decay | $\lambda_{\text{wd}}$ | $10^{-4}$ |
| | Gradient clip norm | $c$ | 1.0 |
| | Warmup epochs | — | 5 |
| | Max epochs | — | 102 |
| | Batch size | $B$ | 32 |
| | Early stopping patience | — | 20 |
| **Augmentation** | Mixup alpha | $\alpha$ | 0.3 |
| | SpecAugment freq mask | $F$ | 24 |
| | SpecAugment time mask | $T_{\text{mask}}$ | 40 |
| | SpecAugment num masks | $N_{\text{masks}}$ | 1 |
| **Regularization** | Spatial dropout (blocks 2–5) | $p_{2..5}$ | 0.05, 0.10, 0.15, 0.20 |
| | Final dropout | $p_{\text{final}}$ | 0.20 |
| | Label smoothing | $\varepsilon$ | 0.0 (disabled) |
| | Max class weight | $w_{\max}$ | 1.5 |

---

## 10. Parameter Count Breakdown

All convolutions use `use_bias=False` (bias is subsumed by BatchNorm's learnable shift $\beta$).

| Layer | Computation | Trainable Parameters |
|-------|------------|---------------------|
| `conv1` (32, 5×5) | $1 \times 5 \times 5 \times 32$ | 800 |
| `bn1` | $\gamma_{32} + \beta_{32}$ | 64 |
| `conv2` (64, 3×3) | $32 \times 3 \times 3 \times 64$ | 18,432 |
| `bn2` | $\gamma_{64} + \beta_{64}$ | 128 |
| `conv3` (128, 3×3) | $64 \times 3 \times 3 \times 128$ | 73,728 |
| `bn3` | $\gamma_{128} + \beta_{128}$ | 256 |
| `conv4` (256, 3×3) | $128 \times 3 \times 3 \times 256$ | 294,912 |
| `bn4` | $\gamma_{256} + \beta_{256}$ | 512 |
| `conv5` (256, 3×3) | $256 \times 3 \times 3 \times 256$ | 589,824 |
| `bn5` | $\gamma_{256} + \beta_{256}$ | 512 |
| `fc_bottleneck` | $256 \times 256 + 256$ | 65,792 |
| `fc_out` | $256 \times 10 + 10$ | 2,570 |
| **Total** | | **~1,047,530** |

With ~13,300 training samples (10 genres × 1,900 × 0.7 split), the **params-per-sample ratio** is:

$$
\frac{1{,}047{,}530}{13{,}300} \approx 79 : 1
$$

---

### Data Configuration Reference

From `settings.json`:

| Setting | Value |
|---------|-------|
| Target genres | Hip-Hop, Pop, Folk, Rock, Metal, Electronic, Classical, Jazz, Country, Blues |
| Samples per genre | 1,900 |
| Additional contribution ratio | 0.5 |
| Train / (val + test) split | 70% / 30% |
| Sample length | 15 s |

---

## 11. Compact Mathematical Representations

### 11.1 Raw Audio → Log-Mel (one line)

The complete feature extraction from a raw waveform $x[n]$ to a normalised log-mel matrix $\hat{\mathbf{X}}$ is:

$$
\hat{\mathbf{X}}
=
\frac{
  \ln\!\left(1 + \mathbf{H}\,\left|\,\sum_{n} x[n+m H]\,w[n]\,e^{-j2\pi kn/N}\,\right|^2\right) - \boldsymbol{\mu}
}{
  \boldsymbol{\sigma}
}
\;\in\;\mathbb{R}^{192\times 1291}
$$

Reading left to right:

| Sub-expression | Operation | Parameters |
|---|---|---|
| $\displaystyle\sum_n x[n+mH]\,w[n]\,e^{-j2\pi kn/N}$ | STFT (Hann window) | $N=512,\;H=256$ |
| $\lvert\cdot\rvert^2$ | Power spectrogram | — |
| $\mathbf{H}\,(\cdot)$ | Mel filterbank projection | $M=192$ triangular filters |
| $\ln(1+\cdot)$ | Logarithmic compression (`log1p`) | — |
| $(\cdot - \boldsymbol{\mu})/\boldsymbol{\sigma}$ | Per-band z-score normalisation | computed over train set |

---

### 11.2 Inference (one line)

At inference time all stochastic operations (Mixup, SpecAugment, Dropout) are inactive. Given the pre-computed, normalised log-mel input $\hat{\mathbf{X}} \in \mathbb{R}^{192 \times 1291}$ (see §11.1), the pipeline to genre prediction collapses to:

$$
\hat{\mathbf{y}}
=
\operatorname{softmax}\!\Bigl(
W_{\mathrm{out}}\,
\phi\bigl(
W_{\mathrm{fc}}\,
\operatorname{GAP}\bigl(
B_5 \circ B_4 \circ B_3 \circ B_2 \circ B_1
[\hat{\mathbf{X}}]
\bigr)+b_{\mathrm{fc}}
\bigr)+b_{\mathrm{out}}
\Bigr)
$$

where

$$
B_1 = \operatorname{MaxPool}\circ\phi\circ\operatorname{BN}\circ\operatorname{Conv}_{5\times5,\,32},
\qquad
B_\ell = \operatorname{MaxPool}\circ\phi\circ\operatorname{BN}\circ\operatorname{Conv}_{3\times3,\,c_\ell},\quad \ell=2,\dots,5
$$

with filter counts $c_\ell \in \{64, 128, 256, 256\}$ and $\phi = \operatorname{ReLU}$.

---

### 11.3 Training Update (one line)

At each mini-batch step $t$, the parameter update starting from $\boldsymbol{\theta}_t$ is:

$$
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta_t\left(
\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}
+
\lambda_{\mathrm{wd}}\,\boldsymbol{\theta}_t
\right),
\quad
\mathbf{g}_t = \operatorname{clip}_{\|\cdot\|_2\le 1}\!\left(
\nabla_{\boldsymbol{\theta}}
\frac{1}{B}\sum_{i=1}^{B}
w_i\,
\mathrm{CE}\!\left(
\tilde{\mathbf{y}}_i,\;
f_{\boldsymbol{\theta}}\!\left(\operatorname{SpecAug}(\operatorname{Mixup}(\hat{\mathbf{X}}_i))\right)
\right)
\right)
$$

where $\hat{\mathbf{m}}_t, \hat{\mathbf{v}}_t$ are the bias-corrected Adam moment estimates of $\mathbf{g}_t$, $\eta_t$ follows the cosine-with-warmup schedule, $\tilde{\mathbf{y}}_i$ is the Mixup-blended soft label, $w_i = \mathbf{c}^\top\mathbf{y}_i$ is the pre-Mixup anchor class weight, and $\lambda_{\mathrm{wd}} = 10^{-4}$.

---

### 11.4 Side-by-Side Comparison

| Aspect | Inference | Training Update |
|--------|-----------|-----------------|
| Input transform | $\ln(1+\mathbf{H}\|\mathrm{STFT}(x)\|^2)$ then z-score | same, plus Mixup $\tilde{\mathbf{X}}$ + SpecAugment |
| Forward pass $f_\theta$ | $B_1\!\circ\!\cdots\!\circ\!B_5 \to \operatorname{GAP} \to \phi(W_{\mathrm{fc}}\cdot) \to \operatorname{softmax}$ | same, but BN uses batch stats and Dropout is active |
| Target | — (predict $\arg\max\hat{\mathbf{y}}$) | soft label $\tilde{\mathbf{y}} = \lambda\mathbf{y}_i+(1-\lambda)\mathbf{y}_j$ |
| Loss | — | $\frac{1}{B}\sum_i w_i\,\mathrm{CE}(\tilde{\mathbf{y}}_i,\hat{\mathbf{y}}_i)$ |
| Gradient | — | clipped $\operatorname{clip}_{\|\cdot\|\le 1}(\nabla_\theta\mathcal{L})$ |
| Weight update | — | AdamW + decoupled L2 ($\lambda_{\mathrm{wd}}=10^{-4}$) |
| Learning rate | — | $\eta_t$: linear warmup (5 ep) → cosine decay to $10^{-6}$ |
