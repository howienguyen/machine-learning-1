# Machine Learning 1 (MLE501.22), MSA30DN, FSB | Nguyen Sy Hung | Mar 2026

# Deep Learning for Music Genre Prediction Using Log-Mel Spectrograms and CNNs

**Nguyễn Sỹ Hùng (25MSA33055)**

**Machine Learning 1 (MLE501.22), FSB – Lecturer: PhD. Nguyen Thi Kim Truc**

## Abstract

This report presents the final project for the Machine Learning 1 course. The project compares two audio feature representations for music genre classification — MFCC (Mel-Frequency Cepstral Coefficients) and log-Mel spectrograms — both fed into Convolutional Neural Networks (CNNs).

Starting from a provided MFCC + CNN baseline trained on the small FMA (Free Music Archive) dataset, the project builds a log-Mel + CNN baseline and then iterates toward a production-like model. The final model, LogMel CNN v2.4.1, is a five-block CNN with approximately one million parameters trained on a custom 18,728-sample dataset covering ten genres, using TFRecord-based input, Mixup augmentation, SpecAugment, graduated spatial dropout, and cosine-annealed AdamW optimization. It achieves a test macro-F1 of 0.847 and test accuracy of 84.8%, a substantial improvement over the initial baselines. The end-to-end pipeline — from raw audio collection through preprocessing, training, and inference — is demonstrated via a real-time music genre prediction web application.

**Keywords.** Music genre classification, log-Mel spectrogram, MFCC, CNN, deep learning, TFRecord, data augmentation, Mixup, SpecAugment.

## Introduction

Music genre classification is the task of automatically assigning a genre label to a piece of music. It is a well-studied problem in music information retrieval (MIR), but it remains challenging because genres are not strictly defined by physics or mathematics alone. While music has measurable properties — rhythm, tempo, pitch, spectral energy distribution — genre boundaries are also shaped by culture, history, artist intent, and listener perception. A single song can reasonably belong to more than one genre, and different listeners may disagree on the correct label.

Despite this ambiguity, automatic genre classification is useful in practice. It supports music recommendation, playlist generation, library organization, and content tagging at scale. The task is well suited for supervised deep learning because labeled datasets are available and the audio signal contains learnable patterns that correlate with genre.

This project is a course assignment, not a research publication. Its purpose is to apply course knowledge and practical self-study to a realistic machine learning problem, working through the full cycle from data collection to a functioning demo application.

## Project Overview

This project treats music genre classification as a practical supervised learning problem. The goal is to train a deep learning model to predict the dominant genre of a 15-second music segment. To do this, the audio is converted into log-Mel spectrograms, which are compact time-frequency representations of sound. These spectrograms are then used as input to a Convolutional Neural Network (CNN), which learns patterns related to genre from labeled training data.

The project is structured around three main phases:

1. **Baseline comparison.** A provided MFCC + CNN baseline (from the FMA repository) is re-engineered and compared against a newly defined log-Mel + CNN baseline. This comparison isolates the effect of the input feature representation on classification performance.

2. **Iterative model improvement.** The log-Mel baseline is incrementally improved through architectural refinements, better regularization, a larger and more balanced custom dataset, and a more robust training pipeline. The result is the LogMel CNN v2.4.1 model.

3. **Demo application.** A real-time inference web service and a system-audio capture client demonstrate the trained model on live or playing music.

This project is mainly for learning and practical exploration. It is not intended to produce a publishable research result or a fully commercial system. Instead, it focuses on applying course knowledge, instructor guidance, and self-study to a realistic problem, while improving understanding of data preparation, model design, training, evaluation, and result analysis.

## Problem Statement

The main problem in this project is to classify a short music segment into one of ten genres using deep learning. More specifically, the system takes a 15-second audio clip, converts it to a log-Mel spectrogram, and predicts the dominant genre label.

This is hard for several reasons. Music is a complex, high-dimensional signal. Genre labels are culturally defined and often overlap — a song labeled "Blues" may share characteristics with "Rock" or "Jazz." Short audio clips may not always contain enough distinctive information for confident classification. And practical constraints (limited compute, limited data) make it impossible to simply scale the model to arbitrary size.

The model does not try to define the true meaning of music genre. Instead, it learns from a labeled dataset and makes the best possible prediction of the dominant genre. A key question behind this project is whether a compact CNN model, trained on personal computing hardware, can still deliver useful genre predictions. This question drives the choices about model size, architecture, training setup, and dataset construction throughout the project.

## Goals and Objectives

The main goals of this project are:

- Prepare the dataset, preprocessing pipeline, and exploratory data analysis (EDA)
- Re-engineer the provided MFCC + CNN baseline for a better speed performance (the original training script for the MFCC + CNN baseline runs quite slow)
- Build a Log-Mel + CNN baseline with a similar setup for fair comparison
- Build a more practical, production-like Log-Mel + CNN version
- Compare and benchmark the different model versions
- Evaluate the results and analyze strengths and weaknesses
- Develop a demo application
- Explore whether a compact model can still be useful in practice

Overall, the project moves step by step from a basic instructional baseline toward a stronger proof-of-concept model.

## Scope and Constraints

This project is limited by time, effort, hardware, and computing resources. It is a final course assignment, so its purpose is to show how the learned concepts can be applied in practice, not to build a full industrial product.

The project assumes a home-lab setup with a normal consumer GPU (NVIDIA GeForce RTX 3090), not a large cloud or multi-GPU system. Training time should stay within a few hours, or at most around 24 hours, not multiple days. Because of this, the model must stay reasonably small and efficient.

Also, this is not a safety-critical system. The model is expected to classify at least eight common music genres, and occasional mistakes are acceptable because they do not cause serious harm. In practice, the model is meant to be a supportive tool for tasks such as tagging, browsing, or lightweight recommendation support, rather than a perfect decision-making system.

## Methodology and Experimental Setup

This project follows a step-by-step experimental approach. It begins with dataset preparation, preprocessing, and exploratory data analysis (EDA). Audio clips are converted into fixed-length 15-second segments and represented as log-Mel spectrograms for CNN-based training. For comparison, the project first uses the provided MFCC + CNN baseline trained on the small FMA dataset. Based on this reference, a new log-Mel + CNN baseline is defined to remain as close as possible to the original model, with only minor adjustments required by the different input representation. This enables a more controlled comparison of how MFCC and log-Mel features affect model performance.

Model development then proceeds incrementally. After re-engineering and understanding the baseline, the project extends the work by building a custom dataset from several open datasets together with manually collected and processed audio clips. The model is then further improved into a more production-like version through practical refinements in preprocessing, architecture, and training design. This version is finally demonstrated through a proof-of-concept application. The models are trained, benchmarked, and evaluated under realistic personal computing constraints, and the results are analyzed to assess both their practical usefulness and their limitations.

The hardware and software environment used throughout the project is summarized below:

| Component | Details |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 (24 GB VRAM) |
| OS | WSL2 on Windows (Linux 6.6.87 kernel) |
| Python | 3.12.3 |
| TensorFlow | 2.20.0 |
| Precision | float32 |
| Earlier runs (v1, MFCC) | Intel XPU, TensorFlow 2.15.1, Python 3.11 |

## Collecting Data, Building a Custom Dataset, and EDA

### Data Sources

The project uses audio from three main sources:

1. **FMA (Free Music Archive), medium subset.** A widely used open dataset for music analysis research. The medium subset contains around 25,000 tracks across 16 genres, each 30 seconds long, distributed as MP3 files with accompanying metadata CSVs. The FMA dataset provides pre-defined train, validation, and test splits.

2. **MTG-Jamendo Dataset.** A large Creative Commons licensed music collection from the Jamendo platform, containing over 55,000 full-length tracks with 195 tags across genre, instrument, and mood categories. For this project, a subset of Jamendo tracks was selected and downloaded using an automated script that filters for tracks with at most two genre tags. This keeps the selection focused on tracks with a clear primary genre. Tracks were downloaded as MP3 and organized by genre.

3. **Manually collected audio tracks.** Additional tracks were gathered from personal playlists and public sources for genres that were underrepresented in the FMA and Jamendo data. These were organized into genre folders under the manual collection directory.

Two of the ten genres — **Bolero** and **Speech** — are not present in FMA and were added entirely from the Jamendo and manually collected sources to broaden the model's practical coverage.

### Preprocessing Pipeline

The dataset construction follows a three-stage pipeline:

**Stage 1 — Manifest Building.** The first script discovers audio files from all sources, collects metadata, and generates fixed-length 15-second segments. Each segment receives a deterministic sample ID based on a BLAKE2b hash of its source identity, making the process reproducible. All segments from the same source audio file are kept in the same train/val/test split to prevent data leakage. The output is a set of Parquet manifest files.

**Stage 2 — Log-Mel Feature Extraction.** The second script reads the manifest and converts each 15-second audio segment into a log-Mel spectrogram. The feature extraction parameters are:

| Parameter | Value |
|---|---|
| Sample rate | 22,050 Hz |
| Number of mel bands | 192 |
| FFT window size (n_fft) | 512 |
| Hop length | 256 |
| Clip duration | 15.0 seconds |
| Output shape | (192, 1291) |

Each spectrogram is saved as a NumPy `.npy` file, grouped by split.

**Stage 3 — TFRecord Conversion.** The third script packages the `.npy` files into split-sharded TFRecords for efficient TensorFlow consumption. The default shard size is 1,024 records per file. The conversion also writes metadata files (config JSON, shard manifests) so that training scripts can discover the data without hardcoded paths.

### Dataset Summary

The final production dataset (used by the v2.4.1 model) contains:

| Split | Samples |
|---|---|
| Train | 13,107 |
| Validation | 2,811 |
| Test | 2,810 |
| **Total** | **18,728** |

The ten genre classes and their training set counts are:

| Genre | Train Samples |
|---|---|
| Blues | 1,367 |
| Bolero | 1,365 |
| Classical | 1,351 |
| Country | 1,206 |
| Hip-Hop | 1,364 |
| Jazz | 1,324 |
| Metal | 1,360 |
| Pop | 1,307 |
| Rock | 1,349 |
| Speech | 1,114 |

The dataset is roughly balanced, with a maximum-to-minimum class count ratio of about 1.23 (Blues at 1,367 vs Speech at 1,114). This moderate imbalance is handled during training through adaptive class weighting rather than forced resampling.

### EDA Observations

Exploratory data analysis revealed several important points:

- **Class balance is reasonable but not perfect.** The imbalance ratio is 1.23, well within the range where class-weighted loss is effective without needing aggressive resampling.
- **Genre confusion patterns exist.** Genres that share acoustic characteristics — such as Blues, Country, Rock, and Pop — tend to confuse the model. This is expected because these genres often feature similar instrumentation (acoustic/electric guitar, drums, vocals).
- **Bolero and Speech are distinctive.** These two genres have clearly different spectral patterns from Western popular music and are classified with high accuracy by all model versions.
- **Short clips have limited information.** A 15-second segment may not always contain the most genre-distinctive part of a song, which sets a ceiling on classification accuracy for ambiguous cases.

## Baselines: MFCC vs Log-Mel

### Model Architectures and Training Setup

#### MFCC + CNN Baseline

The MFCC baseline is a faithful reproduction of the "ConvNet on MFCC" architecture from the FMA repository baselines. It is a three-layer convolutional network designed for MFCC input:

| Layer | Details |
|---|---|
| Input | (13, 2582, 1) — 13 MFCCs x 2582 time frames from 30-second clips |
| Conv2D #1 | 3 filters, kernel (13, 10), stride (1, 4), ReLU |
| Conv2D #2 | 15 filters, kernel (1, 10), stride (1, 4), ReLU |
| Conv2D #3 | 65 filters, kernel (1, 10), stride (1, 4), ReLU |
| Flatten | 2,470 units |
| Dense (output) | n_classes, softmax |
| **Total parameters** | **30,441** |

The first convolutional layer uses a kernel that spans the entire frequency axis (13 MFCC coefficients), which is appropriate for MFCC because the DCT already compresses spectral information into a small number of decorrelated coefficients.

**Training setup:**

| Parameter | Value |
|---|---|
| Optimizer | SGD |
| Learning rate | 1e-3 |
| Loss | Categorical crossentropy |
| Epochs | 16-32 |
| Batch size | 16 |
| Dataset | FMA small (8 genres) |

#### Log-Mel CNN Baseline (v1)

The first log-Mel baseline was designed to match the MFCC baseline as closely as possible while adapting for the larger and more structured log-Mel input. An important architectural lesson emerged early: naively copying the MFCC architecture by using a convolution kernel that spans the full frequency axis produced worse results with log-Mel input. This is because MFCC input is already compressed by the DCT, so a full-axis kernel is natural for it. But log-Mel spectrograms are local time-frequency representations — a kernel spanning all 192 mel bands forces global spectral template matching in a single layer, which bypasses the hierarchical feature learning that CNNs are designed for.

The solution was to use small local kernels (5x5 for the first layer, 3x3 for subsequent layers) and rely on stacked convolution + pooling blocks to build up receptive field progressively. The resulting v1 architecture is a five-block CNN:

| Block | Filters | Kernel | Other |
|---|---|---|---|
| conv1 | 32 | 5x5 | BatchNorm, ReLU, MaxPool 2x2 |
| conv2 | 64 | 3x3 | BatchNorm, ReLU, MaxPool 2x2 |
| conv3 | 128 | 3x3 | BatchNorm, ReLU, MaxPool 2x2 |
| conv4 | 256 | 3x3 | BatchNorm, ReLU, MaxPool 2x2 |
| conv5 | 256 | 3x3 | BatchNorm, ReLU, MaxPool 2x2 |
| Head | — | — | GlobalAveragePooling2D, Dense(10, softmax) |
| **Total parameters** | **~983,000** | | |

**Feature configuration:** 192 mel bands x 861 time frames from 10-second clips at 22,050 Hz.

**Training setup:**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR schedule | Cosine annealing, warmup 3 epochs |
| LR max | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 99 |
| Actual epochs | 42 (early stopped) |
| SpecAugment | freq_mask=15, time_mask=25, num_masks=2 |
| Label smoothing | 0.02 |
| Dataset | FMA medium (10 genres), 10-second clips |
| Train/Val/Test | 9,818 / 1,238 / 1,237 |
| Hardware | Intel XPU, TF 2.15.1 |
| Training time | ~166 minutes |

### Benchmarking and Evaluation — Comparison Results

#### Overall Metrics

| Model | Dataset | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| MFCC + CNN (FMA tiny, 8 genres) | 2,951 train / 448 test | 29.24% | 0.259 |
| Log-Mel CNN v1 (FMA medium, 10 genres) | 9,818 train / 1,237 test | 58.12% | 0.578 |

Note: The MFCC baseline was only run on the FMA tiny subset (8 genres, ~3,800 total samples), which is a much smaller and harder dataset. The log-Mel v1 model was trained on the larger FMA medium split (10 genres, ~12,300 total samples) with 10-second clips. A direct numeric comparison is therefore not fully fair, but the results illustrate the overall direction.

#### Log-Mel CNN v1 — Detailed Results

| Metric | Train | Validation | Test |
|---|---|---|---|
| Accuracy | 75.17% | 62.60% | 58.12% |
| Macro-F1 | 0.753 | 0.628 | 0.578 |

**Per-genre test F1 scores (v1):**

| Genre | Test F1 |
|---|---|
| Classical | 0.82 |
| Hip-Hop | 0.77 |
| Metal | 0.63 |
| Jazz | 0.59 |
| Electronic | 0.58 |
| Blues | 0.56 |
| Country | 0.55 |
| Folk | 0.54 |
| Rock | 0.44 |
| Pop | 0.30 |

#### Key Observations from the Baseline Comparison

1. **The MFCC baseline performed poorly** even on its own small dataset. With only 30K parameters and a basic SGD optimizer, it struggled to learn enough for reliable classification.

2. **The log-Mel baseline did substantially better**, benefiting from a larger dataset, a deeper architecture with BatchNorm, AdamW with cosine annealing, and SpecAugment. However, it still showed significant overfitting — a 17.5-point F1 gap between train (0.753) and test (0.578).

3. **Genre confusion followed predictable patterns.** Genres with similar instrumentation (Blues, Country, Rock, Pop, Folk) formed a confusion cluster. Metal had high recall (0.97) but low precision (0.47) because its uncapped class weight of 2.13 caused over-prediction.

4. **The architecture must match the input representation.** The most important architectural lesson was about kernel design: a full-frequency-axis kernel works for MFCC (which is already compressed by DCT) but is harmful for log-Mel (which needs local, hierarchical convolution). This insight guided all subsequent model versions.

## Production-like Neural Network Model and Music Genre Prediction Application

After the baseline comparison, the project focused on iteratively improving the log-Mel CNN toward a practical, production-quality model. This involved several rounds of refinement across the v2.x model line (v2.0 through v2.4.1), addressing overfitting, regularization, dataset expansion, normalization, training stability, and input pipeline efficiency. The final validated model is **LogMel CNN v2.4.1**.

### Model Architecture

The v2.4.1 model retains the five-block CNN backbone from v1 but adds graduated spatial dropout, a 256-unit bottleneck, and several regularization improvements. The full architecture has **1,049,002 parameters** (approximately 4 MB):

| Layer | Type | Output Shape | Parameters |
|---|---|---|---|
| logmel | InputLayer | (None, 192, 1291, 1) | 0 |
| conv1 | Conv2D, 32 filters, 5x5 | (None, 192, 1291, 32) | 800 |
| bn1 | BatchNormalization | — | 128 |
| relu1 | ReLU | — | 0 |
| pool1 | MaxPooling2D, 2x2 | (None, 96, 645, 32) | 0 |
| conv2 | Conv2D, 64 filters, 3x3 | (None, 96, 645, 64) | 18,432 |
| bn2 | BatchNormalization | — | 256 |
| relu2 | ReLU | — | 0 |
| sdrop2 | SpatialDropout2D (0.05) | — | 0 |
| pool2 | MaxPooling2D, 2x2 | (None, 48, 322, 64) | 0 |
| conv3 | Conv2D, 128 filters, 3x3 | (None, 48, 322, 128) | 73,728 |
| bn3 | BatchNormalization | — | 512 |
| relu3 | ReLU | — | 0 |
| sdrop3 | SpatialDropout2D (0.10) | — | 0 |
| pool3 | MaxPooling2D, 2x2 | (None, 24, 161, 128) | 0 |
| conv4 | Conv2D, 256 filters, 3x3 | (None, 24, 161, 256) | 294,912 |
| bn4 | BatchNormalization | — | 1,024 |
| relu4 | ReLU | — | 0 |
| sdrop4 | SpatialDropout2D (0.15) | — | 0 |
| pool4 | MaxPooling2D, 2x2 | (None, 12, 80, 256) | 0 |
| conv5 | Conv2D, 256 filters, 3x3 | (None, 12, 80, 256) | 589,824 |
| bn5 | BatchNormalization | — | 1,024 |
| relu5 | ReLU | — | 0 |
| sdrop5 | SpatialDropout2D (0.20) | — | 0 |
| pool5 | MaxPooling2D, 2x2 | (None, 6, 40, 256) | 0 |
| gap | GlobalAveragePooling2D | (None, 256) | 0 |
| fc_bottleneck | Dense, 256 units, ReLU | (None, 256) | 65,792 |
| dropout | Dropout (0.20) | (None, 256) | 0 |
| fc_out | Dense, 10 units, softmax | (None, 10) | 2,570 |

**Key architectural and training choices in the v2.x line:**

- **Graduated spatial dropout** (0.05 to 0.20 across blocks 2-5): applies lighter regularization in early layers where features are more general, and stronger regularization in deeper layers where overfitting is more likely.
- **Wider 256-unit bottleneck** (up from 128 in v1): preserves the full feature space from the backbone before the classifier head.
- **Mixup augmentation** (alpha = 0.3): blends pairs of training samples and their labels to smooth the decision boundary and reduce overfitting.
- **SpecAugment** (freq_mask=24, time_mask=40, 1 mask each): randomly masks frequency and time bands in the spectrogram during training.
- **Pre-Mixup anchor-based class weighting**: computes per-sample loss weights from the original (pre-blend) class label, preserving the class-balance signal that would otherwise be diluted by Mixup.
- **Adaptive class weights** capped at 1.5: prevents any single class from dominating gradient updates.
- **Label smoothing** (0.05): softens the one-hot target distribution slightly.
- **Gradient clipping** (clipnorm = 1.0): stabilizes training when combined with class-weighted loss and Mixup.
- **AdamW optimizer** with cosine-annealed learning rate (max 5e-4, 5-epoch warmup) and weight decay 1e-4.
- **Train-only per-mel-bin standardization**: the mean and standard deviation of each mel band are computed from the training split only (two streaming passes), then applied to all splits. This prevents data leakage and produces saved normalization artifacts for inference.
- **TFRecord input pipeline**: training data is consumed from split-sharded TFRecords with fixed map/read parallelism and autotuned prefetch, rather than per-sample NumPy file loads. This reduced per-epoch time significantly.
- **Early stopping** on validation macro-F1 with patience of 9 epochs and best-weight restoration.

### Results and Evaluation

The final validated run trained for 107 of 136 maximum epochs with batch size 48, stopping early when validation macro-F1 plateaued.

#### Overall Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| Accuracy | 95.22% | 83.99% | **84.84%** |
| Macro-F1 | 0.953 | 0.841 | **0.847** |

#### Per-Genre Test Performance

| Genre | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Bolero | 0.990 | 0.983 | 0.986 | 293 |
| Speech | 0.963 | 1.000 | 0.981 | 237 |
| Metal | 0.969 | 0.956 | 0.963 | 296 |
| Hip-Hop | 0.901 | 0.895 | 0.898 | 295 |
| Classical | 0.799 | 0.986 | 0.883 | 291 |
| Jazz | 0.849 | 0.751 | 0.797 | 285 |
| Rock | 0.766 | 0.747 | 0.757 | 285 |
| Blues | 0.780 | 0.735 | 0.757 | 294 |
| Country | 0.721 | 0.786 | 0.752 | 257 |
| Pop | 0.744 | 0.650 | 0.694 | 277 |

#### Progress Across Model Versions

| Model Version | Dataset | Test Accuracy | Test Macro-F1 |
|---|---|---|---|
| MFCC + CNN baseline | FMA tiny (8 genres, 3.8K) | 29.24% | 0.259 |
| Log-Mel CNN v1 | FMA medium (10 genres, 12.3K) | 58.12% | 0.578 |
| Log-Mel CNN v2.4.1 | Custom (10 genres, 18.7K) | **84.84%** | **0.847** |

The v2.4.1 model improved test macro-F1 by **26.9 points** over the v1 baseline, achieved through a combination of a larger and more balanced dataset, stronger and more targeted regularization, architectural refinements, and a more efficient training pipeline.

### Demo Applications (Consumer Layer)

Two demo applications consume the trained model:

#### Inference Web Service

A FastAPI-based inference server loads the best v2.4.1 checkpoint and serves genre predictions. It supports:

- **REST endpoint** (`/predict_json`): accepts an audio file upload and returns genre predictions with confidence scores.
- **WebSocket streaming** (`/ws/stream`): accepts chunked PCM audio data in real time and returns predictions as each 15-second window completes.
- **Health and model info endpoints** (`/health`, `/model`).
- Single-crop and three-crop inference modes (three-crop averages predictions from the start, middle, and end of a clip for more robust results).

#### System Audio Capture Client

A Flask-based desktop client captures system audio output in real time using Windows WASAPI loopback recording. It:

- Captures audio at the system's native sample rate (48,000, 44,100, or 22,050 Hz) and converts it to 22,050 Hz mono.
- Maintains a rolling 15-second window of audio.
- Sends the current window to the inference web service for prediction.
- Displays the predicted genre and confidence scores in a local browser UI.

This allows a user to play any music on their computer and see the model's real-time genre prediction.

## Discussion and Reflection

### What Worked Well

1. **The iterative approach paid off.** Starting from a simple baseline and making targeted improvements at each step produced a clear progression from 0.578 to 0.847 test macro-F1. Each improvement was motivated by analysis of the previous version's failure modes.

2. **Regularization made a large difference.** The combination of Mixup, SpecAugment, graduated spatial dropout, and class weight capping together reduced overfitting substantially. The train-test F1 gap shrank from 17.5 points (v1) to about 10.6 points (v2.4.1), even with a more capable model trained for more epochs.

3. **Dataset expansion and balancing helped.** Supplementing FMA with Jamendo and manually collected tracks — especially for Bolero and Speech — gave the model more diverse examples and broadened its practical coverage.

4. **The TFRecord pipeline improved training throughput.** Switching from per-sample `.npy` file reads to split-sharded TFRecords reduced per-epoch time significantly, allowing more experiments in less wall-clock time.

5. **The architectural kernel-design insight was foundational.** Recognizing early that log-Mel spectrograms need local kernels (not full-frequency-axis kernels) prevented a fundamental design mistake from propagating through all subsequent model versions.

### Limitations and Areas for Future Work

1. **Pop remains the weakest genre** (F1 = 0.694). Pop music is stylistically diverse and overlaps heavily with Rock, Country, and Hip-Hop. A 15-second window may not always capture enough distinctive information.

2. **The Blues-Rock-Country-Jazz confusion cluster persists.** These genres share instrumentation and harmonic vocabulary. More diverse training data or genre-specific augmentation strategies might help.

3. **The model uses only a single fixed-length window.** Multi-scale or attention-based approaches that look at multiple temporal resolutions could capture longer-range structure.

4. **The dataset is still relatively small** at 18,728 samples. Larger datasets with more diverse sources would likely improve generalization.

5. **The current approach uses a single dominant-genre label.** Real music is often multi-genre. A multi-label formulation might be more realistic, though it requires different evaluation methodology.

6. **No systematic hyperparameter search was performed.** The improvements were driven by manual analysis and iterative adjustment. Automated search (e.g., Bayesian optimization) might find better configurations.

## Conclusion

This project demonstrated that a compact CNN model (~1M parameters) trained on personal computing hardware can achieve useful music genre classification performance. Starting from a provided MFCC + CNN baseline that scored 0.259 test macro-F1, the project iterated through a log-Mel + CNN baseline (0.578) to a production-like v2.4.1 model that reached **0.847 test macro-F1 across 10 genres**.

The key lessons from this project are:

1. **Input representation matters.** Log-Mel spectrograms contain more information than MFCCs, but the model architecture must be designed to exploit the local time-frequency structure of log-Mel inputs. Naively copying an MFCC-appropriate architecture produces worse results.

2. **Regularization is at least as important as architecture.** Mixup, SpecAugment, graduated dropout, class weight capping, and label smoothing together accounted for a large share of the improvement from v1 to v2.4.1.

3. **Data quality and balance matter more than raw quantity.** Expanding the dataset with targeted genre-specific collection (especially for Bolero and Speech) and maintaining reasonable class balance through adaptive weighting had a direct positive effect on performance.

4. **A practical end-to-end pipeline is achievable.** From raw audio collection through preprocessing, training, and real-time inference, the full system runs on a single consumer GPU and can produce live genre predictions on playing music.

The model is not perfect — Pop, Rock, Blues, and Country remain challenging, and the single-label formulation is a simplification of how genres actually work. But for a course project, the result shows that careful engineering of the full pipeline can produce a genuinely useful system even with limited resources.

## References

1. Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). *FMA: A Dataset for Music Analysis.* Proceedings of the 18th International Society for Music Information Retrieval Conference (ISMIR).

2. Bogdanov, D., Won, M., Tovstogan, P., Porter, A., & Serra, X. (2019). *The MTG-Jamendo Dataset for Automatic Music Tagging.* ICML 2019 Machine Learning for Music Discovery Workshop.

3. Li, T., Ogihara, M., & Li, Q. (2010). *A Comparative Study on Content-Based Music Genre Classification.* Proceedings of the International MultiConference of Engineers and Computer Scientists (IMECS).

4. Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Proceedings of Interspeech.

5. Zhang, H., Cisse, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). *Mixup: Beyond Empirical Risk Minimization.* Proceedings of the International Conference on Learning Representations (ICLR).

6. Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization.* Proceedings of the International Conference on Learning Representations (ICLR).

7. TensorFlow documentation: *tf.data: Build TensorFlow input pipelines.* https://www.tensorflow.org/guide/data