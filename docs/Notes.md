Thu Mar  5 15:32:25 UTC 2026

In a near future, we will focus on design the training and another the model to solve the same problem later and this time will focus on true quality - not only for comparation with the baselines. Therefore maybe we will look at better solutions for optimizer, 30s fix-length, and etc... even another approaches, methodologies, more complex model architecture, and etc...

Awesome—that’s the right evolution: **phase 1 = fair baselines**, **phase 2 = chase real quality**. Once you stop treating “consistency” as sacred, you can optimize the whole system like an engineer instead of a scientist doing a controlled ablation.

Here’s a high-signal roadmap for “true quality” on FMA-style genre classification, with the why behind each lever.

## 1) Training upgrades that almost always help

### Optimizer + schedule (the boring magic)

Baselines with plain SGD(1e-3) are basically training with ankle weights.

Common “quality defaults”:

* **AdamW** (Adam + weight decay) with a **learning-rate schedule**
  Why: fast convergence + better generalization than plain Adam in many vision/audio CNNs.
* **SGD + momentum (0.9)** + cosine decay
  Why: often strong final generalization for CNNs.

Add-ons that matter:

* **Warmup** (first 1–5% of steps): prevents early instability.
* **Early stopping on macro-F1** (not just val loss): aligns stopping with your goal under imbalance.

### Regularization (make overfitting expensive)

* **Weight decay (L2)**: cheap and effective.
* **Dropout** (careful: usually after dense layers or late conv blocks).
* **SpecAugment** (for log-mel): time masking + frequency masking.
  Why: huge boost in robustness without changing labels.

### Class imbalance handling (quality, not cosmetics)

* Keep the natural distribution, but train smart:

  * **class weights** or
  * **balanced sampling** (e.g., oversample rare genres per batch)
* Always track **macro-F1** and per-class metrics.

## 2) Fixed 30s is convenient—not always optimal

If your real-world use is “10s clip classification,” training on rigid 30s can be misaligned. Better strategies:

### Random crop policy (simple and strong)

* Train on **random 10–15s crops** from each track (new crop each epoch).
* Evaluate on:

  * single crop (fast) **and/or**
  * multi-crop voting (higher accuracy, slower)

Why it helps:

* Acts like augmentation.
* Reduces reliance on “one lucky segment.”
* Lets you learn genre cues that appear intermittently.

### Variable-length models (more modern)

Instead of forcing fixed frames:

* Use architectures that can pool over time:

  * **Global average pooling over time**
  * **Temporal attention pooling**
  * **RNN/Transformer pooling** (if you go there)

Why: you stop teaching the model that “padding means something.”

## 3) Model architecture upgrades (still practical)

### Strong “next step” from your baseline CNN

* Add **BatchNorm + ReLU**
* Add **MaxPool** or strided conv blocks
* Use **GlobalAveragePooling2D** instead of Flatten (less overfit)
* Optional: **Residual blocks** (small ResNet)

This often gives a bigger jump than swapping MFCC vs log-mel.

### Go beyond CNN: audio-native approaches

If you want to level up:

* **CRNN** (CNN front-end + GRU/LSTM): good for capturing longer rhythmic structure.
* **Audio Transformers** on log-mel patches: strong but heavier.
* **Pretrained embeddings** (high leverage):

  * VGGish / YAMNet-ish style embeddings
  * OpenL3 / PANNs / HTSAT / AST-style features (depends what you can run)
  * Then train a small classifier on top

Why pretrained helps: genre classification benefits a lot from representations learned on huge audio corpora.

## 4) Methodology: how to improve without fooling yourself

When you start tuning for quality, the risk is “validation overfitting.” Use:

* A true **test set** you touch once at the end.
* A **validation set** for tuning.
* Prefer **macro-F1** as the primary target metric (accuracy as secondary).
* Track training curves and per-genre confusion to understand *what changed*.

## 5) A sane progression plan (minimal chaos)

1. Keep log-mel input (often best starting point)
2. Add **SpecAugment + AdamW + LR schedule**
3. Switch to **random 10–15s crops** (train) + multi-crop eval
4. Upgrade model to a small **ResNet-ish** CNN with GAP
5. If needed: pretrained audio embeddings or transformer

That sequence typically yields consistent gains and teaches you a lot about what actually matters.

When you’re ready to pivot into “quality mode,” the fun part is you’ll start seeing *why* genres are confused (e.g., Electronic vs Experimental) and can decide whether to solve it with data, augmentation, architecture, or label hierarchy—not just “add more layers and pray.”

Important: Maybe this one 3) Model architecture upgrades (still practical) can be done as the final phase after all others as mentioned above.


---
---

https://www.alphaxiv.org/overview/1612.01840v3

## Research Paper Report: FMA: A Dataset for Music Analysis

This report provides a detailed analysis of the research paper "FMA: A Dataset for Music Analysis" by Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, and Xavier Bresson. The paper introduces a significant new dataset aimed at advancing research in Music Information Retrieval (MIR).

### 1. Authors, Institution(s), and Research Group Context

The primary authors of this paper are Michaël Defferrard, Kirell Benzi, and Pierre Vandergheynst, all affiliated with **LTS2, EPFL (École Polytechnique Fédérale de Lausanne), Switzerland**. Xavier Bresson is affiliated with **SCSE, NTU (Nanyang Technological University), Singapore**, noting that his work was conducted while he was at EPFL.

EPFL is a world-renowned engineering and science institute with a strong reputation in areas such as signal processing, machine learning, and computer science. The LTS2 (Laboratory for Non-linear Systems) at EPFL, led by Professor Pierre Vandergheynst, is particularly recognized for its work in graph signal processing, spectral methods, and machine learning, with applications ranging from image processing to audio analysis. This context is crucial, as the creation of a large-scale, well-structured dataset like FMA aligns well with a group focused on developing robust machine learning models that require significant and diverse data for training and validation. Their expertise in signal processing would have been invaluable in ensuring the quality and consistency of the audio data and derived features. Xavier Bresson's involvement, even if his primary affiliation shifted, suggests a collaboration stemming from his time at EPFL, likely contributing expertise in mathematical modeling or data science, given NTU's strong engineering programs. The collaborative nature of creating such a comprehensive resource underscores the multi-disciplinary effort often required in large-scale data science projects.

### 2. Broader Research Landscape

The field of Music Information Retrieval (MIR) is dedicated to developing computational methods for browsing, searching, organizing, and analyzing large music collections. It encompasses a wide array of tasks, including music genre recognition (MGR), artist identification, mood classification, and automatic tagging. A fundamental challenge hindering progress in MIR, particularly with the rise of data-hungry machine learning paradigms like deep learning, has been the scarcity of large, open, and high-quality benchmark datasets.

The authors explicitly draw a parallel to the field of computer vision (CV), where the emergence of large-scale datasets such as MNIST, CIFAR, and most notably ImageNet, catalyzed a revolution in deep learning research. ImageNet, with its 1.3 million images, facilitated the breakthrough of deep convolutional neural networks in the ILSVRC2012 challenge, leading to unprecedented advancements in CV. The MIR community, however, has historically struggled with this issue due to restrictive music copyrights and the sheer scale of audio data required.

Previous MIR datasets, as summarized in Table 1 of the paper, generally suffer from several limitations:
*   **Small Scale:** Many early datasets like GTZAN (1,000 clips) are tiny, leading to overfitting and limiting the generalizability of models.
*   **Limited Audio Availability:** While datasets like the Million Song Dataset (MSD) or AudioSet are large, they often provide only pre-computed features or links to external online services for audio download. This prevents researchers from experimenting with new feature extraction methods, leveraging end-to-end learning (e.g., directly from raw audio waveforms), or ensuring long-term reproducibility if external services disappear or change.
*   **Poor Audio Quality/Length:** Some datasets provide short clips (10-30 seconds) or low-quality audio, which may not represent full-length tracks accurately and limits the study of musical structure or temporal dynamics.
*   **Sparse Metadata:** Many datasets lack comprehensive metadata beyond basic labels, hindering research into rich contextual relationships or multi-modal analysis.
*   **Restrictive Licensing:** Copyright issues have historically made it difficult to publicly distribute large collections of music audio, forcing researchers to rely on proprietary features or indirect access.

The FMA dataset directly addresses these limitations, aiming to provide a foundational resource comparable in its impact to ImageNet for computer vision, but within the MIR domain. It positions itself as a robust, open-access alternative that directly provides full-length, high-quality audio along with rich metadata and pre-computed features, all under permissive Creative Commons licenses. This focus on openness and direct audio access represents a significant step towards fostering more collaborative, reproducible, and advanced research in MIR.

### 3. Key Objectives and Motivation

The primary objective of this research is to introduce the **Free Music Archive (FMA) dataset**, a large-scale, open, and easily accessible collection of Creative Commons-licensed music audio and associated metadata. The overarching motivation is to overcome the critical hurdle posed by the limited availability of such datasets in the Music Information Retrieval (MIR) field, which has historically restrained the development and evaluation of data-intensive machine learning models, particularly deep learning.

Specific motivations and objectives include:

*   **Enabling Data-Heavy Models:** To provide a sufficiently large and diverse audio dataset that allows for the effective training of complex models like deep neural networks, which can learn features directly from raw audio or high-level representations, minimizing reliance on hand-crafted features.
*   **Fostering Reproducible Research:** To establish a common, open benchmark dataset that allows researchers worldwide to objectively compare their algorithms and models against a standardized baseline, thus advancing the field more efficiently and transparently. This addresses a common issue in MIR where different researchers use disparate datasets, making direct comparisons difficult.
*   **Providing Comprehensive Data:** To offer full-length, high-quality audio, differentiating FMA from datasets that provide only short clips or lower fidelity recordings. Additionally, to include rich, multi-level metadata (track, album, artist, user data, free-form text) to support a wider range of MIR tasks beyond basic classification, enabling more nuanced and contextual analyses.
*   **Overcoming Copyright Barriers:** By exclusively using Creative Commons-licensed music, the authors provide a legally permissible solution for broad dataset distribution, which has historically been a major impediment to open-access audio collections.
*   **Simplifying Accessibility:** To make the dataset easy to download and use, eliminating the need for complex web crawling or reliance on ephemeral external services. This includes providing pre-computed features and usage examples.
*   **Supporting Diverse MIR Tasks:** While specifically highlighting Music Genre Recognition (MGR) as a prime application due to the detailed genre hierarchy, the dataset is designed to be versatile enough for tasks like artist identification, year prediction, automatic tagging, music structure analysis, and even exploration for recommender systems.
*   **Promoting Feature Learning and End-to-End Systems:** By providing raw audio, the dataset encourages research into deep learning architectures that can learn representations directly from waveforms, moving beyond the limitations of relying solely on pre-engineered features.

In essence, the FMA dataset is presented as a cornerstone for future MIR research, aiming to unlock new possibilities for algorithm development and evaluation, much like ImageNet did for computer vision.

### 4. Methodology and Approach

The creation of the FMA dataset involved a systematic approach to data collection, structuring, and preparation, ensuring it meets the objectives of being large, accessible, and rich in content.

#### 4.1. Data Source and Collection

The FMA dataset is primarily a comprehensive dump of the **Free Music Archive (FMA)** website, a free and open online library directed by WFMU, a long-running freeform radio station. The FMA platform distinguishes itself by combining user-generated content with curatorial oversight, and crucially, all music on the platform is released under permissive Creative Commons licenses.

The data collection was performed on April 1st, 2017. The process involved:
*   **Track Identification:** Iterating through track IDs on the FMA website, starting from the largest known ID (155,320), to identify valid and available tracks.
*   **Metadata Collection:** Utilizing the FMA's public API to gather comprehensive metadata for each track, album, and artist. This included track titles, album details, artist names, genre information (including a hierarchical taxonomy of 161 genres), duration, bit rate, user engagement metrics (#listens, #comments, #favorites), creation/release dates, and free-form text (tags, biographies).
*   **Audio Download:** Downloading the corresponding MP3 audio files via HTTPS for each identified track.
*   **Filtering and Cleaning:** The collected data underwent a filtering process.
    *   Missing IDs (tracks deleted from the archive) were excluded.
    *   Tracks whose MP3s could not be downloaded or processed (e.g., trimmed by ffmpeg) were removed.
    *   Crucially, tracks with licenses prohibiting redistribution were discarded, ensuring the entire dataset adheres to permissive Creative Commons licenses. This left 106,574 tracks out of an initial 109,727 valid tracks.
*   **Data Structure:** All collected metadata was cleaned, uniformly formatted, and merged into a single relational table, `tracks.csv`, providing a user-friendly structure despite some inherent data redundancy (e.g., artist information repeated for all tracks by that artist). Auxiliary files like `genres.csv` detail the genre hierarchy.

The authors made a deliberate choice not to extensively "clean" the dataset by removing outliers (e.g., tracks with too many genres, very long durations, or belonging to rare genres). Their rationale was to preserve the "real-world" nature of the data, arguing that algorithms should be robust to such variations and that outliers have minimal impact on overall performance. Researchers are, however, free to apply their own filtering.

#### 4.2. Dataset Content and Characteristics

The resulting FMA dataset boasts:
*   **Size:** 917 GiB of audio, totaling 343 days of music.
*   **Scale:** 106,574 tracks from 16,341 artists and 14,854 albums.
*   **Audio Quality:** Full-length, high-quality MP3s (mostly 44,100 Hz sampling rate, 320 kbit/s bit rate, stereo).
*   **Metadata Richness:** Comprehensive track-level, album-level, and artist-level metadata, along with user-generated tags and free-form text (e.g., artist biographies), with varying levels of coverage across fields (Table 2).
*   **Genre Hierarchy:** A crucial feature is the built-in hierarchical taxonomy of 161 genres (16 top-level genres with numerous sub-genres), annotated by the artists themselves. The dataset distinguishes between `genres` (artist-indicated), `genres_all` (all genres traversing the hierarchy to the root), and `genres_top` (root genres).

#### 4.3. Pre-computed Features

To facilitate research, especially for those without the computational resources or desire to extract features from raw audio, the authors pre-computed 518 features using the `librosa` Python library (version 0.5.0). These include common MIR features such as Chroma, Tonnetz, MFCC (Mel-Frequency Cepstral Coefficients), Spectral Centroid, Spectral Bandwidth, Spectral Contrast, Spectral Rolloff, RMS Energy, and Zero-Crossing Rate. For most features, seven statistics (mean, standard deviation, skew, kurtosis, median, minimum, maximum) were computed over windows of 2048 samples with 512-sample hops, providing a compact representation of the track's sonic characteristics. These are provided in `features.csv`.

#### 4.4. Subsets and Splits

To accommodate different computational resources and research needs, the authors propose four nested subsets:
1.  **Full:** The complete dataset (106,574 tracks, 161 genres, full length audio).
2.  **Large:** The full dataset, but with audio limited to 30-second clips extracted from the middle of each track (or the entire track if shorter). This significantly reduces data size (98 GiB).
3.  **Medium:** A curated subset of 25,000 30-second clips. This subset focuses on tracks with only one top-level genre, chosen based on metadata completeness and popularity, to simplify single-label genre prediction tasks. It retains 16 top genres but is unbalanced.
4.  **Small:** A balanced subset of 8,000 30-second clips (1,000 clips per genre) from the 8 most popular top-level genres of the medium set. This subset is designed to be comparable to GTZAN but with FMA's advantages.

For reproducibility, the authors propose an 80/10/10% train/validation/test split for all subsets. Key constraints for these splits include:
*   **Stratified Sampling:** To ensure representation of all genres, especially minority ones, across all splits.
*   **Artist Filter:** A critical constraint where tracks by the same artist are assigned to only one split (train, validation, or test). This prevents "artist effect" or "album effect," which can artificially inflate accuracy in classification tasks if an artist's unique sonic signature is learned instead of general genre characteristics.

All code used for data collection, analysis, subset generation, feature computation, and baseline evaluation is publicly shared, promoting transparency and enabling community extension.

### 5. Main Findings and Results

The core "findings" of this paper revolve around the detailed characterization of the FMA dataset itself and the initial baseline performance evaluation for music genre recognition.

#### 5.1. Dataset Characteristics

The FMA dataset successfully fulfills its design goals, presenting a robust and comprehensive resource:
*   **Scale and Scope:** It contains 106,574 tracks, amounting to 917 GiB and 343 days of audio, making it significantly larger than any other readily available, quality audio dataset for MIR (Table 1). It includes music from 16,341 artists and 14,854 albums.
*   **Content Diversity:** The dataset captures a wide range of musical content, particularly strong in experimental, electronic, and rock genres (Figure 6). While biased towards non-mainstream, Creative Commons music, this is a necessary trade-off for permissive licensing and direct audio access.
*   **Audio Quality:** The provision of full-length, high-quality (mostly 320 kbit/s, 44.1 kHz, stereo) audio clips distinguishes FMA from many existing datasets that offer only short, low-fidelity snippets.
*   **Metadata Richness:** The extensive metadata covering track, album, and artist details, along with user-generated tags and free-form text, offers rich contextual information for various research tasks. The inclusion of hierarchical genre information (161 genres with 16 top-level categories) is particularly valuable.
*   **Accessibility and Reproducibility:** The dataset is designed for ease of use, with all data and code publicly available, checksummed, and hosted in a long-term digital archive, ensuring future accessibility and reproducibility of research results.

#### 5.2. Baseline Genre Recognition Performance

The authors conducted baseline experiments for music genre recognition (MGR) on the "medium" subset of FMA (25,000 30s clips, single top genre, 16 top genres). These baselines use the pre-computed `librosa` features and four standard machine learning classifiers: Linear Regression (LR), k-Nearest Neighbors (kNN), Support Vector Machines (SVM) with RBF kernel, and Multilayer Perceptron (MLP). The results, reported as test set accuracies in Table 6, are as follows:

*   **Feature Performance:**
    *   MFCCs (Mel-Frequency Cepstral Coefficients) generally performed the best among individual feature sets, achieving an accuracy of 61% with MLP and SVM.
    *   Spectral Contrast also showed good performance (54% with SVM).
    *   Combinations of features, such as MFCC + Spectral Contrast, slightly improved performance (63% with SVM), indicating the benefit of complementary information.
    *   Using all 518 pre-computed features resulted in the highest reported accuracy of 63% with SVM, slightly outperforming MLP (58%) and LR (61%), while kNN lagged at 52%.
*   **Classifier Performance:** SVM and MLP generally yielded the best results, demonstrating the utility of non-linear models for MGR on this dataset. Linear Regression was surprisingly competitive with a maximum of 61%.
*   **Task Difficulty:** The reported accuracies (max 63%) suggest that MGR on the FMA medium subset, even with traditional features and classifiers, is a non-trivial task. The authors explicitly state these results are baselines, not state-of-the-art, and serve to provide a reference point for future research using deep learning or more advanced techniques directly on audio.

These baselines are crucial as they offer initial performance indicators and a starting point for researchers to compare their novel algorithms against, without the need for extensive initial setup or feature engineering.

### 6. Significance and Potential Impact

The FMA dataset represents a significant contribution to the Music Information Retrieval (MIR) community, with far-reaching potential impacts on research and development in the field.

#### 6.1. Advancing MIR Research

*   **Enabling Deep Learning:** The most prominent impact is its potential to unlock the power of deep learning for audio analysis in MIR. By providing a large volume of full-length, high-quality audio, FMA removes a major bottleneck that previously limited the application of data-hungry neural network architectures. Researchers can now train models directly on raw waveforms or spectrograms, allowing for end-to-end learning and potentially discovering novel features without manual engineering, which has been hypothesized as a cause of stagnation in MIR tasks.
*   **Reproducibility and Benchmarking:** FMA establishes a much-needed open benchmark. The provision of standardized train/validation/test splits, coupled with the "artist filter" to prevent data leakage and inflated results, ensures that research findings are more robustly comparable and reproducible. This standardization is critical for accelerating progress and fostering a more collaborative research environment.
*   **Diverse Research Applications:** Beyond music genre recognition, the rich metadata and full-length audio enable a multitude of other MIR tasks:
    *   **Music Classification and Annotation:** Artist identification, year prediction, mood classification (if tags are interpreted for mood), and fine-grained automatic tagging.
    *   **Music Structure Analysis:** The availability of full-length tracks allows for detailed studies of musical structure, temporal dynamics, and changes throughout a composition.
    *   **Metadata Analysis:** The extensive metadata can be used by musicologists or data scientists to study relationships between musical properties and higher-level representations, user engagement (play counts, favorites), and social network effects.
    *   **Recommender Systems:** While aggregated user data is currently available, the potential to collect anonymized user activity (favorites, listens) could pave the way for large-scale content-based recommender system evaluations.
*   **Exploration of Genre Hierarchies:** The explicit inclusion of a hierarchical genre taxonomy is a unique strength, allowing researchers to explore multi-label classification, multi-level genre prediction, and the inherent fuzziness and relationships between musical categories.

#### 6.2. Future Directions and Limitations

The authors are transparent about the dataset's limitations and suggest future work:
*   **Content Bias:** The dataset is biased towards experimental, electronic, and rock music, reflecting the nature of Creative Commons content on the Free Music Archive. It largely lacks mainstream, commercially successful music. This raises questions about whether algorithms trained on FMA generalize to mainstream music, a critical area for future evaluation.
*   **Ground Truth Validation:** The genre labels are artist-provided, which can introduce inconsistencies or subjectivity. Future work could involve validating ground truth through independent annotators or crowd-sourcing to assess inter-annotator agreement and refine labels.
*   **Missing Metadata:** While rich, some common MIR tasks like mood classification or instrument recognition are not directly supported by explicit labels. However, deeper analysis of the available free-form tags might reveal latent information for these tasks.
*   **Expandability:** The dataset can be further enriched by scraping more information from the website, cross-referencing with other music databases (e.g., MusicBrainz, Last.fm for lyrics, cover songs, etc.), or through community-driven crowd-sourcing efforts.

Despite these limitations, the FMA dataset provides a powerful foundation. Its commitment to openness, quality, and scale aligns perfectly with the evolving needs of the MIR community and the broader machine learning landscape. By fostering a collaborative and reproducible research environment, FMA is poised to drive significant advancements in how we understand, organize, and interact with large music collections.

### Conclusion

"FMA: A Dataset for Music Analysis" introduces a meticulously constructed, large-scale, open-access dataset that directly addresses critical shortcomings in existing Music Information Retrieval (MIR) resources. By providing 917 GiB of full-length, high-quality Creative Commons-licensed audio from over 100,000 tracks, coupled with rich multi-level metadata and a hierarchical genre taxonomy, the FMA dataset serves as an indispensable tool for advancing MIR research. Its emphasis on accessibility, reproducibility, and enabling data-hungry machine learning models, particularly deep learning, positions it as a foundational benchmark akin to ImageNet in computer vision. While acknowledging biases inherent in its Creative Commons source, the FMA represents a monumental step towards a more open, transparent, and collaborative future for music analysis research.