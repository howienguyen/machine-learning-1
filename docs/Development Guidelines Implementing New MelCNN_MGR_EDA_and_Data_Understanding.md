Tue Mar 10 08:57:59 UTC 2026
New version

# Development Guidelines: Implementing the Manifest/Log-Mel EDA Notebook

## 1. Purpose of this document

This document explains how to design and implement the manifest-centric EDA notebook for the MelCNN-MGR pipeline.

The current primary notebook is:

`MelCNN-MGR/notebooks/MelCNN_MGR_Manifest_LogMel_EDA.ipynb`

The older `MelCNN_MGR_EDA_and_Data_Understanding.ipynb` can still serve as reference material, but it is no longer the best name or framing for the current production-like path.

This document serves two roles at the same time:

1. a **development guide** for building the notebook cleanly and correctly;
2. a **tutorial** for understanding the dataset manifests, sampling logic, duration handling, and final split behavior.

The notebook is not meant to train a model. Its job is to help you understand the dataset before feature extraction and before CNN training.

It belongs in the current production-like chain here:

```text
download_by_genre_limits.py
	-> extract_mtg_processed_samples.py

data sources: FMA + additional_datasets
	-> 1_build_all_datasets_and_samples.py
	-> 2_build_log_mel_dataset.py
	-> MelCNN_MGR_Manifest_LogMel_EDA.ipynb
	-> logmel_cnn_v1.py
```

---

## 2. What the notebook should study

The attached pipeline script builds three parquet artifacts:

* `manifest_all_datasets.parquet`
* `manifest_all_samples.parquet`
* `manifest_final_samples.parquet`

These three files define the natural EDA boundary of the project:

* **file-level discovery and audit**
* **segment-level expansion**
* **final training split selection**

So the notebook should be **manifest-centric**, not just spectrogram-centric.

A good mental model is:

**raw audio discovery -> usability filtering -> segment generation -> final split selection -> training readiness**

For the current project, that mental model should be extended one step further downstream:

**raw audio discovery -> usability filtering -> segment generation -> final split selection -> log-mel feature readiness -> training readiness**

---

## 3. Why this notebook matters in the pipeline

`1_build_all_datasets_and_samples.py` sits between raw audio collections and downstream feature extraction. Its responsibilities are to:

* discover in-scope audio files;
* normalize source metadata into a shared schema;
* probe actual durations;
* assign reason codes explaining usability or skip status;
* expand usable files into fixed-length segment rows;
* assign final train/validation/test splits at the source-audio level.

That means the notebook is really a **data audit and pipeline-understanding notebook**.

In the current production-like path, it should also help answer whether the outputs are ready for:

1. `2_build_log_mel_dataset.py`
2. `logmel_cnn_v1.py`

It should answer questions like:

* What data do I really have?
* Which files are usable or skipped, and why?
* How does file duration affect the number of available segments?
* How does the combined dataset change from file level to sample level?
* Is the final dataset balanced and leakage-safe?

---

## 4. Core inputs the notebook must use

The notebook should load the following inputs:

* `manifest_all_datasets.parquet`
* `manifest_all_samples.parquet`
* `manifest_final_samples.parquet`
* `settings.json`
* optional downstream log-mel split indexes when they exist

The script and markdown documentation show that the sampling configuration is read from `settings.data_sampling_settings`, especially:

* `target_genres`
* `sample_length_sec`
* `min_duration_delta`
* `number_of_samples_expected_each_genre`
* `train_n_val_test_split_ratio_each_genre`

Your notebook should display these settings early, because they define how the manifests were produced.

---

## 5. Notebook design philosophy

Build the notebook as a sequence of increasingly concrete questions.

### Layer 1: dataset audit

Understand what exists and what was skipped.

### Layer 2: duration and sampling understanding

Understand how duration affects segment generation and sample yield.

### Layer 3: final split understanding

Understand how Stage 2 selects and assigns final training, validation, and test rows.

### Layer 4: training-readiness understanding

Understand whether the final result is balanced, interpretable, and safe for downstream model building.

This layered structure is much better than writing one giant notebook blob full of random charts.

---

## 6. Recommended notebook structure

### Section 1. Introduction and scope

Explain:

* what the notebook analyzes;
* which pipeline artifacts it depends on;
* that it is focused on EDA and data understanding, not model training;
* where it sits between `2_build_log_mel_dataset.py` and `logmel_cnn_v1.py`.

### Section 2. Configuration and path setup

Create a clean configuration cell with:

* paths to the three parquet files;
* path to `settings.json`;
* an optional output directory for saved plots;
* a switch for whether to save figures.

Also parse and display the relevant settings from `settings.json`.

### Section 3. Load and inspect manifests

Load all three parquet files and inspect:

* shapes;
* columns;
* dtypes;
* null counts;
* first few rows.

This is the first sanity check.

### Section 4. File-level dataset composition

Use `manifest_all_datasets.parquet` to analyze:

* total number of discovered audio candidates;
* counts by `source`;
* counts by `genre_top`;
* counts by `reason_code`;
* counts by `sampling_eligible`.

This section should include class-distribution charts and imbalance measures.

### Section 5. Skip logic and reason-code analysis

This section should focus on:

* `reason_code`
* `sampling_exclusion_reason`
* skipped files by source and genre

The purpose is to understand which files were filtered out and why.

### Section 6. Duration analysis

Use both:

* `actual_duration_s`
* `duration_s`

Explain the difference clearly:

* `actual_duration_s` is the measured duration from the audio probe;
* `duration_s` is the normalized duration used by the script for eligibility and segmentation.

This section should include:

* duration histograms;
* boxplots by genre;
* boxplots by source;
* the delta between measured and normalized duration.

### Section 7. Duration grouping and sample-yield analysis

This is the custom section you requested.

Here the notebook should compute a **duration-derived sample-yield metric** using your notebook rule:

* round upward to the next whole second;
* divide by `sample_length_sec`.

This must be presented as a notebook-side analytic metric, not silently confused with the script’s current Stage 1 segment-row expansion rule.

This section should include:

* interpretable duration bins;
* optional 1D K-means on duration;
* counts of files by genre and duration group;
* total yield by genre and duration group;
* average yield per file by genre and duration group.

### Section 8. Actual emitted sample-row analysis

Use `manifest_all_samples.parquet` to analyze the segment rows actually emitted by the script.

Study:

* total segment rows;
* segments by genre;
* segments by source;
* distribution of `total_segments_from_audio`;
* relationship between duration and emitted segment count.

This is where you compare “theoretical notebook yield” vs “actual current script segment rows.”

### Section 9. Final split analysis

Use `manifest_final_samples.parquet` to analyze:

* final counts by `final_split`;
* final counts by genre and split;
* per-genre target vs actual selected rows;
* split balance.

### Section 10. Leakage-safety validation

The script assigns final splits at the source-audio level by grouping segment ids after removing the `:segNNNN` suffix.

The notebook should verify that no base source-audio group appears in more than one final split.

### Section 11. Class balance evolution across stages

Track class balance across:

* `manifest_all_datasets.parquet`
* `manifest_all_samples.parquet`
* `manifest_final_samples.parquet`

This section should show how the genre distribution changes from file-level discovery to training-ready segments.

### Section 12. Key findings and readiness summary

End the notebook with a markdown summary of findings such as:

* which genres are most abundant or underrepresented;
* which reason codes dominate skipped files;
* whether duration distribution differs by genre;
* which duration groups contribute most segment supply;
* whether the final split is reasonably balanced and leakage-safe.

---

## 7. Core EDA concepts the notebook should teach

This notebook should also function as a tutorial. So each section should briefly explain the EDA concept it uses.

### A. Class distribution analysis

This tells you whether the dataset is balanced across genres.

Useful metrics:

* sample counts per genre;
* class proportions;
* imbalance ratio.

Why it matters:

A classifier can look decent while mainly learning the majority classes.

### B. Reason-code analysis

This is dataset auditing.

Useful metrics:

* number of skipped files;
* skip percentage;
* counts by `reason_code` and `sampling_exclusion_reason`.

Why it matters:

A dataset is not just defined by what enters the pipeline, but also by what gets filtered out.

### C. Duration analysis

This explains the file-length landscape.

Useful metrics:

* mean, median, min, max duration;
* per-genre duration summary;
* normalized-vs-actual duration difference.

Why it matters:

Duration directly controls how many fixed-length samples can be generated.

### D. Duration grouping

This turns a continuous variable into interpretable groups.

Useful methods:

* manual bins;
* 1D K-means as an optional exploratory supplement.

Why it matters:

It lets you answer: which kinds of files contribute most of the training supply?

### E. Segment-expansion analysis

This studies what actually happened when Stage 1 created segment rows.

Useful metrics:

* total segments;
* segments per genre;
* average segments per source audio;
* long-tail behavior in segment supply.

Why it matters:

A small number of long audio files can dominate the total number of segments.

### F. Final split analysis

This studies training readiness.

Useful metrics:

* counts per split;
* counts per split by genre;
* selected total vs target total.

Why it matters:

You want a final manifest that is both balanced and suitable for downstream training.

### G. Leakage-safety validation

This checks that all segments from one source audio remain in only one final split.

Why it matters:

If one audio file leaks into both training and test via different segments, your evaluation becomes suspiciously magical.

---

## 8. Important implementation rule: distinguish two segment notions

This is one of the most important development rules.

The attached script currently computes emitted segment rows using the normalized `duration_s` and a floor-based segment-count rule in Stage 1.

Your notebook, however, also needs a **duration-derived yield metric** based on your requested rule:

1. round `duration_s` upward to the next whole second;
2. divide by `sample_length_sec`.

These are not the same beast.

So the notebook should expose them separately.

### Recommended derived columns

Create notebook-side columns such as:

* `duration_sec_ceil` = ceiling of `duration_s`
* `sample_yield_ratio` = `duration_sec_ceil / sample_length_sec`

Then keep the script-emitted count separate, for example:

* `sampling_num_segments`
* `total_segments_from_audio`

### Why this separation matters

Because otherwise the notebook may show a yield value like `2.6` while the emitted Stage 1 manifest contains only `2` actual segment rows, and that mismatch will look like a bug even when it is just a different definition.

Be explicit. State it in markdown. Do not hide it.

---

## 9. Recommended visualizations

Use simple, readable visualizations. The goal is understanding, not chart cosplay.

### Recommended chart types

* bar chart for counts by genre;
* bar chart for counts by source;
* stacked bar chart for genre by reason code;
* histogram of actual duration;
* histogram of normalized duration;
* boxplot of duration by genre;
* boxplot of duration by source;
* stacked bar chart of file count by genre and duration group;
* stacked bar chart of sample-yield total by genre and duration group;
* histogram of emitted segment count per audio file;
* grouped bar chart of final split counts by genre;
* heatmap of genre vs final split counts.

### Plotting style advice

* keep labels readable;
* sort categories when useful;
* annotate important bars if counts are small enough;
* do not overload one figure with too many categories;
* use saved helper functions to keep the notebook tidy.

---

## 10. Recommended helper functions

To keep the notebook maintainable, implement helper functions for repeated tasks.

Good candidates include:

* `load_settings()`
* `show_basic_frame_info(df, name)`
* `compute_imbalance_ratio(counts)`
* `build_duration_bins(df, col)`
* `compute_duration_yield(df, duration_col, sample_length_sec)`
* `summarize_by_genre(df, value_col)`
* `plot_count_bar(series, title)`
* `plot_grouped_counts(df, row_col, group_col)`
* `check_final_split_leakage(final_df)`
* `extract_base_sample_id(sample_id)`

These helpers make the notebook easier to debug and easier to reuse later.

---

## 11. Suggested data-quality checks

A solid EDA notebook should not just plot things. It should also assert basic expectations.

Useful checks include:

* required files exist;
* required columns exist in each parquet;
* `final_split` only contains `training`, `validation`, `test`;
* `reason_code` is never null where it should exist;
* `segment_end_sec - segment_start_sec == sample_length_sec` for emitted samples;
* each base source-audio group appears in only one final split;
* `total_segments_from_audio` is constant within one source-audio group.

These checks make the notebook part tutorial, part audit harness.

---

## 12. How to explain the three manifests in the notebook

The notebook should teach the meaning of each manifest clearly.

### `manifest_all_datasets.parquet`

One row per discovered audio candidate.

Use it to answer:

* what exists;
* what source it comes from;
* what genre label it has;
* whether it is usable or skipped;
* what duration it has;
* how many segments it could produce according to the script.

### `manifest_all_samples.parquet`

One row per emitted fixed-length segment.

Use it to answer:

* how the dataset expands at segment level;
* which genres produce more segment rows;
* whether long files dominate sample supply.

### `manifest_final_samples.parquet`

One row per selected final segment used for downstream model consumers.

Use it to answer:

* what the final train/validation/test dataset looks like;
* whether selection is balanced;
* whether split assignment is leakage-safe.

---

## 13. Tutorial notes to include as markdown cells

To make the notebook educational, add short markdown explanations before each major analysis block.

Examples:

### Before class distribution analysis

Explain that class distribution matters because imbalance can bias training and interpretation.

### Before reason-code analysis

Explain that skipped rows are valuable evidence about dataset quality and preprocessing rules.

### Before duration analysis

Explain that duration is not just metadata; it controls how much segment supply each file can contribute.

### Before final split analysis

Explain that split quality is part of data understanding, not only part of model evaluation.

These markdown cells make the notebook readable later when you revisit the project after your brain has been replaced by coffee vapor.

---

## 14. Practical implementation sequence

Build the notebook in this order.

### Step 1

Implement config, imports, display options, and file loading.

### Step 2

Implement basic inspection helpers and validate that the manifests load correctly.

### Step 3

Implement file-level composition analysis and reason-code summaries.

### Step 4

Implement duration analysis and normalized-vs-actual comparisons.

### Step 5

Implement your notebook-side duration-yield metric and duration grouping analysis.

### Step 6

Implement segment-row analysis from `manifest_all_samples.parquet`.

### Step 7

Implement final split analysis and leakage checks.

### Step 8

Add markdown interpretation cells and a final summary section.

This order mirrors the data pipeline itself and keeps the notebook logically clean.

---

## 15. Common pitfalls to avoid

### Pitfall 1: mixing actual duration with normalized duration

Always explain which duration field is being used.

### Pitfall 2: mixing notebook yield with emitted segment count

Keep them as separate concepts.

### Pitfall 3: using only counts without percentages

Counts alone can hide imbalance.

### Pitfall 4: ignoring skipped files

Skipped files are part of the story.

### Pitfall 5: trusting final split balance without leakage checks

A split can look balanced and still be invalid if one audio source leaks across splits.

### Pitfall 6: creating too many low-value plots

Every figure should answer a concrete question.

---

## 16. What a strong finished notebook should deliver

A well-implemented notebook should let you confidently state:

* how many audio files exist at file level;
* how many are usable and unusable;
* why files are skipped;
* how duration behaves across genres and sources;
* which duration groups contribute the largest share of sample supply;
* how many segment rows are actually emitted;
* how class balance evolves from file level to segment level to final split;
* whether the final manifest is leakage-safe and training-ready.

That is the real success criterion.

---

## 17. Final development recommendation

Treat this notebook as a **first-class pipeline artifact**, not as a disposable side notebook.

That means:

* keep section headers clean;
* write short explanatory markdown cells;
* use helper functions;
* validate assumptions with checks;
* keep analysis definitions explicit;
* make the notebook rerunnable without manual edits;
* save figures or summary tables where useful.

In other words, build it like a tool you will trust later, not a one-night exploratory creature stitched together from pandas and optimism.

---

## 18. One-line summary

`MelCNN_MGR_EDA_and_Data_Understanding.ipynb` should function as a manifest-driven audit notebook that explains how raw audio files become usable fixed-length training segments, how duration shapes sample supply, and whether the final dataset is balanced, interpretable, and leakage-safe.
