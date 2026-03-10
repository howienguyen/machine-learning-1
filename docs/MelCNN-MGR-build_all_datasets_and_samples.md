# MelCNN-MGR `build_all_datasets_and_samples.py`

## Purpose

`MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py` is the manifest builder for the MelCNN-MGR data pipeline.

It creates the three core parquet artifacts used by downstream preprocessing and model training:

1. `manifest_all_datasets.parquet`
2. `manifest_all_samples.parquet`
3. `manifest_final_samples.parquet`

At a high level, the script:

1. Loads data sampling settings from `MelCNN-MGR/settings.json`
2. Builds or loads FMA candidates
3. Scans additional datasets from `additional_datasets/data`
4. Normalizes both sources into one shared dataset manifest
5. Computes fixed-length segment rows
6. Assigns final train/validation/test splits at the source-audio level
7. Writes parquet outputs plus a human-readable report and config snapshot

This document explains the script’s responsibilities, execution flow, decision rules, and output schemas.

## Two-stage model

The script is best understood as a two-stage pipeline, and the CLI can execute Stage 1 only, Stage 2 only, or both stages in one invocation.

### Stage 1: intermediate manifest generation

Stage 1 produces:

1. `manifest_all_datasets.parquet`
2. `manifest_all_samples.parquet`

These files are the auditable intermediate products. They capture discovery, filtering, duration handling, and fixed-length segment expansion.

### Stage 2: final training-manifest selection

Stage 2 consumes the Stage 1 sample rows and produces:

1. `manifest_final_samples.parquet`

This final parquet is the lean segment-level manifest intended for model-training consumers and downstream feature-building code.

So the separation is real at the data-contract level, and the CLI now exposes that separation directly through a mode flag.

## Main role in the pipeline

This script sits between raw audio discovery and feature extraction.

Its job is not to compute mel spectrograms or train models. Its job is to produce a deterministic, auditable sampling plan that answers these questions:

1. Which audio files exist and are in scope?
2. Which genre label does each file map to?
3. Is each file usable or skipped, and why?
4. How many fixed-length segments can be generated from each usable file?
5. Which segments belong to the final train, validation, and test sets?

The downstream log-mel builder can then consume `manifest_final_samples.parquet` without having to repeat dataset discovery logic.

## Script inputs

### Configuration input

The script reads `settings.data_sampling_settings` from `MelCNN-MGR/settings.json`.

The relevant keys are:

| Key | Meaning |
| --- | --- |
| `target_genres` | List of target genre labels to keep |
| `sample_length_sec` | Fixed segment length used to build sample rows |
| `min_duration_delta` | Small tolerance used to derive the FMA minimum duration threshold |
| `number_of_samples_expected_each_genre` | Per-genre target size for the final manifest |
| `train_n_val_test_split_ratio_each_genre` | Fraction of final selected samples assigned to training |

### File-system inputs

| Input | Default path | Purpose |
| --- | --- | --- |
| FMA metadata | `FMA/fma_metadata` | Source of `tracks.csv` when rescanning |
| FMA audio root | `FMA/fma_medium` | Source of FMA audio files |
| Cached FMA manifest | `MelCNN-MGR/data/processed/metadata_manifest_<subset>.parquet` | Optional fast path instead of rebuilding from `tracks.csv` |
| Additional datasets root | `additional_datasets/data` | Extra labeled audio folders outside FMA |

### Command-line inputs

The script exposes CLI arguments for all major input paths plus runtime options such as:

| Argument | Meaning |
| --- | --- |
| `--settings` | Path to `settings.json` |
| `--mode` | Execution mode: `stage1`, `stage2`, or `both` |
| `--medium-manifest` | Path to cached FMA parquet manifest |
| `--fma-metadata-root` | Path to FMA metadata folder |
| `--fma-audio-root` | Path to FMA audio folder |
| `--fma-subset` | FMA subset: `small`, `medium`, or `large` |
| `--min-duration` | Optional explicit minimum duration override |
| `--force-rescan` | Rebuild FMA candidates from `tracks.csv` even if a cached parquet exists |
| `--additional-root` | Path to additional datasets folder |
| `--all-datasets-out` | Output path for `manifest_all_datasets.parquet` |
| `--all-samples-out` | Output path for `manifest_all_samples.parquet` |
| `--final-samples-out` | Output path for `manifest_final_samples.parquet` |
| `--split-seed` | Random seed for deterministic final split assignment |
| `--log-level` | Logging verbosity |

## Outputs

The script writes the following artifacts by default under `MelCNN-MGR/data/processed`:

| Output | Purpose |
| --- | --- |
| `manifest_all_datasets.parquet` | Stage 1 output: one row per discovered audio candidate |
| `manifest_all_samples.parquet` | Stage 1 output: one row per fixed-length segment |
| `manifest_final_samples.parquet` | Stage 2 output: selected segment rows for final train/validation/test usage |

Report/config behavior depends on mode:

| Mode | Report/config base |
| --- | --- |
| `stage1` | `manifest_all_datasets.report.txt`, `manifest_all_datasets.config.json` |
| `stage2` | `manifest_final_samples.report.txt`, `manifest_final_samples.config.json` |
| `both` | `manifest_all_datasets.report.txt`, `manifest_all_datasets.config.json` |

## Execution modes

The CLI exposes the stage boundary directly with `--mode`.

### `--mode stage1`

Runs only:

1. FMA candidate build/load
2. additional dataset scan
3. combined dataset manifest generation
4. segment expansion into `manifest_all_samples.parquet`

Outputs written:

1. `manifest_all_datasets.parquet`
2. `manifest_all_samples.parquet`

### `--mode stage2`

Runs only final split assignment.

Input requirement:

1. an existing Stage 1 sample manifest at the path given by `--all-samples-out`

Outputs written:

1. `manifest_final_samples.parquet`

### `--mode both`

Runs the full Stage 1 plus Stage 2 pipeline.

## End-to-end workflow

The script runs in five main steps.

### Step 1: Build FMA candidates

The FMA path has two modes.

#### Mode A: cached parquet

If `metadata_manifest_<subset>.parquet` exists and `--force-rescan` is not set, the script loads that parquet and then enriches it by probing actual audio durations.

#### Mode B: rescan from `tracks.csv`

If the cache is missing or `--force-rescan` is enabled, the script:

1. Loads `tracks.csv`
2. Extracts the FMA columns needed for sampling and split logic
3. Filters rows to the chosen subset and target genres
4. Computes the expected MP3 path for each `track_id`
5. Checks whether the file exists
6. Probes actual audio duration from the audio file
7. Assigns a `reason_code`
8. Computes segment eligibility

The FMA path convention is:

`track_id = 4537` -> `004/004537.mp3`

This is built by zero-padding the numeric track id to 6 digits and using the first 3 digits as the folder name.

### Step 2: Scan additional datasets

The script scans `additional_datasets/data` recursively.

The expected folder structure is source-oriented:

1. source folder
2. genre folder
3. audio files under that genre folder

It maps folder names to target genres using `_genre_alias_targets(...)`.

Current alias handling includes:

| Folder key | Target mapping |
| --- | --- |
| `hiphop` | `Hip-Hop` |
| `raphiphop` | `Hip-Hop` |
| `folkcountry` | `Folk`, `Country` when both are part of the target list |

Each matching file is probed for duration and normalized into the same shared schema used by FMA rows.

### Step 3: Combine dataset manifest

FMA rows and additional-dataset rows are concatenated into one dataframe with a common schema.

The combined dataframe is then:

1. restricted to a canonical column order
2. deduplicated by `artifact_id`
3. sorted by `genre_top`, `source`, `artifact_id`

This final combined dataframe becomes `manifest_all_datasets.parquet`.

### Step 4: Build sample segments

Only rows meeting all of the following are expanded into segment rows:

1. `reason_code == "OK"`
2. `sampling_eligible == True`
3. `sampling_num_segments` is not null

For each eligible row, the script generates:

`floor(duration_s / sample_length_sec)`

segment rows.

Each segment row receives a segment-level `sample_id` in this form:

`<artifact_id>:seg0000`

Examples:

1. `fma:4537:seg0000`
2. `dortmund-university:Rock:additional_datasets/data/dortmund-university/rock/foo.wav:seg0001`

These rows become `manifest_all_samples.parquet`.

At this point, Stage 1 is complete.

### Step 5: Assign final splits

The final split assignment works on segment rows, but it groups them by source-audio identity first.

That grouping is derived from `sample_id` by stripping the trailing `:segNNNN` suffix.

This prevents segment leakage across train, validation, and test:

1. all segments from the same source audio stay in the same final split
2. split assignment is deterministic for a given random seed
3. per-genre selection is capped by `number_of_samples_expected_each_genre`

The resulting selected rows become `manifest_final_samples.parquet`.

This is Stage 2.

When `--mode stage2` is used, the script starts here by loading the existing Stage 1 sample manifest from `--all-samples-out`.

## Duration handling

The script intentionally keeps two duration concepts.

### `actual_duration_s`

This is the actual measured duration read from the file.

The script probes duration from the file itself instead of trusting metadata.

### `duration_s`

This is the normalized duration used for eligibility and segment counting.

Normalization rule:

1. round the measured duration to one decimal place
2. round up to the next whole second

Examples:

| Actual `actual_duration_s` | Normalized `duration_s` |
| --- | --- |
| `29.988571` | `30.0` |
| `29.900011` | `30.0` |
| `29.04` | `29.0` |
| `30.0` | `30.0` |

This separation is important because:

1. `actual_duration_s` preserves what was actually measured
2. `duration_s` produces stable sampling behavior around whole-second thresholds

## Audio duration probe order

The script uses two probing backends:

| Backend | Method |
| --- | --- |
| `ffprobe` | External command reading container duration metadata |
| `soundfile` | Python library reading frame count and sample rate |

Probe order depends on file type:

| File type | Probe order |
| --- | --- |
| Most formats | `ffprobe` first, then `soundfile` |
| `.wav`, `.flac`, `.aif`, `.aiff`, `.au`, `.ogg` | `soundfile` first, then `ffprobe` |

The goal is practical reliability:

1. `ffprobe` is generally more dependable for compressed formats such as MP3 and M4A
2. `soundfile` is efficient and reliable for common PCM or libsndfile-friendly formats
3. if the first probe fails, the second backend still gets a chance

If both methods fail, the row is marked `AUDIO_READ_FAILED`.

## Reason codes and skip logic

Each dataset row gets a single `reason_code` explaining its usability.

Current reason codes used by this script include:

| Reason code | Meaning |
| --- | --- |
| `OK` | row is usable |
| `NO_AUDIO_FILE` | expected file is missing |
| `AUDIO_READ_FAILED` | audio duration could not be read |
| `NO_LABEL` | label is missing |
| `EXCLUDED_LABEL` | label is intentionally excluded |
| `NO_SPLIT` | FMA split metadata is missing or invalid |
| `TOO_SHORT` | file is shorter than the minimum duration threshold |
| `TOO_SHORT_FOR_SAMPLE_LENGTH` | long enough to exist in the dataset manifest, but too short to produce one full fixed-length sample |

Skip behavior is split into two layers:

1. dataset-level exclusion via `reason_code`
2. sample-generation exclusion via `sampling_exclusion_reason`

The helper `summarize_skipped_audio_paths(...)` aggregates skipped filepaths for reporting, and `log_skipped_audio_paths(...)` prints both the skipped file count and the individual path lines.

## Final split strategy

Final split assignment is not copied directly from FMA. The script builds its own final split distribution over the generated segment rows.

Important details:

1. selection is per genre
2. the target size per genre is capped by `number_of_samples_expected_each_genre`
3. the training share is determined by `train_n_val_test_split_ratio_each_genre`
4. the remaining rows are divided between validation and test
5. grouping happens at source-audio level, not per segment

This means `manifest_final_samples.parquet` is designed for downstream training consumption, while `manifest_all_datasets.parquet` remains the broader audit table.

## Reporting and reproducibility

In addition to parquet outputs, the script writes two supporting artifacts.

### Report file

The report includes:

1. configuration summary
2. dataset row counts
3. skipped-file count
4. row counts by source and genre
5. reason-code breakdown
6. split distribution for FMA rows
7. artist leakage check for FMA OK rows
8. sample counts by source and genre
9. final sample counts by split and genre

### Config snapshot

The config JSON records the exact settings used to create the build outputs.

This makes the preprocessing step reproducible and auditable.

## Parquet schema reference

The tables below describe the parquet schemas written by this script. These schemas are derived from the current writer logic in the script.

### `manifest_all_datasets.parquet`

One row per discovered audio candidate across FMA and additional datasets.

| Column | Type | Meaning |
| --- | --- | --- |
| `source` | string | Source dataset name such as `fma-medium` or an additional dataset folder name |
| `artifact_id` | string | Base source-audio identifier for the discovered artifact |
| `source_track_id` | string or null | Original source-side track id when available |
| `track_id` | nullable Int64 | Numeric track id for FMA rows; null for additional datasets |
| `genre_top` | string | Top-level target genre label |
| `filepath` | string | Absolute file path to the audio file |
| `audio_exists` | bool | Whether the audio file exists on disk |
| `filesize_bytes` | nullable Int64 | File size in bytes when readable |
| `actual_duration_s` | float | Actual measured audio duration from file probe |
| `duration_s` | float | Normalized duration used for sampling decisions |
| `reason_code` | string | Primary status explaining why the row is usable or skipped |
| `sampling_eligible` | bool | Whether the row can produce at least one full sample segment |
| `sampling_num_segments` | nullable Int64 | Number of sample segments that can be produced |
| `sampling_exclusion_reason` | string or null | Why sampling is not possible even if the row exists in the dataset manifest |
| `manifest_origin` | string | Provenance marker showing which manifest or source folder created the row |

### `manifest_all_samples.parquet`

One row per fixed-length segment generated from the eligible dataset rows.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Segment-level id in the form `<artifact_id>:segNNNN` |
| `source` | string | Source dataset name |
| `genre_top` | string | Target genre label |
| `filepath` | string | Absolute audio file path |
| `track_id` | nullable Int64 | Numeric track id for FMA rows |
| `sample_length_sec` | float | Fixed segment length used for every sample row |
| `segment_index` | int | Zero-based segment index within the source audio |
| `segment_start_sec` | float | Segment start time in seconds |
| `segment_end_sec` | float | Segment end time in seconds |
| `total_segments_from_audio` | nullable Int64 | Total number of segments derived from the source audio |
| `duration_s` | float | Normalized duration used when segmenting this source audio |
| `actual_duration_s` | float | Actual measured duration of the source audio |
| `reason_code` | string | Copied from the source dataset row; expected to be `OK` for generated sample rows |

### `manifest_final_samples.parquet`

Lean final training manifest. One row per selected segment.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Segment-level sample id |
| `source` | string | Source dataset name |
| `genre_top` | string | Target genre label |
| `filepath` | string | Absolute audio path |
| `track_id` | nullable Int64 | Numeric track id for FMA rows |
| `sample_length_sec` | float | Fixed segment length |
| `segment_index` | int | Zero-based segment index |
| `segment_start_sec` | float | Segment start time in seconds |
| `segment_end_sec` | float | Segment end time in seconds |
| `total_segments_from_audio` | nullable Int64 | Total number of segments derived from the source audio |
| `duration_s` | float | Normalized duration used when segmenting this source audio |
| `actual_duration_s` | float | Actual measured duration of the source audio |
| `reason_code` | string | Status field carried through for downstream safety checks |
| `final_split` | string | Final split assigned by this script: `training`, `validation`, or `test` |

## Design decisions worth knowing

### Why `artifact_id` and `sample_id` are separated

The script intentionally uses two identity levels:

1. `artifact_id` for source-audio artifacts in `manifest_all_datasets.parquet`
2. `sample_id` for fixed-length segment rows in `manifest_all_samples.parquet` and `manifest_final_samples.parquet`

This avoids collisions that would happen if only integer `track_id` were used, while keeping the segment manifests focused on segment-level identities.

### Why `manifest_final_samples.parquet` is lean

The final manifest intentionally omits several fields that are useful for auditing but not required by downstream segment-level consumers.

For example, it omits:

1. `source_type`
2. `relative_path`
3. `subset`
4. `split`
5. `artist_id`
6. `file_ext`

This keeps the final training manifest compact while retaining the key fields needed for loading, segment slicing, labeling, and split selection.

### Why skipped files are still represented upstream

The script does not try to hide unusable inputs. Instead, it keeps them in `manifest_all_datasets.parquet` with explicit reason codes.

That makes it possible to answer questions like:

1. how many files were unreadable?
2. how many files were too short?
3. which exact paths were skipped?

This is critical for dataset auditing and rerun reproducibility.

## Typical execution example

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
  --mode both \
  --fma-subset medium \
  --log-level INFO
```

Stage 1 only:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
  --mode stage1 \
  --fma-subset medium \
  --log-level INFO
```

Stage 2 only:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
  --mode stage2 \
  --all-samples-out MelCNN-MGR/data/processed/manifest_all_samples.parquet \
  --final-samples-out MelCNN-MGR/data/processed/manifest_final_samples.parquet \
  --log-level INFO
```

To force a rebuild from `tracks.csv` instead of using a cached FMA parquet:

```bash
python MelCNN-MGR/preprocessing/build_all_datasets_and_samples.py \
  --mode both \
  --fma-subset medium \
  --force-rescan \
  --log-level INFO
```

## Relationship to downstream consumers

The most important downstream consumer is the log-mel builder.

That downstream stage expects the segment-level final manifest and relies on fields such as:

1. `sample_id`
2. `filepath`
3. `genre_top`
4. `segment_start_sec`
5. `sample_length_sec`
6. `final_split`
7. `reason_code`

As a result, `build_all_datasets_and_samples.py` defines the ground truth for which audio segments enter later feature extraction and model training.

## Practical summary

If you want to understand this script operationally, think of it as three things at once:

1. a dataset discovery tool
2. a rules engine for audio eligibility and fixed-length sampling
3. a manifest compiler that turns raw audio collections into deterministic training inputs

Its main contract is simple:

1. `manifest_all_datasets.parquet` tells you what exists and why it is usable or skipped
2. `manifest_all_samples.parquet` tells you which fixed-length segments can be generated
3. `manifest_final_samples.parquet` tells downstream stages exactly which segment rows to use
