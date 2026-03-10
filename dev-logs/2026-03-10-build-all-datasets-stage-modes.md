# Dev Log — 2026-03-10 — Stage Modes for `1_build_all_datasets_and_samples.py`

## Scope

This change formalizes the existing two-stage manifest pipeline into an explicit CLI contract.

Updated artifacts:

1. `MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py`
2. `README.md`
3. `docs/MelCNN-MGR-1_build_all_datasets_and_samples.md`

## Summary

`1_build_all_datasets_and_samples.py` now supports a `--mode` flag with three values:

1. `stage1`
2. `stage2`
3. `both`

This removes the previous limitation where the script always executed Stage 1 and Stage 2 back-to-back.

## New CLI behavior

### `--mode stage1`

Runs only the intermediate-manifest steps:

1. build or load FMA candidates
2. scan additional datasets
3. combine dataset rows into `manifest_all_datasets.parquet`
4. expand eligible rows into `manifest_all_samples.parquet`

Outputs:

1. `manifest_all_datasets.parquet`
2. `manifest_all_samples.parquet`
3. `manifest_all_datasets.report.txt`
4. `manifest_all_datasets.config.json`

### `--mode stage2`

Runs only final split assignment.

Input requirement:

1. an existing Stage 1 sample manifest at the path passed through `--all-samples-out`

Outputs:

1. `manifest_final_samples.parquet`
2. `manifest_final_samples.report.txt`
3. `manifest_final_samples.config.json`

### `--mode both`

Preserves the original end-to-end behavior.

Outputs:

1. `manifest_all_datasets.parquet`
2. `manifest_all_samples.parquet`
3. `manifest_final_samples.parquet`
4. `manifest_all_datasets.report.txt`
5. `manifest_all_datasets.config.json`

## Implementation notes

### Stage 2 loader

A new helper loads and validates the Stage 1 sample parquet before split assignment:

1. checks that the file exists
2. checks required columns such as `sample_id`, `duration_s`, `actual_duration_s`, and `reason_code`
3. restores `track_id` and `total_segments_from_audio` to nullable integer dtype

### Logging and reports

Summary logging and report generation were made mode-aware.

That was necessary because a Stage 2-only run has no `all_datasets` dataframe in memory, while a Stage 1-only run has no final split summary.

## Why this matters

This makes the manifest workflow operationally cleaner:

1. Stage 1 can be rerun when source discovery or duration probing changes
2. Stage 2 can be rerun cheaply when only split assignment settings change
3. the CLI now matches the conceptual model already documented for the pipeline

## Example commands

Stage 1 only:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py --mode stage1
```

Stage 2 only:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py --mode stage2
```

Both stages:

```bash
python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py --mode both
```