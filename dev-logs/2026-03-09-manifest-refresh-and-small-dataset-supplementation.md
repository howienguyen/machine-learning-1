# Dev Log — 2026-03-09 — Manifest Refresh and Small-Dataset Supplementation

## Scope

This log consolidates the recent preprocessing and dataset-understanding changes made
around the FMA small/medium workflow, supplementation planning, and the current run
sequence expected before the `baseline_logmel_cnn_v20a.py`, `baseline_logmel_cnn_v20a1.py`,
and `baseline_logmel_cnn_v21.py` notebook-scripts are executed.

Covered changes:

1. Reviewed and updated the manifest pipeline assumptions.
2. Reworked the supplementation design from in-memory oversampling to post-manifest supplementation.
3. Expanded the data-understanding notebooks for exact-vs-cumulative subset analysis.
4. Added `collect_extra_samples_for_small_dataset.py`.
5. Fixed exact-small exclusion logic in the collector.
6. Moved supplementation defaults into `MelCNN-MGR/settings.json`.
7. Updated the collector to recover FMA-medium filepaths from `metadata_manifest_medium.parquet`.
8. Verified that manifest outputs already expose `sample_id` and `source`.
9. Added `load_extra_samples_for_small_dataset_splits.py`.
10. Validated deterministic split loading and rerun-safe deduplication.
11. Fixed `build_manifest.py` so omitted `--audio-root` now follows `--subset`.
12. Added a supplementation-aware split-distribution notebook for small-split EDA.

---

## Change 1 — Manifest Identity Columns Are Now First-Class

### Summary
`MelCNN-MGR/preprocessing/build_manifest.py` now writes explicit provenance columns into
both the full manifest and the split parquet files.

### Current output model

- `sample_id` is a globally unique identifier in the form `source:track_id`
- `source` records the dataset name, currently `fma` for the base manifests
- split parquet files inherit these columns directly

### Why this matters

The project is moving toward multi-source supplementation, so `track_id` alone cannot
be treated as globally unique once non-FMA sources are added.

---

## Change 2 — Supplementation Design Moved After Manifest Generation

### Old idea
Class balancing was originally discussed as an in-memory oversampling step inside a
single training script.

### New design
The design was rewritten so supplementation happens after `build_manifest.py` and is
treated as a dataset-construction step rather than a model-specific training trick.

### Key design decisions

- identify underrepresented classes by ratio rule rather than blind duplication
- prefer real tracks from other sources first
- duplicate only as a fallback
- apply supplementation to training only by default

This design is documented in:

- `docs/Development_Guidelines_Oversampling_for_baseline_logmel_cnn_v21.md`

---

## Change 3 — Data-Understanding Notebook Upgrades

Two notebooks were updated to make subset semantics explicit and to support broader EDA.

### `MelCNN-MGR/notebooks/Data-Understanding-Train-Val-Test-Genre-Distribution.ipynb`

Added or corrected:

- configurable subset filtering
- exact vs cumulative subset mode
- top-N genre filtering
- broader whole-dataset genre and split EDA
- refreshed result interpretation text to match the new controls

### `MelCNN-MGR/notebooks/Data-Understanding-Small-vs-Medium-Genre-Expansion.ipynb`

Added comparison views for:

- exact small vs cumulative medium
- genres present in medium but absent from small
- exact large vs exact medium

### Why this matters

The supplementation workflow depends on understanding which labels are absent or weakly
represented in exact small, not just in cumulative subset views.

---

## Change 4 — Added `collect_extra_samples_for_small_dataset.py`

### Purpose
This script collects candidate extra samples for the small dataset from multiple sources.

### Sources scanned

1. exact-medium FMA tracks not already present in exact small
2. `additional_datasets/dortmund-university`
3. `additional_datasets/gtzan`

### Output

- `MelCNN-MGR/data/processed/extra_samples_for_small_dataset.json`

### Contents of the JSON report

- candidate rows grouped by source and genre
- capped selected rows per target genre
- summary counts and shortfalls
- folder coverage notes for Dortmund and GTZAN

---

## Change 5 — Fixed the Exact-Small Exclusion Bug

### Problem
The first collector implementation treated `metadata_manifest_small.parquet` as though
it were a table of exact-small tracks only.

### Root cause
`metadata_manifest_small.parquet` is a full manifest with reason codes and inclusion state,
not an exact-membership list.

### Fix
The collector now filters the manifest to `subset == "small"` before building the set of
track IDs to exclude.

### Impact

Without this fix, the collector would exclude the wrong rows and misreport available
medium candidates.

---

## Change 6 — Supplementation Defaults Moved to Settings

`MelCNN-MGR/settings.json` now carries the small-dataset supplementation defaults.

### Current keys

```json
{
  "small_dataset_supplementation": {
    "target_genres": [
      "Hip-Hop",
      "Pop",
      "Folk",
      "Rock",
      "Electronic",
      "Classical",
      "Jazz",
      "Country",
      "Blues"
    ],
    "n_extra_expected": 300,
    "train_n_val_test_split_ratio": 0.8
  }
}
```

### Current usage status

- `target_genres` is consumed by the collector
- `n_extra_expected` is consumed by the collector
- `train_n_val_test_split_ratio` is present in settings but is not yet consumed by a supplementation script

---

## Change 7 — FMA-Medium Filepaths Now Come from the Medium Manifest

### Problem
Some `fma-medium` rows in `extra_samples_for_small_dataset.json` previously had
`"filepath": null`.

### Fix
The collector now loads `metadata_manifest_medium.parquet` and uses its `filepath`,
`audio_exists`, and `reason_code` columns for exact-medium candidates.

### Why this is the right source

`metadata_manifest_medium.parquet` is already the reconciled source of truth after
filesystem resolution, so it is safer than trying to reconstruct paths ad hoc.

---

## Change 8 — Current Rerun Workflow Before Training

The recommended order before running the main log-mel notebook-scripts is now:

1. remove old derived processed artifacts and feature caches
2. rebuild `small` manifest with `FMA/fma_small`
3. rebuild `medium` manifest with `FMA/fma_medium`
4. rerun `collect_extra_samples_for_small_dataset.py`
5. run `load_extra_samples_for_small_dataset_splits.py` to build updated `small` split parquets
6. only then run the training notebook-script

### Current CLI behavior

`build_manifest.py` now derives the default audio root from `--subset` when
`--audio-root` is omitted.

That means:

- `--subset small` -> `FMA/fma_small`
- `--subset medium` -> `FMA/fma_medium`
- `--subset large` -> `FMA/fma_large`

An explicit `--audio-root` still overrides the derived path.

---

## Change 9 — Added `load_extra_samples_for_small_dataset_splits.py`

### Purpose

This new preprocessing script materializes the selected rows from
`extra_samples_for_small_dataset.json` into the `small` train/val/test parquet files.

### Inputs

- `MelCNN-MGR/settings.json`
- `MelCNN-MGR/data/processed/extra_samples_for_small_dataset.json`
- `MelCNN-MGR/data/processed/metadata_manifest_medium.parquet`
- `MelCNN-MGR/data/processed/train_small.parquet`
- `MelCNN-MGR/data/processed/val_small.parquet`
- `MelCNN-MGR/data/processed/test_small.parquet`

### Current behavior

- uses `target_genres` from settings to decide which genres to supplement
- caps each genre at `n_extra_expected`
- applies a deterministic per-genre shuffle before allocation
- uses `train_n_val_test_split_ratio` for train allocation
- splits the remainder equally between validation and test
- keeps FMA-medium rows on their original integer `track_id`
- assigns deterministic negative integer surrogate `track_id` values to external rows
- preserves multi-source provenance via `sample_id` and `source`

### Why surrogate IDs were needed

External datasets do not carry FMA track IDs, but the existing split parquet schema uses
an integer `track_id` index.  The loader therefore synthesizes stable integer surrogates
for external rows while keeping source provenance in `sample_id`.

## Change 10 — Validation and Rerun Safety

The new loader script was validated in a temporary output directory before touching the
live processed parquet files.

### Validated allocation on the current selection payload

- Hip-Hop: 300 selected -> train 240, val 30, test 30
- Pop: 300 selected -> train 240, val 30, test 30
- Folk: 300 selected -> train 240, val 30, test 30
- Rock: 300 selected -> train 240, val 30, test 30
- Electronic: 300 selected -> train 240, val 30, test 30
- Classical: 300 selected -> train 240, val 30, test 30
- Jazz: 300 selected -> train 240, val 30, test 30
- Country: 278 selected -> train 222, val 28, test 28
- Blues: 294 selected -> train 236, val 29, test 29

### Aggregate validated additions

- train: 2138 rows added
- validation: 267 rows added
- test: 267 rows added

### Idempotence fix and final validation

The first implementation only deduplicated by synthesized surrogate `track_id`, which
was insufficient for reruns because a second execution could mint a fresh surrogate for
the same external file.

The loader now deduplicates external rows by natural source identity:

- `source + filepath` for external datasets
- `source + track_id` for FMA-medium rows

After that fix, a second rerun against already supplemented temporary outputs added:

- train: 0 rows
- validation: 0 rows
- test: 0 rows

and skipped 2672 already loaded rows.

## Additional Note — Cache Keys Still Need Attention

The current log-mel feature caches in the baseline notebook-scripts still name cached
`.npy` files by `track_id`, so cache invalidation remains necessary before multi-source
experiments unless those cache keys are upgraded to use `sample_id`.

---

## Change 11 — `build_manifest.py` Now Derives `audio_root` From `subset`

### Problem

The CLI previously accepted `--subset medium` while still defaulting `--audio-root`
to the small-audio directory unless the caller overrode it explicitly.

### Failure mode

That made the following command hazardous:

```bash
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium
```

because Phase B would resolve filepaths under `FMA/fma_small`, causing in-subset
medium rows to fall to `NO_AUDIO_FILE` and yielding zero usable medium samples.

### Fix

The builder now resolves the default audio directory from the requested subset at runtime.

### Result

These forms are now equivalent for the standard FMA layout:

```bash
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium --audio-root FMA/fma_medium
```

with the explicit form still available for custom layouts.

---

## Change 12 — Added a Supplementation-Aware Split EDA Notebook

### New notebook

- `MelCNN-MGR/notebooks/Data-Understanding-Train-Val-Test-Genre-Distribution-Supplementation-Aware.ipynb`

### Why it was added

The original split-distribution notebook is still useful as a generic train/validation/test
sanity check, but it should not be overloaded with workflow-specific supplementation logic.

The new notebook isolates the small-dataset supplementation questions:

- official exact-small split counts from `tracks.csv`
- current processed `train_small` / `val_small` / `test_small` parquet contents
- projected additions implied by `extra_samples_for_small_dataset.json`
- source-mix and drift changes after materializing the current payload

### Current observed state

At the time this notebook was added, the live `small` split parquets still matched the
official 8-genre FMA-small baseline, while the JSON payload described the projected
supplementation additions separately.

That is why the notebook compares three states rather than assuming the live parquets are
already supplemented:

- official exact-small
- current processed splits
- projected after current payload

---

## Validation Snapshot

The collector was rerun successfully after the earlier fixes and settings integration.
It produced non-empty selections for all configured target genres, with the expected
shortfall only in classes where all scanned sources still did not provide the full target.

Representative outcomes from the latest validated run:

- Hip-Hop: selected 300
- Pop: selected 300
- Folk: selected 300
- Rock: selected 300
- Electronic: selected 300
- Classical: selected 300
- Jazz: selected 300
- Country: selected 278
- Blues: selected 294

These counts reflect the candidate-selection stage before split loading. The validated
split-loader stage is documented above and is now part of the current workflow.