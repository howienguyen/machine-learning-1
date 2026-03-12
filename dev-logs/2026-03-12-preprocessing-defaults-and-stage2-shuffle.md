# Dev Log — 2026-03-12 — Preprocessing Defaults, Stage 2 Additional Shuffle, and `logmel_cnn_v2.py` Dataset Root Alignment

## Scope

This log covers a small repo-wide alignment pass across preprocessing, training defaults, and the main pipeline docs.

Updated artifacts:

1. `MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py`
2. `MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py`
3. `MelCNN-MGR/notebooks/logmel_cnn_v2.py`
4. `docs/MelCNN-MGR-1_build_all_datasets_and_samples.md`
5. `docs/MelCNN-MGR-Production-Like-Pipeline.md`
6. `dev-logs/2026-03-12-preprocessing-defaults-and-stage2-shuffle.md`

## Summary

Three related behaviors are now aligned and documented consistently:

1. Stage 1 falls back to `15.0` only when `sample_length_sec` is missing or invalid inside an otherwise valid `data_sampling_settings` object.
2. Stage 2 uses settings only for default output-root naming, with a `15.0` fallback if settings cannot be read, while keeping runtime extraction length manifest-driven unless `--sample-length-sec` is provided.
3. Stage 2 now deterministically shuffles the additional-source sample manifest before grouped final split assignment.

## Change 1 — Field-Level `sample_length_sec` Fallback in Stage 1

`1_build_all_datasets_and_samples_v1_1.py` keeps settings-file readability and `target_genres` validation strict, but no longer fails just because `data_sampling_settings.sample_length_sec` is missing or invalid.

Instead, it now:

1. uses the configured value when it is numeric and positive
2. falls back to `DEFAULT_SAMPLE_LENGTH_SEC = 15.0` otherwise
3. logs a warning so the fallback is visible during runs

## Change 2 — Safe Settings-Derived Default for Stage 2 Cache Root

`2_build_log_mel_dataset.py` previously read settings at import time and looked for `sample_length_sec` at the wrong JSON level.

It now uses a small helper that:

1. reads `settings.data_sampling_settings.sample_length_sec`
2. returns that value when valid
3. falls back to `15.0` on any read or validation failure

That helper is used only for the default output-root suffix.

## Change 3 — Explicit Stage 2 Additional-Source Shuffle

Stage 2 already used seeded randomness during grouped final split assignment, but the additional sample manifest itself was previously loaded in deterministic sorted order.

The pipeline now makes this ingress behavior explicit:

1. `manifest_additional_all_samples.parquet` is deterministically shuffled with `--split-seed` before Stage 2 grouped selection
2. grouped split assignment still happens at source-audio identity level
3. all segments from the same source audio still stay in the same final split

## Change 4 — `logmel_cnn_v2.py` Default Dataset Root Now Follows Settings

`logmel_cnn_v2.py` no longer hard-codes `logmel_dataset_15s` as its default cache root.

It now derives the default dataset suffix from `settings.data_sampling_settings.sample_length_sec`, with a `15.0` fallback if settings cannot be read.

The actual clip duration used for training still comes from the loaded log-mel manifests.

## Documentation Alignment

The main pipeline docs were updated to reflect the current repo state:

1. the manifest-builder script name is now `1_build_all_datasets_and_samples_v1_1.py`
2. the per-source Stage 1a and Stage 1b parquet outputs are documented instead of the older combined `manifest_all_*` naming
3. the Stage 2 additional-source shuffle behavior is called out explicitly
4. the log-mel builder now documents settings-driven default cache-root naming with manifest-driven runtime length
5. the configuration reference now matches the current `settings.json` snapshot instead of stale `10s` examples

## Validation Status

Validated in this session:

1. static editor validation for the updated preprocessing scripts
2. static editor validation for `logmel_cnn_v2.py`, aside from pre-existing TensorFlow import-resolution warnings in the editor environment

Not yet done:

1. a full end-to-end rerun of Stage 1 -> Stage 2 -> log-mel build using the updated defaults
2. an empirical comparison showing whether the Stage 2 additional-source shuffle changes the exact final selected rows under the current dataset state