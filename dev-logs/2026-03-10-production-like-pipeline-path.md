# Dev Log — 2026-03-10 — Production-Like Pipeline Path Clarification

## Scope

This log records the canonical production-like path for the current MelCNN-MGR multi-source workflow.

Updated documentation targets:

1. `docs/MelCNN-MGR-Preprocessing.md`
2. `docs/MelCNN-MGR-1_build_all_datasets_and_samples.md`
3. `docs/Development Guidelines Implementing New MelCNN_MGR_EDA_and_Data_Understanding.md`

## Canonical pipeline

The production-like path should now be understood as two connected chains.

### Upstream MTG/Jamendo acquisition chain

```text
utils/download_by_genre_limits.py
    -> extract_mtg_processed_samples.py
```

This chain is responsible for obtaining and preparing additional MTG/Jamendo audio under `additional_datasets/data` so that it can later be scanned by the MelCNN-MGR manifest builder.

### Main MelCNN-MGR data and training chain

```text
data sources: FMA + additional_datasets
    -> MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples.py
    -> MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py
    -> MelCNN-MGR/model_training/MelCNN_MGR_Manifest_LogMel_EDA.ipynb
    -> MelCNN-MGR/model_training/logmel_cnn_v1.py
```

## Why this clarification matters

Several older documents still describe parts of the project as if the primary path were only:

1. FMA-only
2. metadata-manifest-only
3. legacy notebook-centric

That is no longer the right default framing.

The current default path is:

1. multi-source at data intake time
2. manifest-centric at dataset-construction time
3. prebuilt-logmel-centric at model-training time
4. notebook-audited through the manifest/log-mel EDA notebook

## Practical interpretation

### What `utils/download_by_genre_limits.py` and `extract_mtg_processed_samples.py` do

They prepare additional-source audio before MelCNN-MGR preprocessing begins.

### What `1_build_all_datasets_and_samples.py` does

It is the dataset-contract stage. It merges FMA and additional-source audio into the shared manifest tables:

1. `manifest_all_datasets.parquet`
2. `manifest_all_samples.parquet`
3. `manifest_final_samples.parquet`

### What `2_build_log_mel_dataset.py` does

It is the downstream feature-building stage that consumes `manifest_final_samples.parquet` and writes the log-mel cache plus split parquet indexes.

### What `MelCNN_MGR_Manifest_LogMel_EDA.ipynb` does

It is the current data-understanding and audit notebook for the manifest-based pipeline.

### What `logmel_cnn_v1.py` does

It is the current prebuilt-logmel training entry that consumes the split parquet indexes emitted by the log-mel builder.

## Documentation consequence

When describing the main MelCNN-MGR workflow going forward, documents should treat the above chain as the default production-like path unless they are explicitly discussing a legacy branch or a baseline-specific historical notebook.