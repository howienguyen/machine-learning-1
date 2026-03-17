# Dev Log — 2026-03-17 — TFRecord Conversion and `logmel_cnn_v2_4_cuda_tf.py`

## Scope

This session added a new TFRecord conversion stage for the prebuilt log-mel
dataset and introduced a TFRecord-based training entry point derived from the
v2.3 CUDA trainer.

Updated artifacts:

1. `MelCNN-MGR/preprocessing/3_convert_npy_2_tfrecord.py`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_4_cuda_tf.py`
3. `dev-logs/2026-03-17-tfrecord-conversion-and-logmel-cnn-v2_4-cuda-tf.md`

## Summary

The existing training flow used per-sample `.npy` log-mel files loaded through
`tf.numpy_function + np.load(...)`. That works functionally, but it keeps a lot
of the input path in Python space and tends to interact poorly with large
numbers of small files, background buffering, and TensorFlow input-pipeline
heuristics.

To move the pipeline toward a more TensorFlow-native data path, a new
conversion script now packages the existing split-grouped `.npy` dataset into
split-sharded TFRecords. A new training script, `logmel_cnn_v2_4_cuda_tf.py`,
was then created by cloning the v2.3 CUDA trainer and refactoring it to read
those TFRecord shards directly.

The result is a training entry point that preserves the existing model,
regularization, warm-start logic, and train-only per-mel-bin normalization,
while switching the input pipeline from per-sample NumPy file reads to
`tf.data.TFRecordDataset`.

## Change 1 — New `3_convert_npy_2_tfrecord.py` Conversion Stage

`MelCNN-MGR/preprocessing/3_convert_npy_2_tfrecord.py` was added as a standalone
preprocessing step after `2_build_log_mel_dataset.py`.

Its default behavior is:

1. input root:
   `/home/hsnguyen/model-training-data-cache/logmel_dataset_<N>s`
2. output root:
   `/home/hsnguyen/model-training-data-cache/logmel_dataset_<N>s_tfrecord`

It reads the split parquet manifests and source `.npy` paths from the existing
log-mel dataset root, then writes TFRecords under mirrored split directories:

1. `train/`
2. `val/`
3. `test/`

Each split is written as one or more shard files such as:

1. `train/train-00000-of-00008.tfrecord`
2. `val/val-00000-of-00001.tfrecord`
3. `test/test-00000-of-00001.tfrecord`

## Change 2 — TFRecord Record Schema

Each record currently stores:

1. `logmel` as a serialized TensorFlow tensor
2. `label` as an integer class id
3. `sample_id`
4. `genre_top`
5. `split_dir`
6. `source_logmel_relpath`

The serialized tensor preserves the precomputed fixed-shape float32 log-mel
array while keeping the record format straightforward to parse with TensorFlow
ops.

## Change 3 — TFRecord-Side Metadata and Manifests

The conversion script also writes metadata and discovery artifacts so training
code does not have to inspect the TFRecord directory ad hoc.

Written outputs include:

1. `tfrecord_config.json`
2. `tfrecord_manifest_all.parquet`
3. `tfrecord_manifest_train.parquet`
4. `tfrecord_manifest_val.parquet`
5. `tfrecord_manifest_test.parquet`
6. `tfrecord_shards_all.parquet`
7. `tfrecord_shards_train.parquet`
8. `tfrecord_shards_val.parquet`
9. `tfrecord_shards_test.parquet`
10. `tfrecord_build_report.txt`

That gives downstream training scripts a clear source of truth for:

1. feature shape
2. compression mode
3. genre-to-label mapping
4. sample counts
5. shard counts
6. shard file paths

## Change 4 — New `logmel_cnn_v2_4_cuda_tf.py` Training Entry Point

`MelCNN-MGR/model_training/logmel_cnn_v2_4_cuda_tf.py` was created by cloning
`logmel_cnn_v2_3_cuda.py`.

The core model and training logic remain intentionally close to v2.3:

1. same CNN backbone
2. same optimizer schedule
3. same warm-start handling
4. same train-only per-mel-bin standardization
5. same fixed `tf.data` buffering settings introduced in the v2.3 lineage

The major change is the input path.

## Change 5 — v2.4 Now Reads TFRecord Manifests and Config

The v2.4 trainer no longer points at the log-mel `.npy` dataset root by
default.

Instead it resolves:

1. `TFRECORD_DATASET_DIR`
2. `tfrecord_manifest_train.parquet`
3. `tfrecord_manifest_val.parquet`
4. `tfrecord_manifest_test.parquet`
5. `tfrecord_shards_train.parquet`
6. `tfrecord_shards_val.parquet`
7. `tfrecord_shards_test.parquet`
8. `tfrecord_config.json`

The trainer now extracts feature shape and compression settings from the
TFRecord-side config instead of relying on the old log-mel config file.

## Change 6 — TFRecord Parsing Replaces `np.load` Dataset Reads

The previous v2.3 dataset path used:

1. `tf.numpy_function(...)`
2. `np.load(...)`
3. per-sample file reads from the `.npy` tree

In v2.4 this is replaced by:

1. `tf.data.TFRecordDataset(...)`
2. `tf.io.parse_single_example(...)`
3. `tf.io.parse_tensor(...)`
4. TensorFlow-side normalization and batching

This keeps the actual record decode inside TensorFlow once the record bytes are
read, instead of bouncing out to Python for every sample.

## Change 7 — Train-Only Per-Mel-Bin Stats Still Preserved

The v2.4 trainer keeps the same per-mel-bin standardization experiment as v2.3.

The difference is how the training statistics are computed:

1. v2.3 streamed over `train_df["logmel_path"]` and loaded `.npy` files
2. v2.4 streams over parsed training TFRecord examples

The logic still uses two passes over the training split only:

1. first pass for per-bin means
2. second pass for per-bin variance/std

So the normalization experiment remains the same even though the storage and
I/O format changed.

## Change 8 — Sample Inspection and Evaluation Path Updated

The sample sanity-check section in v2.4 now loads one example from a TFRecord
shard instead of reading a `.npy` file directly.

The evaluation pipeline continues to consume `tf.data.Dataset` objects, but now
those datasets are all built from TFRecord shards for:

1. train clean-eval
2. validation
3. test

The metric computation and reporting code remain otherwise unchanged.

## Change 9 — Run Metadata Updated for TFRecord Experiments

The v2.4 run report now records TFRecord-specific dataset metadata, including:

1. TFRecord dataset root
2. source log-mel root
3. split manifest paths
4. split shard-manifest paths
5. dataset format = `tfrecord`
6. compression mode
7. shard counts per split
8. fixed `tf.data` settings including parallel reads, parallel calls, and
   prefetch size

This matters because the input format is now part of the experiment definition,
not just an implementation detail.

## Validation Status

Validated in this session:

1. `MelCNN-MGR/preprocessing/3_convert_npy_2_tfrecord.py` passed static error checks
2. `MelCNN-MGR/model_training/logmel_cnn_v2_4_cuda_tf.py` passed static error checks
3. the v2.4 trainer no longer contains the old `.npy` data-loader path for
   training/evaluation
4. run-report fields were updated to reflect the fixed `tf.data` settings and
   TFRecord dataset metadata

Not yet validated end-to-end:

1. a full TFRecord conversion run over the current cached log-mel dataset
2. a training smoke test using `logmel_cnn_v2_4_cuda_tf.py`
3. throughput comparison between v2.3 `.npy` input and v2.4 TFRecord input
4. runtime profiling to confirm whether the observed batch stalls reduce under
   the new input format