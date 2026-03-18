# Dev Log — 2026-03-18 — Validating `logmel_cnn_v2_4_1_cuda_tf.py`

## Scope

This session validated the latest TFRecord trainer end-to-end and recorded the
runtime settings that proved stable on the current WSL2 + CUDA stack.

Updated artifacts:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_4_1_cuda_tf.py`
2. `MelCNN-MGR/demo-models/logmel-cnn-v2_4-cuda-tf-20260317-151350/`
3. `MelCNN-MGR/inference_web_service/app.py`
4. `dev-logs/2026-03-18-logmel-cnn-v2_4_1-cuda-tf-validation.md`

## Summary

The latest `logmel_cnn_v2_4_1_cuda_tf.py` configuration completed a full
training run successfully and produced a new demo-model export rooted at:

1. `MelCNN-MGR/demo-models/logmel-cnn-v2_4-cuda-tf-20260317-151350/`

This closes the open validation gap from the earlier v2.4 TFRecord and
stabilization notes. The trainer now has a verified end-to-end run with:

1. train-only per-mel-bin normalization artifacts
2. TFRecord input manifests and shard reads
3. fixed dataset map/read parallelism
4. configurable prefetch/autotune RAM budget settings
5. exported `.keras` checkpoints and run report artifacts

## Stable Runtime Configuration

The successful run used the following dataset/runtime settings:

1. `dataset_parallelism_mode = "fixed"`
2. `dataset_num_parallel_calls = 6`
3. `dataset_num_parallel_reads = 6`
4. `dataset_prefetch_mode = "autotune"`
5. `dataset_autotune_ram_budget_bytes = 2147483648` (2 GiB)

This combination ran stably on the current environment:

1. WSL2
2. Python 3.12.3
3. TensorFlow 2.20.0
4. CUDA backend on NVIDIA GeForce RTX 3090

During earlier testing, enabling `tf.data.AUTOTUNE` for TFRecord map/read
parallelism caused a TensorFlow `device_event_mgr.cc:226 Unexpected Event
status: 1` abort. The validated configuration keeps map/read parallelism fixed
and limits autotuning to prefetch plus an explicit RAM budget.

## Training Outcome

The validated run report is:

1. `MelCNN-MGR/demo-models/logmel-cnn-v2_4-cuda-tf-20260317-151350/run_report_logmel_cnn_v2_4_1_cuda_tf.json`

Key outcomes from that run:

1. epochs requested: `136`
2. epochs completed: `107`
3. batch size: `48`
4. best validation macro-F1 epoch: `98`
5. best validation loss epoch: `103`

Final reported metrics:

1. train: accuracy `0.9522`, macro-F1 `0.9527`
2. validation: accuracy `0.8399`, macro-F1 `0.8406`
3. test: accuracy `0.8484`, macro-F1 `0.8468`

These results materially improve on the earlier v2.2 reference while preserving
the TFRecord input path and the per-mel-bin normalization workflow.

## Exported Artifacts

The new demo-model directory contains:

1. `best_model_macro_f1.keras`
2. `logmel_cnn_v2_4_1_cuda_tf.keras`
3. `norm_stats.npz`
4. `normalization_stats.json`
5. `console_output.txt`
6. `run_report_logmel_cnn_v2_4_1_cuda_tf.json`

The normalization metadata confirms that statistics were computed from the
train split only and saved alongside the checkpoint for inference reuse.

## Inference Adoption

The inference web service default model target was updated to point at the new
validated export via `MODEL_NAME = "logmel-cnn-v2_4-cuda-tf-20260317-151350"`
in `MelCNN-MGR/inference_web_service/app.py`.

That change aligns the demo/inference path with the newly validated training
output rather than leaving the service on an older v2.2 export.

## Validation Status

Validated in this session:

1. `logmel_cnn_v2_4_1_cuda_tf.py` completed end-to-end training successfully
2. the TFRecord input path worked through normalization, training, validation,
   test evaluation, and artifact export
3. the latest exported model was promoted into the inference web service
4. the dataset prefetch/autotune RAM budget configuration worked with fixed
   map/read parallelism on the current CUDA stack

Residual note:

1. `tf.data.AUTOTUNE` remains unsuitable for TFRecord map/read parallelism on
   this stack even though prefetch autotuning is now usable with an explicit RAM
   budget