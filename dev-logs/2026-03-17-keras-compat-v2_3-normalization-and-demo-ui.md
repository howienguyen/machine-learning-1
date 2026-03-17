# Dev Log — 2026-03-17 — Keras Compatibility, `logmel_cnn_v2_3_cuda.py`, and Demo UI Fixes

## Scope

This session covered four related updates across training, inference, and the
demo client:

1. cross-version Keras compatibility for loading newer `.keras` checkpoints in
   the inference stack
2. a new training entry point `MelCNN-MGR/model_training/logmel_cnn_v2_3_cuda.py`
   for explicit train-only per-mel-bin standardization
3. inference-side normalization metadata loading and reporting for the v2.3
   artifact format
4. a frontend rendering fix in the demo audio-capture app so inference results
   actually appear on screen

Updated artifacts:

1. `MelCNN-MGR/model_inference/inference_logmel_cnn_v2_x.py`
2. `MelCNN-MGR/inference_web_service/app.py`
3. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`
4. `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`
5. `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py`
6. `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`
7. `MelCNN-MGR/model_training/logmel_cnn_v3_1.py`
8. `MelCNN-MGR/model_training/logmel_cnn_v2_3_cuda.py`
9. `dev-logs/2026-03-17-keras-compat-v2_3-normalization-and-demo-ui.md`

## Summary

The existing inference service failed to load the demo `logmel-cnn-v2_2` model
because the checkpoint had been saved by a newer Keras serialization stack than
the one available in the `.venv311` inference environment. The failure first
appeared as a missing internal module path for `keras.src.models.functional`,
but further inspection showed the incompatibility was broader and included
Keras 3-style layer config fields and Functional graph serialization.

To make inference robust, the loader now uses a compatibility strategy with
three layers:

1. direct `load_model(..., compile=False)` when possible
2. internal module-path aliasing and archive-level config rewrites for simple
   compatibility breaks
3. a last-resort manual rebuild of the known v2.x CNN architecture directly
   from archive JSON plus `model.weights.h5`

That allowed the current `MelCNN-MGR/inference_web_service/app.py` stack to load
the trained demo model successfully and answer `GET /health` in the target
Python 3.11 / TensorFlow-Keras 2.15 environment.

In parallel, a new `logmel_cnn_v2_3_cuda.py` training script was created by
cloning the v2.2 CUDA trainer and making the per-mel-bin standardization
experiment explicit, reproducible, and surfaced through saved artifacts.

## Change 1 — Cross-Version Keras Compatibility in `inference_logmel_cnn_v2_x.py`

The inference module now handles `.keras` archives saved by newer Keras builds
more defensively.

Main additions:

1. a fallback-safe registration path for custom learning-rate schedules
2. internal module aliasing for moved Keras internals such as
   `keras.src.models.functional`
3. compatibility rewrites for selected newer config fields
4. a manual architecture rebuild path for the known v2.x CNN family when full
   graph deserialization is not possible

This matters because the original failure was not just one missing import.
The archive also carried newer InputLayer/config conventions and a Functional
graph format that older TF/Keras builds do not deserialize cleanly.

## Change 2 — Service Startup Validated Against the Existing Demo Model

After the compatibility work, `MelCNN-MGR/inference_web_service/app.py` was
verified to:

1. initialize `LogMelCNNV2XInference` successfully for
   `MelCNN-MGR/demo-models/logmel-cnn-v2_2-20260316-145637-bolero`
2. start uvicorn successfully in the `.venv311` inference environment
3. respond successfully to `GET /health`

The remaining noisy startup output came from TensorFlow device/runtime probing,
not from model-deserialization failure.

## Change 3 — Backward-Compatible `register_keras_serializable` Usage

Several training scripts still used `@tf.keras.saving.register_keras_serializable(...)`.
That breaks in environments where `tf.keras.saving` is not available.

The affected scripts were updated to resolve the decorator through a fallback:

1. prefer `tf.keras.saving.register_keras_serializable` when present
2. otherwise use `tf.keras.utils.register_keras_serializable`

Updated scripts:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py`
3. `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`
4. `MelCNN-MGR/model_training/logmel_cnn_v3_1.py`

This keeps training entry points runnable across the current project
environments without hard-coding one Keras API surface.

## Change 4 — Demo App Inference Rendering Fix

`MelCNN-MGR/demo-app/web_audio_capture_v1.py` was reviewed because the backend
logs clearly showed successful `partial_result` responses, but the browser UI
stayed stuck at “Waiting for inference...”.

The root cause was frontend-side, not backend-side:

1. the polling UI wrote to DOM nodes that were missing because related HTML
   sections were commented out
2. the resulting JavaScript exception was swallowed by an empty polling `catch`
3. once the exception fired, the visible partial-prediction update code never
   finished running

The fix added:

1. null-guards around optional DOM-node updates
2. `console.error(...)` in the inference polling path instead of silent failure

This allows visible result cards to keep updating even if optional UI panels are
hidden or commented out.

## Change 5 — New `logmel_cnn_v2_3_cuda.py` Training Entry Point

`MelCNN-MGR/model_training/logmel_cnn_v2_3_cuda.py` was created by cloning the
v2.2 CUDA trainer and then making the normalization experiment explicit.

Main changes in v2.3:

1. run family updated to `logmel-cnn-v2_3`
2. output model name updated to `logmel_cnn_v2_3`
3. run report path updated to `run_report_logmel_cnn_v2_3.json`
4. confusion-matrix labeling updated to v2.3 naming

Functionally, v2.3 now frames the preprocessing as a deliberate train-only
per-mel-bin standardization experiment rather than leaving it implicit behind
generic `mu/std` naming.

## Change 6 — Explicit Train-Only Per-Mel-Bin Standardization in v2.3

The new v2.3 trainer computes normalization statistics using a two-pass,
streaming, train-split-only routine.

Implementation details:

1. pass 1 accumulates per-mel-bin sums and counts over all training samples and
   frames
2. pass 2 accumulates per-mel-bin squared error using the pass-1 means
3. `std_per_bin` is clamped with `NORMALIZATION_EPS = 1e-6`
4. normalization is applied on the fly in the dataset loader as:
   `(x - mean_per_bin[:, None]) / std_per_bin[:, None]`

This matches the intended experiment design while keeping memory usage bounded.

## Change 7 — v2.3 Normalization Artifacts and Run Metadata

The v2.3 trainer now saves normalization more explicitly.

`norm_stats.npz` now contains:

1. legacy-compatible `mu`
2. legacy-compatible `std`
3. explicit `mean_per_bin`
4. explicit `std_per_bin`
5. `genre_classes`

It also writes:

1. `normalization_stats.json`

with metadata such as:

1. normalization type
2. `computed_from = train_split_only`
3. epsilon
4. number of mel bins

The run report for v2.3 also now includes a dedicated `normalization` section
recording shapes, ranges, and artifact paths.

## Change 8 — Inference Loader Now Reads `mean_per_bin/std_per_bin` Explicitly

The inference module was extended so it can prefer explicit per-bin statistics
when present in `norm_stats.npz`.

Behavior now is:

1. if `mean_per_bin/std_per_bin` exist, use them as the canonical normalization
   source and derive broadcast `mu/std` from them
2. if they do not exist, fall back to legacy `mu/std`
3. if `normalization_stats.json` exists, merge its metadata into the resolved
   normalization info

This means older runs continue to work unchanged, while newer v2.3 runs expose
their normalization mode explicitly.

## Change 9 — Normalization Metadata Surfaced Through the Web Service

`MelCNN-MGR/inference_web_service/app.py` now includes resolved normalization
metadata in:

1. startup logs
2. `GET /health`
3. `GET /model`

That makes it easier to confirm from outside the service whether a run is using:

1. legacy `mu/std`
2. explicit per-mel-bin standardization metadata

without opening model artifacts manually.

## Validation Status

Validated in this session:

1. the inference service successfully loaded the existing demo v2.2 model after
   the Keras-compatibility changes and answered `GET /health`
2. `MelCNN-MGR/demo-app/web_audio_capture_v1.py` passed static error checks
   after the frontend rendering fix
3. `MelCNN-MGR/model_training/logmel_cnn_v2_3_cuda.py` passed static error checks
4. normalization loading in `MelCNN-MGR/model_inference/inference_logmel_cnn_v2_x.py`
   was validated for both:
   - legacy `mu/std` artifacts
   - synthetic v2.3-style `mean_per_bin/std_per_bin` artifacts with metadata

Not fully validated end-to-end yet:

1. a real training run of `logmel_cnn_v2_3_cuda.py`
2. a live inference-service launch pointed at an actual v2.3-trained model
3. browser-side confirmation that the demo app now renders rolling partial
   predictions continuously during capture
4. investigation of a separate runtime/device issue seen in one direct loader
   probe: `visible_device_list` contained an invalid device id during TensorFlow
   model rebuild in the current `.venv311` environment