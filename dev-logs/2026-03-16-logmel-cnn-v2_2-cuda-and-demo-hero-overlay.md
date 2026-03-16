# Dev Log — 2026-03-16 — `logmel_cnn_v2_2_cuda.py` Runtime Hardening and Demo Hero Overlay

## Scope

This log captures three small but targeted updates made in this session:

1. runtime-stability fixes in `MelCNN-MGR/model_training/logmel_cnn_v2_2_cuda.py`
2. a new low-opacity background-image overlay for the hero card in `MelCNN-MGR/demo-app/web_audio_capture_v1.py`
3. explicit XPU backend selection work in `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`

Updated artifacts:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_2_cuda.py`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`
3. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`
4. `dev-logs/2026-03-16-logmel-cnn-v2_2-cuda-and-demo-hero-overlay.md`

## Summary

The CUDA training script was hardened against the TensorFlow/Keras environment currently used in `.venv` by disabling XLA auto-JIT before import, disabling oneDNN graph rewrites, disabling TensorFlow's layout optimizer, forcing float32 precision policy, keeping model compile with `jit_compile=False`, and making model loading more resilient when custom learning-rate schedules are present in saved `.keras` checkpoints.

The standard `logmel_cnn_v2_2.py` training script was also updated so backend selection can be forced to `xpu`, `cuda`, `cpu`, or `auto`, with early environment-based CUDA suppression when `xpu` or `cpu` is requested.

Separately, the demo app hero card now includes an extra background-image overlay layer based on `MelCNN-MGR/demo-app/images/hero_background.png`, positioned behind the card content for a softer blended visual treatment.

## Change 1 — Early TensorFlow Runtime Guardrails in `logmel_cnn_v2_2_cuda.py`

The training script now sets TensorFlow runtime flags before importing TensorFlow:

1. `TF_XLA_FLAGS=--tf_xla_auto_jit=-1`
2. `TF_ENABLE_ONEDNN_OPTS=0`
3. `ONEDNN_VERBOSE=none`
4. `DNNL_VERBOSE=0`

This matters because the XLA/JIT-related instability shows up only if the flag is present before TensorFlow initializes.

The script also now explicitly sets:

1. `keras.mixed_precision.set_global_policy("float32")`
2. `tf.config.optimizer.set_experimental_options({"layout_optimizer": False})`
3. startup prints for XLA auto-JIT, oneDNN state, layout-optimizer status, and active precision policy

That keeps training behavior explicit and avoids the CUDA-side Grappler layout rewrite that was failing around `SpatialDropout2D` at the start of training.

## Change 2 — Compile Path Explicitly Keeps JIT Disabled

The training constants now keep:

1. `BATCH_SIZE = 32`
2. `JIT_COMPILE = False`

The compile helper uses that flag directly during `model.compile(...)`.

This preserves the existing training configuration while making the no-JIT intent explicit in the shared optimizer / staged-unfreeze flow.

## Change 3 — Safer Keras Serialization for Warm Starts and Reloads

The cosine schedule in `logmel_cnn_v2_2_cuda.py` now uses:

1. `@tf.keras.utils.register_keras_serializable(package="MelCNN")`

The script also adds a `_load_keras_model(...)` helper that:

1. first tries normal `tf.keras.models.load_model(str(path), compile=...)`
2. retries with `custom_objects={"CosineAnnealingWithWarmup": CosineAnnealingWithWarmup}` when needed

This makes checkpoint loading and best-model evaluation more robust in environments where Keras deserialization is stricter or less forgiving with custom schedules.

## Change 4 — Final Classifier Output Forced to Float32

The classifier head now declares:

1. `layers.Dense(..., activation="softmax", dtype="float32", name="fc_out")`

That ensures the final output tensor remains float32 even if other runtime behaviors change later.

## Change 5 — Demo Hero Card Background Overlay

`MelCNN-MGR/demo-app/web_audio_capture_v1.py` now includes a dedicated hero-card overlay element:

1. a `.hero-card-overlay` absolute-positioned layer inside the target hero card
2. the overlay uses `hero_background.png` through the existing `hero_bg_url`
3. the layer is non-interactive and sits behind the card content with low opacity

This was added specifically for:

1. `<div class="card hero-card" style="margin-bottom: 16px; text-align: center;">`

The existing blurred card background remains in place, so the new layer acts as an additional visual texture instead of replacing the current hero styling.

## Change 6 — Explicit XPU Backend Selection in `logmel_cnn_v2_2.py`

`MelCNN-MGR/model_training/logmel_cnn_v2_2.py` previously used fixed GPU-first auto-detection.

It now supports an explicit backend override:

1. CLI: `--device-backend {auto,xpu,cuda,cpu}`
2. environment: `LOGMEL_CNN_DEVICE_BACKEND`

The device-selection logic was split into backend-specific helpers for:

1. CUDA probing
2. XPU probing through `intel_extension_for_tensorflow`
3. CPU fallback

## Change 7 — Early CUDA Suppression for Explicit XPU/CPU Runs

During testing, a later eager-context failure showed that changing visible devices after TensorFlow import was too late for reliable XPU forcing in this environment.

`logmel_cnn_v2_2.py` now resolves the requested backend before importing TensorFlow and, when `xpu` or `cpu` is explicitly requested, sets:

1. `CUDA_VISIBLE_DEVICES=-1`

before the TensorFlow import executes.

This prevents CUDA from being initialized first and avoids the invalid mixed-device state seen during XPU debugging.

The runtime-selection block still keeps explicit device-hiding helpers as a second line of defense after import, but the important change is that backend intent is now known early enough to affect TensorFlow initialization.

## Validation Status

Validated in this session:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_2_cuda.py` passed `python -m py_compile` in the project `.venv`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_2.py` passed `python -m py_compile` in the project `.venv311`
3. `MelCNN-MGR/demo-app/web_audio_capture_v1.py` passed `python -m py_compile` in the project `.venv`

Not yet done:

1. a live training smoke test for `logmel_cnn_v2_2_cuda.py` against the active TensorFlow runtime
2. a successful end-to-end XPU training run for `logmel_cnn_v2_2.py` in the current ITEX environment
3. browser-side visual validation of the updated hero-card overlay in the demo app
