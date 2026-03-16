# Dev Log — 2026-03-16 — Log-Mel Runtime Compatibility Fixes and Demo Hero Overlay Update

## Scope

This log covers two small but related maintenance updates completed on March 16, 2026:

1. TensorFlow/Keras runtime compatibility and stability fixes for the `logmel_cnn_v2_x` training scripts.
2. A UI refinement for the demo app hero card background in `web_audio_capture_v1.py`.

Updated artifacts:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_2_cuda.py`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`
3. `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`
4. `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py`
5. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`
6. `dev-logs/2026-03-16-logmel-runtime-and-demo-hero-overlay.md`

## Summary

The training-side work fixed a Keras API compatibility break in the current `.venv` environment and hardened the CUDA-oriented v2.2 entrypoint against XLA and precision-related instability.

The demo-app-side work added a second, very faint hero background image layer so the intro card can visually reuse the project artwork without reducing text readability.

## Change 1 — Keras Serializable Decorator Compatibility

### Problem

Several `logmel_cnn_v2_x` scripts still used:

```python
@tf.keras.saving.register_keras_serializable(package="MelCNN")
```

In the active TensorFlow/Keras environment, that namespace is unavailable and raises:

```python
AttributeError: module 'keras._tf_keras.keras' has no attribute 'saving'
```

This prevented the non-CUDA `logmel_cnn_v2_2.py` script from even starting model compilation/training.

### Fix

All affected scripts now use the compatible decorator path:

```python
@tf.keras.utils.register_keras_serializable(package="MelCNN")
```

Updated files:

1. `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`
2. `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`
3. `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py`

The CUDA variant already used the correct API and did not need that specific fix.

## Change 2 — `logmel_cnn_v2_2_cuda.py` Runtime Stabilization

### Problem

The CUDA-specific training entrypoint needed explicit protection against runtime instability in the current TensorFlow environment. In practice, the main issues were:

1. XLA auto-JIT being enabled implicitly too late or inconsistently
2. mixed-precision safety not being explicit
3. checkpoint/model reload robustness for custom learning-rate schedules
4. an aggressive default batch size for the current setup

### Fix

`MelCNN-MGR/model_training/logmel_cnn_v2_2_cuda.py` now:

1. sets `TF_XLA_FLAGS=--tf_xla_auto_jit=-1` before importing TensorFlow
2. forces global Keras mixed precision policy to `float32`
3. compiles the model with `jit_compile=False`
4. lowers the default `BATCH_SIZE` to `16`
5. keeps the classifier output explicitly in `float32`
6. uses a safer `load_model(...)` fallback with `custom_objects={"CosineAnnealingWithWarmup": ...}`

This keeps the v2.2 CUDA entrypoint aligned with the behavior that actually works in the project `.venv` environment.

## Change 3 — Demo App Hero Card Background Overlay

### Goal

The hero card in:

1. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`

needed an additional visual background layer using:

1. `MelCNN-MGR/demo-app/images/hero_background.png`

with the image kept highly subdued so the text remains readable.

### Fix

A dedicated overlay element was added inside the hero card and styled to:

1. cover the full hero-card bounds
2. use the existing `hero_background.png` asset via the template asset URL
3. render at `opacity: 0.10`
4. ignore pointer events and sit below the card content

This makes the added image approximately 90% transparent while preserving the existing hero-card gradient and glow layers.

## Validation Status

Validated in this session:

1. `py_compile` passed for `MelCNN-MGR/model_training/logmel_cnn_v2_2.py`
2. `py_compile` passed for `MelCNN-MGR/model_training/logmel_cnn_v2_1.py`
3. `py_compile` passed for `MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py`
4. `py_compile` passed for `MelCNN-MGR/model_training/logmel_cnn_v2_2_cuda.py`
5. `py_compile` passed for `MelCNN-MGR/demo-app/web_audio_capture_v1.py`

Not yet done:

1. a full end-to-end training run of `logmel_cnn_v2_2.py` after the decorator fix
2. a full end-to-end training run of `logmel_cnn_v2_2_cuda.py` under real GPU load
3. manual browser verification of the final hero-card visual blend in the running demo app