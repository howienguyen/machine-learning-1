# Dev Log — 2026-03-05 — MelCNN-MGR Training Pipeline Improvements

## Scope
This log covers four incremental improvements made to `baseline_mfcc_cnn_v5.ipynb` on March 5, 2026:
1. Fix a per-batch inference overhead in the evaluation loop.
2. Ensure training results survive notebook output clearing.
3. Isolate every training run in its own timestamped directory.
4. Consolidate all run artifacts into a single structured JSON report.

---

## Change 1 — Fix `model.predict` Per-Batch Overhead (Section 9)

### Problem
The evaluation helper `eval_dataset` iterated over a `tf.data.Dataset` manually,
calling `model.predict(xb, verbose=0)` once per batch:

```python
for xb, yb in ds:
    pred = model.predict(xb, verbose=0)   # ← wrong tool for the job
```

`model.predict()` is designed for **whole-dataset inference** — it wraps its input in
an internal `tf.data` pipeline, spins up progress callbacks, and handles batching
automatically.  Calling it once per batch inside a manual loop fires that scaffolding
overhead on every iteration.  For a 400-sample validation set in batches of 16
(25 iterations) the wasted dispatch adds up noticeably on XPU.

### Fix
Replace with the raw Keras forward-pass call:

```python
pred = model(xb, training=False).numpy()
```

`model(xb, training=False)` runs a single forward pass with no extra machinery and
returns an `EagerTensor`; `.numpy()` converts it immediately.  Results are numerically
identical.  The `training=False` flag is required so batch-norm and dropout layers
(if added later) behave correctly at inference time.

| Call style | When to use |
|---|---|
| `model.predict(ds)` | Whole-dataset inference in one call |
| `model(xb, training=False)` | Manual batch loop |

---

## Change 2 — Best-Epoch Print + Result Persistence (Section 7)

### Problem
`model.fit()` prints per-epoch metrics live, but those outputs disappear when the
notebook is saved with outputs cleared or when kernel state is lost.  There was no
durable record of training results.

### Fix — Part A: Best-epoch console summary
After `model.fit()`, identify and print the best epoch (minimum `val_loss`):

```python
best = min(range(len(history.history["val_loss"])),
           key=lambda i: history.history["val_loss"][i])
print(f"\nBest epoch : {best + 1} / {EPOCHS}")
for k in ["loss", "accuracy", "val_loss", "val_accuracy"]:
    print(f"  {k:<16}: {history.history[k][best]:.4f}")
```

This output appears as a cell output block that VS Code keeps even after re-running
subsequent cells — as long as Section 7 itself is not re-run, the summary stays visible.

### Fix — Part B: JSON file on disk
The full epoch-by-epoch metrics dict from `history.history` is serialised to
`train_history_{SUBSET}.json` in the run directory.  Even if all notebook outputs are
cleared and the kernel is restarted, the file remains.

---

## Change 3 — Timestamped Run Directory (Section 7)

### Problem
All runs wrote their artifacts (model weights, results) to the same flat path
`MelCNN-MGR/models/mfcc_cnn/`.  Each new run silently overwrote the previous one.
There was no way to compare two runs or recover a previous model.

### Design
A subdirectory is created at the start of every training run:
```
MelCNN-MGR/models/mfcc-cnn-<yyyymmdd-HHMMSS>/
```

The timestamp is generated inside the **training cell** (Section 7), not the config
cell, so it captures when training *actually ran*, not when the notebook was opened.

**NTFS constraint:** this workspace lives on `/mnt/d/` (Windows NTFS via WSL).
Colons (`:`) are illegal in NTFS file and directory names, so the format is
`%Y%m%d-%H%M%S` (e.g. `20260305-143022`) rather than the ISO 8601 `T14:30:22`.

The global `RUN_DIR` variable (set in Section 7, declared as `None` in the config
cell) is used by all subsequent sections so they all write into the same run directory.

**Artifacts per run:**
```
mfcc-cnn-20260305-143022/
    baseline_mfcc_cnn.keras
    run_report_small.json
```

---

## Change 4 — Single Comprehensive JSON Run Report

### Problem (v1: separate files)
After changes 2 and 3, two separate files were saved per run:
- `train_history_{SUBSET}.json` — raw float arrays, no context
- `eval_results_{SUBSET}.txt` — formatted text, no structure

A `MelCNN-MGR/results/` directory also existed for older text-format results files,
creating a second place for outputs.

### Solution: one JSON report per run

All per-run information is consolidated into a single `run_report_{SUBSET}.json`.

**Section 7 (training)** writes the file with:
- Run metadata: `run_id`, `subset`, `generated_at`, `model_file`
- Full config: device, audio backend, all MFCC extraction params, all training hyperparams
- Dataset summary: `n_classes`, `genres` list, and usable sample counts for each split
- Model architecture: `model.name`, `total_params`, and `model.summary()` captured as a string
- Training history: array of `{epoch, loss, accuracy, val_loss, val_accuracy}` records, plus a
  `best_epoch` object and timing stats
- `"evaluation": null` — placeholder filled by Section 9

**Section 9 (evaluation)** loads the file and fills in `"evaluation"`:
- Top-level timing
- Per-split (`train` / `validation` / `test`) accuracy and macro-F1
- Per-genre `precision`, `recall`, `f1-score`, `support` for every split
  (from sklearn's `classification_report(output_dict=True)`)

### Result layout

```json
{
  "run_id": "mfcc-cnn-20260305-071607",
  "subset": "small",
  "generated_at": "2026-03-05 07:20:53",
  "model_file": "baseline_mfcc_cnn.keras",
  "config": {
    "device": "/XPU:0",
    "audio_backend": "ffmpeg",
    "num_workers": 6,
    "mfcc_extraction": { "sample_rate": 22050, "n_mfcc": 13, "n_fft": 512, "hop_length": 256, "n_frames": 2582, "mfcc_shape": [13, 2582], "cache_shared": true },
    "training": { "epochs": 20, "batch_size": 16, "optimizer": "SGD", "lr": 0.001, "loss": "categorical_crossentropy" }
  },
  "dataset": { "n_classes": 8, "genres": ["Electronic", "..."], "train_samples": 6396, "val_samples": 800, "test_samples": 800 },
  "model_architecture": { "name": "mfcc_2dcnn_baseline", "total_params": 30441, "summary": "..." },
  "training_history": {
    "epochs": [ { "epoch": 1, "loss": 2.08, "accuracy": 0.13, "val_loss": 2.06, "val_accuracy": 0.14 }, "..." ],
    "best_epoch": { "epoch": 17, "loss": 1.81, "accuracy": 0.36, "val_loss": 1.89, "val_accuracy": 0.34 },
    "timing_seconds": 1320.4,
    "seconds_per_epoch": 66.0
  },
  "evaluation": {
    "timing_seconds": 14.7,
    "splits": {
      "train":      { "accuracy": 0.36, "macro_f1": 0.35, "per_genre": { "Rock": { "precision": 0.41, "recall": 0.38, "f1-score": 0.39, "support": 900 }, "..." } },
      "validation": { "..." },
      "test":       { "..." }
    }
  }
}
```

### Files removed / no longer created
- `MelCNN-MGR/results/mfcc_cnn_results_{SUBSET}.txt` — superseded by `evaluation` key in JSON
- `train_history_{SUBSET}.json` — superseded by `training_history` key in JSON
- `RESULTS_DIR` constant removed from config cell

---

## Files Changed

| File | Change |
|---|---|
| `MelCNN-MGR/notebooks/baseline_mfcc_cnn_v5.ipynb` | Config cell, Section 7 training cell, Section 9 eval cell, Section 9 markdown |

## Run Directory Layout (post-changes)

```
MelCNN-MGR/models/
  mfcc-cnn-20260305-071607/
    baseline_mfcc_cnn.keras
    run_report_small.json
  mfcc-cnn-20260305-161500/
    baseline_mfcc_cnn.keras
    run_report_small.json
```

Each run directory is fully self-contained and never overwritten by subsequent runs.
