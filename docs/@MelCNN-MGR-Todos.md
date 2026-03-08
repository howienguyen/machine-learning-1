# TODO list for MelCNN MGR project

Mon Mar  2 14:00:00 UTC 2026

Main reference: docs\Final-Project-Proposal.md

1. Understanding the dataset fma_medium & Preparing data: train, validation, & test datasets --> DONE 

(D:\mse\nguyen_sy_hung_codebases\machine-learning-1\MelCNN-MGR\notebooks\Data-Understanding-Train-Val-Test-Genre-Distribution.ipynb)

The idea is that train dataset & test dataset should have the "same distribution shape" (i.e similar genre ratios across train/validation/test) - would it better in this way in the scope/context of the project?

In MelCNN MGR, it’s preferable (though not strictly required) that train/validation/test have similar genre proportions, because it reduces the chance that performance differences are driven by a different class mix rather than MFCC vs log-mel (Goal 1) or optimization choices (Goal 2). Since FMA provides an official split in fma_metadata/tracks.csv, the best practice is to use that split for comparability and to avoid leakage traps, then measure and report genre proportions as a “distribution-shape” sanity check. If proportions differ noticeably, the evaluation remains valid, but results should emphasize Macro-F1 and per-genre F1, and may optionally use class-weighted loss.

2. The MFCC CNN baseline solution: Reproducing the baseline by developing and using our code but based on the same method (MFCC → 2D CNN) in the original FMA repo (FMA\fma-repo). --> DONE (v2 with manifest pipeline below)

**Step 2a — Build the metadata manifest (one-time preprocessing step):**

Script: MelCNN-MGR\preprocessing\build_manifest.py
  - Implements the four-phase pipeline from docs/MelCNN-MGR-Preprocessing.md:
      Phase A: collect candidates from tracks.csv (metadata-first)
      Phase B: resolve filepaths + check filesystem existence
      Phase C: assign exactly one reason_code per row (OK / NOT_IN_SUBSET / NO_AUDIO_FILE / etc.)
      Phase D: write subset-suffixed artifacts to MelCNN-MGR/data/processed/
  - Outputs (all filenames include the subset tag, e.g. _medium):
      metadata_manifest_medium.parquet     — full table, all 106 574 rows
      train_medium.parquet                 — 13 522 OK rows, split=training
      val_medium.parquet                   — 1 705  OK rows, split=validation
      test_medium.parquet                  — 1 773  OK rows, split=test
      metadata_manifest_config_medium.json — config snapshot for reproducibility
      metadata_manifest_report_medium.txt  — quality-gate summary (reason codes, genre/split distribution, artist-leakage check)

Run (must be done before the training script):
  python MelCNN-MGR/preprocessing/build_manifest.py                      # defaults: subset=medium
  python MelCNN-MGR/preprocessing/build_manifest.py --subset medium
  python MelCNN-MGR/preprocessing/build_manifest.py --subset medium --decode-probe   # also probe audio headers

**Step 2b — Train the MFCC CNN baseline (reads from manifest, no raw metadata needed):**

Script: MelCNN-MGR\baseline_mfcc_cnn_v2.py  ← current version
  - Reads train/val/test splits from MelCNN-MGR/data/processed/{split}_medium.parquet
  - Faithfully reproduces FMA baselines.ipynb §3.1 "ConvNet on MFCC"
  - Architecture: (13, 2582, 1) → Conv2D(3) → Conv2D(15) → Conv2D(65) → Dense(16, softmax)
  - Hyperparams: SGD lr=1e-3, batch_size=16, epochs=20 (same as original)
  - Modern TF2/Keras API (replaces deprecated Keras 1.x fit_generator / nb_epoch / etc.)
  - MFCC caching: MelCNN-MGR/cache/mfcc_{split}_{subset}.npy (skip re-extraction on re-runs)
  - Outputs: accuracy + Macro-F1 + per-genre F1, model → MelCNN-MGR/models/, results → MelCNN-MGR/results/

Run:
  python MelCNN-MGR\baseline_mfcc_cnn_v2.py
  python MelCNN-MGR\baseline_mfcc_cnn_v2.py --epochs 20 --batch-size 16
  python MelCNN-MGR\baseline_mfcc_cnn_v2.py --clear-cache   # force MFCC re-extraction

Note: MelCNN-MGR\baseline_mfcc_cnn_v1.py is the original standalone version (reads tracks.csv directly, no manifest). Kept for reference only.

3. Log-mel + CNN: using the **same CNN architecture** and the **same training setup** (MelCNN-MGR/data/processed) as the MFCC CNN baseline solution (MelCNN-MGR/notebooks/baseline_mfcc_cnn_v5.ipynb) uses, changing only the input representation.
3.1. Developing the Log-mel + CNN solution (i.e the training notebook for Log-mel + CNN, just like we have baseline_mfcc_cnn_v5.ipynb) --> DONE

(D:\mse\nguyen_sy_hung_codebases\machine-learning-1\MelCNN-MGR\notebooks\baseline_logmel_cnn_v1.ipynb)

Key design choices (controlled comparison with baseline_mfcc_cnn_v5.ipynb):
  - Same STFT params: sr=22050, n_fft=512, hop_length=256 -> 2582 time frames
  - 128 log-mel bands (same mel filterbank used internally by MFCC; MFCC just applies DCT+truncation on top)
  - Log compression: log(1 + S) via np.log1p
  - Feature shape: (128, 2582) instead of (13, 2582)
  - CNN: Conv2D(3, (128,10)) instead of Conv2D(3, (13,10)); all other layers identical
  - Same training recipe: SGD lr=1e-3, batch_size=16, epochs=20, categorical_crossentropy
  - Same preprocessing: per-band train-only mean/std normalisation, tf.data streaming
  - Same caching: per-track .npy in cache/logmel/shared/, parquet indexes
  - Same run output: timestamped models/logmel-cnn-<ts>/ with .keras + run_report JSON

3.2. Comparing this Log-mel + CNN with the MFCC CNN baseline

3.3. Fix Conv1 kernel design issue — representation-fair architecture → DONE

(D:\mse\nguyen_sy_hung_codebases\machine-learning-1\MelCNN-MGR\notebooks\baseline_logmel_cnn_v10.ipynb)

Root cause: `baseline_logmel_cnn_v1.ipynb` used `Conv2D(3, (128,10))` in Conv1, collapsing
the entire 128-band frequency axis in the first layer. This prevented the CNN from learning
local spectro-temporal patterns and caused Log-Mel accuracy to fall below the MFCC baseline.
Full analysis: `docs/Conv1 Kernel Design Issue in Log-Mel Baseline.md`
Dev log: `dev-logs/2026-03-06-baseline-logmel-cnn-v10.md`

Key changes from v1 → v10:
  - Conv1 kernel: (128,10) global → (5,5) local  [~51× fewer params per filter]
  - Architecture: 3 stride-only conv layers → 4 Conv+BN+MaxPool blocks
  - Classifier: Flatten+Dense → GlobalAveragePooling2D+Dropout(0.3)+Dense
  - Optimizer: SGD(lr=1e-3) → Adam(lr=1e-3)
  - Callbacks: none → EarlyStopping(patience=5) + ReduceLROnPlateau(patience=3)
  - Epochs: 20 → 30 (max; EarlyStopping may stop earlier)
  - Run artifacts: logmel-cnn-v10-{ts}/ with baseline_logmel_cnn_v10.keras

4. Reusable inference modules — DONE (2026-03-08)
Dev log: `dev-logs/2026-03-08-inference-modules.md`

Both trained models now have matching standalone inference modules and CLI example scripts
that can be run independently of the training notebooks/scripts.

4.1. Log-Mel CNN v20a inference module → DONE
  Files:
    MelCNN-MGR/inference_logmel_v20a.py         ← MelCNNInference class
    MelCNN-MGR/examples/inference_logmel_v20a_example.py
  Loads: baseline_logmel_cnn_v20a.keras + norm_stats.npz from a logmel-cnn-v20a-* run dir
  Default mode: three_crop (3 × 10s clips, averaged)
  Run:
    python MelCNN-MGR/examples/inference_logmel_v20a_example.py \
        --run-dir MelCNN-MGR/models/logmel-cnn-v20a-<ts> --subset small --random 5

4.2. MFCC CNN v5 inference module → DONE
  Files:
    MelCNN-MGR/inference_mfcc_v5.py              ← MFCCCNNInference class
    MelCNN-MGR/examples/inference_mfcc_v5_example.py
  Loads: baseline_mfcc_cnn.keras + norm_stats.npz from a mfcc-cnn-* run dir
  Default mode: single_crop (full 30s clip)
  Prerequisite: norm_stats.npz is now saved by baseline_mfcc_cnn_v5.ipynb (Section 7)
  Run:
    python MelCNN-MGR/examples/inference_mfcc_v5_example.py \
        --run-dir MelCNN-MGR/models/mfcc-cnn-<ts> --subset small --random 5