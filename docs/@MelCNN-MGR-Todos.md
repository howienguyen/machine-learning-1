# TODO list for MelCNN MGR project

Mon Mar  2 14:00:00 UTC 2026

Main reference: docs\Final-Project-Proposal.md

1. Understanding the dataset fma_medium & Preparing data: train, validation, & test datasets --> DONE 

(D:\mse\nguyen_sy_hung_codebases\machine-learning-1\MelCNN-MGR\notebooks\Data-Understanding-Train-Val-Test-Genre-Distribution.ipynb)

The idea is that train dataset & test dataset should have the "same distribution shape" (i.e similar genre ratios across train/validation/test) - would it better in this way in the scope/context of the project?

In MelCNN MGR, it’s preferable (though not strictly required) that train/validation/test have similar genre proportions, because it reduces the chance that performance differences are driven by a different class mix rather than MFCC vs log-mel (Goal 1) or optimization choices (Goal 2). Since FMA provides an official split in fma_metadata/tracks.csv, the best practice is to use that split for comparability and to avoid leakage traps, then measure and report genre proportions as a “distribution-shape” sanity check. If proportions differ noticeably, the evaluation remains valid, but results should emphasize Macro-F1 and per-genre F1, and may optionally use class-weighted loss.

2. Reproducing the baseline by developing and using our code but based on the same method (MFCC → 2D CNN) in the original FMA repo (FMA\fma-repo).

Script: MelCNN-MGR\baseline_mfcc_cnn.py
  - Faithfully reproduces FMA baselines.ipynb §3.1 "ConvNet on MFCC"
  - Architecture: (13, 2582, 1) → Conv2D(3) → Conv2D(15) → Conv2D(65) → Dense(16, softmax)
  - Hyperparams: SGD lr=1e-3, batch_size=16, epochs=20 (same as original)
  - Modern TF2/Keras API (replaces deprecated Keras 1.x fit_generator / nb_epoch / etc.)
  - MFCC caching: MelCNN-MGR/cache/mfcc_{split}.npy (skip re-extraction on re-runs)
  - Outputs: accuracy + Macro-F1 + per-genre F1, model → MelCNN-MGR/models/, results → MelCNN-MGR/results/

Run:
  python MelCNN-MGR\baseline_mfcc_cnn.py
  python MelCNN-MGR\baseline_mfcc_cnn.py --epochs 20 --batch-size 16
  python MelCNN-MGR\baseline_mfcc_cnn.py --clear-cache   # force MFCC re-extraction