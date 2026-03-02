# TODO list for MelCNN MGR project

Mon Mar  2 14:00:00 UTC 2026

Main reference: docs\Final-Project-Proposal.md

1. Understand the dataset fma_medium

2. Prepare data: train dataset & test dataset --> DONE (D:\mse\nguyen_sy_hung_codebases\machine-learning-1\MelCNN-MGR\notebooks\Data-Understanding-Train-Val-Test-Genre-Distribution.ipynb)

The idea is that train dataset & test dataset should have the "same distribution shape" (i.e similar genre ratios across train/validation/test) - would it better in this way in the scope/context of the project?

In MelCNN MGR, it’s preferable (though not strictly required) that train/validation/test have similar genre proportions, because it reduces the chance that performance differences are driven by a different class mix rather than MFCC vs log-mel (Goal 1) or optimization choices (Goal 2). Since FMA provides an official split in fma_metadata/tracks.csv, the best practice is to use that split for comparability and to avoid leakage traps, then measure and report genre proportions as a “distribution-shape” sanity check. If proportions differ noticeably, the evaluation remains valid, but results should emphasize Macro-F1 and per-genre F1, and may optionally use class-weighted loss.


2. Reproduce the basline