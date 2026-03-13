# Raw input materials for the project report 

### Pre-processing

Mon Mar  9 10:14:44 UTC 2026

```bash
(.venv311) hsnguyen@hsnwsl:/mnt/d/mse/nguyen_sy_hung_codebases/machine-learning-1$ python MelCNN-MGR/Lab/build_manifest.py \
  --subset medium \
  --audio-root FMA/fma_medium \
  --metadata-root FMA/fma_metadata \
  --out-dir MelCNN-MGR/data/processed
10:19:50 INFO     Loading FMA/fma_metadata/tracks.csv
10:19:56 INFO       Loaded 106574 total tracks
10:19:56 INFO     Phase A: collecting candidates for subset='medium'
10:19:56 INFO       Total rows: 106574 | In subset 'medium': 17000
10:19:56 INFO     Phase B: resolving filepaths under FMA/fma_medium
10:21:50 INFO       Audio files found: 25000 / 106574
10:21:50 INFO     Phase C: assigning reason codes (min_duration=30s)
10:21:50 INFO       NOT_IN_SUBSET          89574
10:21:50 INFO       OK                     15749
10:21:50 INFO       EXCLUDED_LABEL         1251
10:21:50 INFO     Phase D: writing outputs to MelCNN-MGR/data/processed
10:21:50 INFO       MelCNN-MGR/data/processed/metadata_manifest_medium.parquet (106574 rows)
10:21:51 INFO       MelCNN-MGR/data/processed/train_medium.parquet (12521 rows)
10:21:51 INFO       MelCNN-MGR/data/processed/val_medium.parquet (1580 rows)
10:21:51 INFO       MelCNN-MGR/data/processed/test_medium.parquet (1648 rows)
10:21:51 INFO       MelCNN-MGR/data/processed/metadata_manifest_config_medium.json
10:21:51 INFO       MelCNN-MGR/data/processed/metadata_manifest_report_medium.txt
10:21:51 INFO     Done in 120.3s — 15749 usable samples out of 106574 candidates.
(.venv311) hsnguyen@hsnwsl:/mnt/d/mse/nguyen_sy_hung_codebases/machine-learning-1$ python MelCNN-MGR/Lab/build_tiny_dataset.py \
  --processed-dir MelCNN-MGR/data/processed \
  --source-subset medium \
  --output-subset tiny
10:22:10 INFO     Source train: MelCNN-MGR/data/processed/train_medium.parquet
10:22:10 INFO     Source val  : MelCNN-MGR/data/processed/val_medium.parquet
10:22:10 INFO     Source test : MelCNN-MGR/data/processed/test_medium.parquet
10:22:10 INFO     ─── Loading source splits ───────────────────────────────────
10:22:10 INFO     Loaded train |  12521 rows × 10 cols
10:22:10 INFO                  | 15 genres  top-3: Rock(4881), Electronic(4250), Hip-Hop(961) …
10:22:10 INFO     Loaded val   |   1580 rows × 10 cols
10:22:10 INFO                  | 15 genres  top-3: Rock(611), Electronic(532), Hip-Hop(120) …
10:22:10 INFO     Loaded test  |   1648 rows × 10 cols
10:22:10 INFO                  | 15 genres  top-3: Rock(611), Electronic(532), Hip-Hop(120) …
10:22:10 INFO     ─── Sampling train split ───────────────────────────────────
10:22:10 INFO       15 genres found: ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
10:22:10 INFO       Sampling up to 70 rows/genre …
10:22:10 WARNING    Short genres (available/requested): Blues(58/70), Easy Listening(13/70), International(14/70)
10:22:10 INFO       → 925 total rows sampled (15 genres, avg 61.7/genre)
10:22:10 INFO     ─── Sampling val split ─────────────────────────────────────
10:22:10 INFO       15 genres found: ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
10:22:10 INFO       Sampling up to 15 rows/genre …
10:22:10 WARNING    Short genres (available/requested): Blues(8/15), Easy Listening(2/15), International(2/15), Spoken(12/15)
10:22:10 INFO       → 189 total rows sampled (15 genres, avg 12.6/genre)
10:22:10 INFO     ─── Sampling test split ────────────────────────────────────
10:22:10 INFO       15 genres found: ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 'Folk', 'Hip-Hop', 'Instrumental', 'International', 'Jazz', 'Old-Time / Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']
10:22:10 INFO       Sampling up to 15 rows/genre …
10:22:10 WARNING    Short genres (available/requested): Blues(8/15), Easy Listening(6/15), International(2/15), Spoken(12/15)
10:22:10 INFO       → 193 total rows sampled (15 genres, avg 12.9/genre)
10:22:10 INFO     ─── Writing outputs ────────────────────────────────────────
10:22:10 INFO     Wrote train → train_tiny.parquet  (925 rows)
10:22:10 INFO     Wrote val   → val_tiny.parquet  (189 rows)
10:22:10 INFO     Wrote test  → test_tiny.parquet  (193 rows)
10:22:10 INFO     Wrote config → tiny_dataset_config_tiny.json
10:22:10 INFO     Wrote report → tiny_dataset_report_tiny.txt
10:22:10 INFO     ─── Summary ────────────────────────────────────────────────
10:22:10 INFO       Source subset   : subset:medium
10:22:10 INFO       Output subset   : tiny
10:22:10 INFO       Genres found : 15
10:22:10 INFO       Rows out     : train=925  val=189  test=193  (total=1307)
10:22:10 INFO       Budget/genre : train=70  val=15  test=15
10:22:10 INFO       Output dir   : MelCNN-MGR/data/processed
10:22:10 INFO       Elapsed      : 0.17s
(.venv311) hsnguyen@hsnwsl:/mnt/d/mse/nguyen_sy_hung_codebases/machine-learning-1$
```



### Baselines

MelCNN-MGR/model_training/baseline_logmel_cnn_v1.ipynb
MelCNN-MGR/model_training/baseline_mfcc_cnn_v5.ipynb

Note: Using either "tiny" or "small" dataset is good enough in this case
