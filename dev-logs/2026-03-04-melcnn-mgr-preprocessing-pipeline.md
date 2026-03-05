# Dev Log — 2026-03-04 — MelCNN-MGR Preprocessing Pipeline

## Scope
This log covers the full build-out of the FMA preprocessing pipeline for the MelCNN-MGR music genre
recognition project. It spans planning the metadata manifest system, implementing `build_manifest.py`,
migrating the training script to consume the manifest, hardening artifact naming with subset suffixes,
and keeping all documents in sync.

## High-Level Timeline
1. Define the `metadata_manifest.parquet` design and document the philosophy in `MelCNN-MGR-Preprocessing.md`.
2. Implement `MelCNN-MGR/preprocessing/build_manifest.py` — four-phase pipeline (Collect → Resolve → Reason → Freeze).
3. Verify ground truth: run the builder against `fma_medium`, confirm 17 000 OK rows and zero artist leakage.
4. Migrate `baseline_mfcc_cnn_v2.py` away from raw `tracks.csv` to the manifest parquets.
5. Harden artifact filenames — include `{subset}` tag in every output so multiple subsets can coexist.
6. Update all docs and fix `DEFAULT_SUBSET` constant to match the project's actual target subset.

---

## Phase 1 — Metadata Manifest Design

### Motivation
FMA ships metadata (`fma_metadata/`) and audio (`fma_medium/`, etc.) as separate trees. Problems arise
in real usage:
- A `track_id` exists in `tracks.csv` but the MP3 is missing (partial download).
- A genre label is null for tracks outside the target subset.
- A corrupt file crashes a training epoch.
- Naïve random splits leak artist identity across train/test.

Solution: generate a single `metadata_manifest_{subset}.parquet` that captures — for every row — where
the file is, what its label and split are, **and why it was excluded if it was**. Training then reads
only `reason_code == "OK"` rows from pre-split files.

### Core design decisions
- `track_id` is the primary key (index in `tracks.csv`, directory component in the audio path).
- The manifest is **deterministic**: same inputs + same rules → same output, forever.
- Every excluded row carries a `reason_code` (not silently dropped) so problems are visible at a glance.
- Official FMA split (`('set', 'split')`) is used as-is — do not rebuild from scratch.

### Schema settled on

**Minimum columns:**

| Column        | Source in tracks.csv         | Notes                              |
|---------------|------------------------------|------------------------------------|
| `track_id`    | index                        | primary key                        |
| `filepath`    | derived                      | absolute path to MP3               |
| `audio_exists`| filesystem stat              |                                    |
| `split`       | `('set', 'split')`           | training / validation / test       |
| `genre_top`   | `('track', 'genre_top')`     | single-label target; 16 genres     |
| `reason_code` | assigned by pipeline         |                                    |

**Extras (always included):**

| Column           | Source                       | Notes                              |
|------------------|------------------------------|------------------------------------|
| `subset`         | `('set', 'subset')`          | small / medium / large             |
| `artist_id`      | `('artist', 'id')`           | leakage check                      |
| `duration_s`     | `('track', 'duration')`      | integer seconds from metadata      |
| `bit_rate`       | `('track', 'bit_rate')`      | many VBR non-standard values       |
| `filesize_bytes` | stat                         | integrity check                    |

---

## Phase 2 — `build_manifest.py` Implementation

Script: `MelCNN-MGR/preprocessing/build_manifest.py`

### Phase A — Collect candidates
- Reads `tracks.csv` with `pd.read_csv(..., index_col=0, header=[0,1])`.
- Parses list-valued columns (`track.genres`, `*.tags`) via `ast.literal_eval`.
- Extracts the 6 needed multi-index columns and renames them to flat names.
- Flags subset membership into a helper column `_in_target_subset`.

### Phase B — Resolve filepaths
- FMA path convention: `{audio_root}/{tid[:3]}/{tid}.mp3` (zero-padded to 6 digits).
- Stats each expected path; records `audio_exists` and `filesize_bytes`.
- Implemented as a pure function `get_audio_path(audio_dir, track_id)` matching `fma-repo/utils.py`.

### Phase C — Reason codes
Priority order (first rule that fires wins):

```
NOT_IN_SUBSET > NO_AUDIO_FILE > NO_LABEL > NO_SPLIT > TOO_SHORT > OK
```

| Code               | Meaning                                        |
|--------------------|------------------------------------------------|
| `OK`               | usable sample                                  |
| `NOT_IN_SUBSET`    | track not in target subset                     |
| `NO_AUDIO_FILE`    | metadata row exists but MP3 missing            |
| `NO_LABEL`         | `genre_top` null or empty                      |
| `NO_SPLIT`         | split value missing or unrecognised            |
| `TOO_SHORT`        | duration < `--min-duration` (default 30 s)     |
| `DECODE_FAIL`      | file exists but fails `soundfile.info()` probe |
| `DUPLICATE_TRACK_ID` | duplicate index in metadata               |

Optional `--decode-probe` flag adds a `soundfile.info()` pass over every OK file and fills
`sample_rate` / `channels` columns.

### Phase D — Freeze outputs
All artifact filenames include the subset tag (e.g. `_medium`) so different subsets coexist safely:

```
MelCNN-MGR/data/processed/
  metadata_manifest_medium.parquet        # all 106 574 rows
  train_medium.parquet                    # 13 522 OK rows, split=training
  val_medium.parquet                      #  1 705 OK rows, split=validation
  test_medium.parquet                     #  1 773 OK rows, split=test
  metadata_manifest_config_medium.json    # CLI args + timestamp
  metadata_manifest_report_medium.txt     # quality-gate summary
```

The report includes: reason-code counts, split distribution, genre distribution, and an
artist-leakage intersection check for all three split pairs.

### CLI
```
python MelCNN-MGR/preprocessing/build_manifest.py            # defaults: subset=medium
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium --decode-probe
```

### Performance note
Phase B (filesystem scan of 106 574 stat() calls) takes ~90 s on WSL2/NTFS. This is a one-time
cost — downstream loaders read from the cached parquets in milliseconds.

---

## Phase 3 — Ground Truth Verification (fma_medium)

Ran the builder against the actual data; confirmed:

| Fact                             | Value                     |
|----------------------------------|---------------------------|
| Total rows in `tracks.csv`       | 106 574                   |
| In medium subset                 | 17 000                    |
| Audio files present              | 17 000 (100 %)            |
| OK (reason_code)                 | 17 000                    |
| Genres                           | 16                        |
| split=training                   | 13 522                    |
| split=validation                 | 1 705                     |
| split=test                       | 1 773                     |
| Artist overlap train∩val         | 0                         |
| Artist overlap train∩test        | 0                         |
| Artist overlap val∩test          | 0                         |
| Duration range                   | 60 s – 600 s (mean ≈ 230 s) |
| `TOO_SHORT` count                | 0 (fma_medium min = 60 s, well above 30 s threshold) |

Conclusion: use the official FMA split unchanged. No artist leakage. No missing audio for medium.

---

## Phase 4 — `baseline_mfcc_cnn_v2.py` Migration

Replaced all direct `tracks.csv` loading with manifest parquet loading.

**Before (v1):** loaded `tracks.csv` with multi-index header on every run, computed paths from
`track_id`, had no audit trail for excluded samples.

**After (v2):** calls `load_manifest_splits(processed_dir, subset)` which reads:
```python
train_df = pd.read_parquet("MelCNN-MGR/data/processed/train_medium.parquet")
val_df   = pd.read_parquet("MelCNN-MGR/data/processed/val_medium.parquet")
test_df  = pd.read_parquet("MelCNN-MGR/data/processed/test_medium.parquet")
```
The `filepath` column is already resolved — no path construction needed in the training loop.

MFCC cache files also gained the subset suffix: `mfcc_{split}_{subset}.npy` → e.g.
`mfcc_training_medium.npy`.

Architecture and hyperparameters are unchanged from the FMA baseline:
- Input `(13, 2582, 1)` — 13 MFCC × 2582 time frames (30 s @ sr=22050, hop=256)
- 3× Conv2D → Flatten → Dense(16, softmax)
- SGD lr=1e-3, batch_size=16, epochs=20

---

## Phase 5 — Subset-Suffixed Artifact Naming (Hardening)

### Problem
Earlier iterations wrote `metadata_manifest.parquet`, `train.parquet`, etc. Running the builder
for a different subset would silently overwrite the previous run's files.

### Fix
`phase_d_write` now derives `subset_suffix` from `config['subset']` and appends it to every
output filename. The legacy non-suffixed files in `MelCNN-MGR/data/processed/` were removed.

### Code pattern
```python
subset_suffix = f"_{subset_name}" if subset_name else ""
manifest_path = out_dir / f"metadata_manifest{subset_suffix}.parquet"
```

This applies to all six outputs: manifest parquet, three split parquets, config JSON, report TXT.

---

## Phase 6 — Document Updates

### `MelCNN-MGR/preprocessing/build_manifest.py`
- Module docstring Outputs section updated to show `{subset}`-suffixed filenames.
- `phase_d_write` docstring updated similarly.
- `DEFAULT_SUBSET` changed from `"small"` → `"medium"` (matches actual project target; also fixes
  the default `DEFAULT_AUDIO_ROOT` to point at `fma_medium`).

### `docs/MelCNN-MGR-Preprocessing.md`
- Phase D Outputs table rewritten to show all 6 suffixed artifacts.
- "Practical naming + storage conventions" section rewritten — replaced vague `run_001/` example
  with the concrete `_medium`-suffix convention and the actual output directory.

### `docs/@MelCNN-MGR-Todos.md`
- Item 2 expanded into **Step 2a** (build_manifest.py, manifest pipeline) and **Step 2b**
  (baseline_mfcc_cnn_v2.py, manifest-aware training script), with exact run commands for each.
- Old `baseline_mfcc_cnn.py` reference (v1, reads tracks.csv directly) clarified as
  "kept for reference only".

---

## Artifacts

| File                                              | Role                                 |
|---------------------------------------------------|--------------------------------------|
| `MelCNN-MGR/preprocessing/build_manifest.py`     | Manifest builder (Phase A–D)         |
| `MelCNN-MGR/baseline_mfcc_cnn_v2.py`             | Manifest-aware training script       |
| `MelCNN-MGR/baseline_mfcc_cnn_v1.py`             | Original standalone script (reference) |
| `MelCNN-MGR/data/processed/metadata_manifest_medium.parquet` | Full manifest, 106 574 rows |
| `MelCNN-MGR/data/processed/train_medium.parquet`  | 13 522 OK training rows              |
| `MelCNN-MGR/data/processed/val_medium.parquet`    | 1 705 OK validation rows             |
| `MelCNN-MGR/data/processed/test_medium.parquet`   | 1 773 OK test rows                   |
| `MelCNN-MGR/data/processed/metadata_manifest_config_medium.json` | Config snapshot |
| `MelCNN-MGR/data/processed/metadata_manifest_report_medium.txt` | Quality-gate report |
| `docs/MelCNN-MGR-Preprocessing.md`               | Pipeline design doc (updated)        |
| `docs/@MelCNN-MGR-Todos.md`                       | Project todos (updated)              |

---

## How to Run

### Step 1 — Build the manifest (one-time, ~90 s on WSL2/NTFS)
```bash
source .venv/bin/activate
python MelCNN-MGR/preprocessing/build_manifest.py --subset medium
# optionally add --decode-probe to verify every audio header
```

### Step 2 — Train the baseline
```bash
python MelCNN-MGR/baseline_mfcc_cnn_v2.py
# first run: MFCC extraction, ~several hours; subsequent runs load from cache
python MelCNN-MGR/baseline_mfcc_cnn_v2.py --clear-cache  # force re-extraction
```

---

## Next Items
- Run the full training and record baseline accuracy / Macro-F1 on the test set.
- Design and implement the log-mel + 2D CNN variant for Goal 1 comparison.
- Consider `--decode-probe` pass to fill `sample_rate` / `channels` in the manifest
  (useful for validating FMA audio is uniformly 44 100 Hz stereo before resampling).
