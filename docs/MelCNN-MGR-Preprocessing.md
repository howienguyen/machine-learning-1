## metadata_manifest.parquet for FMA — Plan & Guidelines

Wed Mar  4 03:31:06 UTC 2026

### Why a metadata manifest exists (motivation)

FMA looks tidy on paper: metadata in `fma_metadata/` and audio in `fma_small/`, `fma_medium/`, etc. In real usage, you often end up with partial downloads, moved folders, subset-only audio, or filters applied to metadata. That’s when training breaks in annoying ways: a `track_id` exists in `tracks.csv` but the MP3 is missing; a file exists but has no usable label; one corrupt file crashes an epoch; or your split logic accidentally leaks artist identity across train/test.

A `metadata_manifest.parquet` solves this by becoming the **single source of truth**: a compact table that explicitly lists (1) what you intend to use, (2) where it is on disk, (3) what the label and split are, and (4) why anything was excluded. The program behind it is simple: **metadata + filesystem scan + rules ⇒ deterministic dataset**.

---

### Core design principle

Treat `track_id` as the primary key. Everything else is derived and validated against it. The metadata manifest should be *deterministic* (same inputs + same rules ⇒ same output) and *auditable* (you can explain every drop).

---

## High-level workflow

You’ll build the metadata manifest in three phases: **collect → validate → freeze**.

### Phase A — Collect candidates (metadata-first)

Start from metadata, because it defines labels and (usually) official splits.

**Inputs (typical):**

| Input                                   | Purpose                                                            | Notes                                               |
| --------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------- |
| `fma_metadata/tracks.csv`               | track rows, `genre_top`, artist info, and various track properties | Multi-index columns; load with `pd.read_csv(..., index_col=0, header=[0,1])`. Columns accessed as `('group', 'field')`, e.g. `('track', 'genre_top')`, `('set', 'split')`. List-valued columns (`track.genres`, `track.genres_all`, `*.tags`) need `ast.literal_eval`. |
| `fma_metadata/genres.csv`               | genre hierarchy and mapping                                        | Flat CSV; `genre_id` as index. `top_level` column gives root genre ID. Mostly for reference; `('track', 'genre_top')` is usually enough for single-label classification. |
| `fma_metadata/features.csv`             | 518 precomputed hand-crafted features (chroma, MFCC statistics, spectral) | Three-level multi-index header; load with `header=[0,1,2]`. Covers all 106 574 tracks. Useful as a baseline or for feature-augmented models, not required for raw-audio pipelines. |
| Audio root folder (e.g., `fma_medium/`) | actual MP3 files                                                   | The real “what exists”                              |

**Output of Phase A:** a dataframe keyed by `track_id`, containing the metadata fields you want (at minimum `genre_top`, plus split info if available).

---

### Phase B — Resolve filepaths (filesystem alignment)

Next, derive the expected filepath for each `track_id` and confirm whether the file exists.

FMA's path convention (from `fma-repo/utils.py`):

* folder = first 3 digits of the 6-digit zero-padded track id
* filename = full 6-digit zero-padded track id + `.mp3`

```python
def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')
```

Example: `track_id = 2` → `000/000002.mp3`, `track_id = 1000` → `001/001000.mp3`

**Output of Phase B:** `filepath`, `audio_exists`, and optionally `filesize_bytes`.

---

### Phase C — Apply rules (filtering + reason codes)

This phase turns “candidates” into “usable samples”, while keeping an audit trail.

You should not simply drop bad rows silently. Instead, record a `reason_code` per row and keep the row in `metadata_manifest.parquet` for debugging. Training then filters to `reason_code == "OK"`.

**Typical filtering rules:**

| Rule                                    | Why it exists                           | Typical reason_code | fma_medium reality |
| --------------------------------------- | --------------------------------------- | ------------------- | ------------------ |
| audio file missing                      | prevents runtime crashes / missing data | `NO_AUDIO_FILE`     | 0 missing (all 17 000 present; still check defensively) |
| missing label (`genre_top` is null)     | no supervised target                    | `NO_LABEL`          | 0 nulls in medium subset |
| not in desired subset                   | avoid training on unexpected data       | `NOT_IN_SUBSET`     | 89 574 tracks outside medium in full tracks.csv |
| split missing/unknown                   | prevents leakage or undefined eval      | `NO_SPLIT`          | all medium rows have a split |
| decode fails (optional but recommended) | corrupt file can kill an epoch          | `DECODE_FAIL`       | rare; still worth probing |
| too short for your clip length          | avoid padding-dominated samples         | `TOO_SHORT`         | **not applicable for fma_medium** — minimum duration is 60 s, well above the standard 30 s clip window |

---

### Phase D — Freeze outputs (reproducible artifacts)

A good metadata manifest build produces two kinds of outputs:

1. the full `metadata_manifest.parquet` with all rows and reasons (auditable)
2. split-specific “ready sets” derived from it (convenience, optional)

**Outputs:**

All artifact filenames include the target subset tag (e.g. `_medium`) so outputs for different subsets can coexist in the same directory.

| Artifact                             | Contents                                                    | Usage                       |
| ------------------------------------ | ----------------------------------------------------------- | --------------------------- |
| `metadata_manifest_{subset}.parquet` | all candidates, including dropped rows + reasons            | debugging + reproducibility |
| `train_{subset}.parquet`             | `metadata_manifest` filtered to split=training and `OK`     | dataloader input            |
| `val_{subset}.parquet`               | filtered to split=validation and `OK`                       | dataloader input            |
| `test_{subset}.parquet`              | filtered to split=test and `OK`                             | final evaluation            |
| `metadata_manifest_config_{subset}.json` | config snapshot used to generate the manifest           | reproducibility             |
| `metadata_manifest_report_{subset}.txt`  | quality-gate summary (reason codes, split/genre counts)  | auditing                    |

---

## Recommended schema for `metadata_manifest.parquet`

Keep it compact but complete. Below is a practical baseline.

### Minimum columns (strongly recommended)

| Column         | Type            | Meaning                                                                      |
| -------------- | --------------- | ---------------------------------------------------------------------------- |
| `track_id`     | int             | primary key; index in `tracks.csv`                                           |
| `filepath`     | string          | absolute or root-relative path to MP3                                        |
| `audio_exists` | bool            | file exists on disk                                                          |
| `split`        | category/string | `training` / `validation` / `test` (source: `('set', 'split')` in tracks.csv) |
| `genre_top`    | category/string | single-label target (source: `('track', 'genre_top')` in tracks.csv); 16 top-level genres in fma_medium, **zero nulls** in the medium subset |
| `reason_code`  | string          | `OK` or why excluded                                                         |

### Useful extras (quality + leakage checks)

| Column           | Type            | Meaning                                                                         |
| ---------------- | --------------- | ------------------------------------------------------------------------------- |
| `subset`         | category/string | `small` / `medium` / `large` (ordered; source: `('set', 'subset')` in tracks.csv) |
| `artist_id`      | int             | helps verify artist leakage across splits (source: `('artist', 'id')` in tracks.csv) |
| `duration_s`     | int             | track duration in **whole seconds** (source: `('track', 'duration')` in tracks.csv — no decode needed); fma_medium range: 60–600 s, mean ≈ 230 s |
| `bit_rate`       | int             | encoding bit rate in bps (source: `('track', 'bit_rate')`); note VBR files produce non-standard values; -1 indicates missing |
| `sample_rate`    | int             | from decode probe; FMA audio is standardized at **44 100 Hz** — only needed if validating; standard clip = 1 321 967 samples ≈ 29.98 s |
| `channels`       | int             | from decode probe                                                               |
| `filesize_bytes` | int             | quick integrity check                                                           |
| `md5`            | string          | heavy but useful for integrity auditing                                         |
| `feat_path`      | string          | if you precompute log-mel/MFCC tensors                                          |

The philosophy is: **store what you need for training, plus just enough to diagnose weirdness**.

---

## Reason codes: make debugging boring (the good kind of boring)

A consistent reason code taxonomy turns chaos into counts.

**Suggested canonical set:**

| reason_code          | Meaning                           | Typical action                |
| -------------------- | --------------------------------- | ----------------------------- |
| `OK`                 | usable sample                     | include                       |
| `NO_AUDIO_FILE`      | metadata row but file missing     | fix dataset / re-download     |
| `NO_LABEL`           | label missing                     | drop or define label policy   |
| `NO_SPLIT`           | no split assignment               | use official split or rebuild |
| `DECODE_FAIL`        | cannot decode audio               | drop; optionally re-encode    |
| `TOO_SHORT`          | shorter than required clip length | drop or pad with caution      |
| `NOT_IN_SUBSET`      | outside requested subset          | drop                          |
| `DUPLICATE_TRACK_ID` | duplicate key due to merge error  | fix pipeline                  |

Keep `reason_code` as a single value per row. If you need multi-reason, add `reason_details` as a string, but don’t overcomplicate early.

---

## Audio decode quality: the `dequantization failed` warning

### What it is

When `librosa.load()` is called on an FMA MP3, libmpg123 (the C-level MP3 decoder) may print:

```
[src/libmpg123/layer3.c:INT123_do_layer3():1844] error: dequantization failed!
```

This message comes from **C-level stderr (OS fd 2)** — it bypasses Python's `sys.stderr` entirely, so `contextlib.redirect_stderr()` cannot catch it and it does not raise an exception.

### Root cause

Layer III (Layer 3 = MP3) encoding uses Huffman coding for the spectral data in each frame.
When the Huffman packet for a frame is corrupt or malformed, the decoder cannot reconstruct the PCM samples for that ~26 ms window.
libmpg123 recovers by **zeroing those samples** and continuing.
The audio array returned by `librosa.load` has the correct length and the MFCC matrix has the correct shape, but values for the affected time windows are replaced with silence.

### Impact in FMA

| Concern | Reality |
|---|---|
| Does it crash extraction? | No — `librosa.load` returns normally, no exception |
| Is the MFCC shape correct? | Yes — always `(13, 2582)` |
| Are MFCC values wrong? | **Yes** — affected ~26ms frames are zeroed |
| Is it counted in `skipped`? | **No** (without the fix) — it passes through silently |
| How many tracks are affected? | ~1–5 % of FMA; one corrupt track can print the message dozens of times |
| Does it affect reproducibility? | No — same corrupt file → same zeroed frames every run |

### The fix (in `baseline_mfcc_cnn_v2.py` / `.ipynb`)

The only reliable way to detect the C-level message in-process on Linux is to temporarily replace OS file descriptor 2 with a pipe during `librosa.load()`, then scan the pipe contents for libmpg123 error strings after decoding completes. fd 2 is always restored in a `finally` block.

`load_mfcc()` returns a `(mfcc, is_clean)` tuple.
`extract_split()` accepts a `skip_degraded: bool` parameter (controlled by `SKIP_DEGRADED` in the config):

| `SKIP_DEGRADED` | Behaviour |
|---|---|
| `False` (default) | Degraded tracks are **included** in the cache but counted separately. Use for the faithful FMA baseline reproduction. |
| `True` | Degraded tracks are **excluded** (counted as `skipped`). Use for cleaner input in your own model variants. Clearing the cache (`CLEAR_CACHE = True`) is required for this change to take effect on an existing cache. |

---

## Split policy guidance (leakage motivation)

If you have an official FMA split available, use it. If you don’t, do not randomly split by track: music is “style-correlated,” and the biggest leakage culprit is **artist**.

**Leakage-safe rule:** ensure `artist_id` does not appear in both train and test.
That’s why `artist_id` in `metadata_manifest.parquet` is useful: you can run a simple intersection check and catch mistakes before training.
**Verified for fma_medium:** the official FMA split (`('set','split')` in tracks.csv) is already artist-leakage free. Intersection of artist IDs across all three split pairs is zero (confirmed by direct check on the actual metadata). **Use the official split without modification** — do not rebuild it from scratch unless you have a specific reason.
---

## Program outline (what the builder should do)

The metadata manifest builder is best treated as a deterministic “build step” in your pipeline.

**Inputs:**

* `audio_root` (e.g., `/data/fma_medium`)
* `metadata_root` (e.g., `/data/fma_metadata`)
* config: subset name, label policy, clip length threshold, decode_probe on/off

**Steps:**

| Step                      | Description                                  | Determinism note               |
| ------------------------- | -------------------------------------------- | ------------------------------ |
| Load metadata             | read `tracks.csv` and extract needed columns | pin the exact metadata version |
| Compute expected filepath | map `track_id → path`                        | pure function                  |
| Scan filesystem           | check existence + size                       | depends only on disk state     |
| Apply label policy        | drop/flag missing label                      | pure rule                      |
| Apply split policy        | join official split or compute grouped split | must be stable (fixed seed)    |
| Optional decode probe     | attempt decode to catch corrupt audio        | cache results                  |
| Assign reason_code        | exactly one code per row                     | deterministic rules            |
| Write parquet             | stable sort by `track_id`                    | reproducible output            |

---

## Quality gates (simple checks before you trust the metadata manifest)

Run these checks and record them in a small build log.

| Gate                 | What you compute                                 | Why it matters                           |
| -------------------- | ------------------------------------------------ | ---------------------------------------- |
| Alignment rate       | `OK_count / total_metadata_candidates`           | tells you how “complete” your dataset is |
| Missing audio count  | `reason_code == NO_AUDIO_FILE`                   | catches path mistakes / partial data     |
| Missing label count  | `NO_LABEL`                                       | label policy sanity                      |
| Split distribution   | counts per split and per genre                   | prevents silent skew                     |
| Artist leakage check | `artists(train) ∩ artists(test)` should be empty | prevents inflated performance            |

---

## Practical naming + storage conventions

* Artifact filenames always include the subset tag: `metadata_manifest_{subset}.parquet`, `train_{subset}.parquet`, `val_{subset}.parquet`, `test_{subset}.parquet`, `metadata_manifest_config_{subset}.json`, `metadata_manifest_report_{subset}.txt`.
* Store all produced artifacts under a shared output directory, e.g. `MelCNN-MGR/data/processed/`; the subset suffix makes different runs coexist safely.
* The config snapshot (`metadata_manifest_config_{subset}.json`) records all CLI arguments + timestamp for reproducibility.
* The report (`metadata_manifest_report_{subset}.txt`) contains reason-code counts, split distribution, genre distribution, and the artist-leakage check.

This is boring infrastructure — and boring infrastructure is what makes experiments trustworthy.

---

## fma_medium: verified ground truth

These facts were verified directly against the actual data files (`tracks.csv` + filesystem scan).

| Fact | Value |
| ---- | ----- |
| Total tracks in medium subset | 17 000 |
| Audio files present on disk | 17 000 (100 %) |
| Tracks with `genre_top` label | 17 000 (0 nulls) |
| Genres (`genre_top` values) | 16 (Rock, Electronic, Experimental, Hip-Hop, Classical, Folk, Old-Time/Historic, Jazz, Instrumental, Pop, Country, Soul-RnB, Spoken, Blues, Easy Listening, International) |
| Split: training | 13 522 |
| Split: validation | 1 705 |
| Split: test | 1 773 |
| Artist overlap across splits | 0 (train∩val=0, train∩test=0, val∩test=0) |
| Duration range | 60 s – 600 s (mean ≈ 230 s) |
| Standard clip window | 29.98 s (1 321 967 samples at 44 100 Hz) |
| Dominant bit rates | 320 kbps (44 %), 256 kbps (22 %), 192 kbps (13 %); many VBR files |
| `tracks.csv` column for split | `('set', 'split')` — values: `training`, `validation`, `test` |
| `tracks.csv` column for subset | `('set', 'subset')` — values: `small`, `medium`, `large` (ordered categorical) |
| `tracks.csv` column for label | `('track', 'genre_top')` |
| `tracks.csv` column for artist | `('artist', 'id')` |
| `tracks.csv` column for duration | `('track', 'duration')` (integer seconds from metadata) |
