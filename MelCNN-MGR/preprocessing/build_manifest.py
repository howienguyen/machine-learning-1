#!/usr/bin/env python3
"""
build_manifest.py
=================
Builds ``metadata_manifest.parquet`` for FMA — the single source of truth
for the training pipeline.

Implements the four phases from docs/MelCNN-MGR-Preprocessing.md::

    Phase A  Collect candidates from tracks.csv (metadata-first)
    Phase B  Resolve filepaths and check filesystem existence
    Phase C  Assign exactly one reason_code per row
    Phase D  Write parquet outputs + config snapshot + report

Usage
-----
    # Default (small subset, fma_small audio, no decode probe)
    python MelCNN-MGR/preprocessing/build_manifest.py

    # Full options
    python MelCNN-MGR/preprocessing/build_manifest.py \\
        --subset medium \\
        --audio-root   FMA/fma_medium \\
        --metadata-root FMA/fma_metadata \\
        --out-dir      MelCNN-MGR/data/processed \\
        --min-duration 30 \\
        --decode-probe

Outputs (written to --out-dir, all names include the subset tag)
----------------------------------------------------------------
    metadata_manifest_{subset}.parquet     all candidates with reason_code
    train_{subset}.parquet                 reason_code==OK, split==training
    val_{subset}.parquet                   reason_code==OK, split==validation
    test_{subset}.parquet                  reason_code==OK, split==test
    metadata_manifest_config_{subset}.json config snapshot (reproducibility)
    metadata_manifest_report_{subset}.txt  quality-gate summary

All parquet outputs include two identity columns:
    sample_id    globally unique sample key, e.g. "fma:12345"
    source       dataset/source name, e.g. "fma"

Reason codes
------------
    OK                  usable sample
    NOT_IN_SUBSET       track is not in the requested subset
    NO_AUDIO_FILE       metadata row exists but MP3 is missing
    NO_LABEL            genre_top is null/empty
    EXCLUDED_LABEL      genre_top is intentionally excluded from training
    NO_SPLIT            split value is missing or unrecognised
    TOO_SHORT           duration < min_duration_s
    DECODE_FAIL         file exists but cannot be decoded (--decode-probe only)
    DUPLICATE_TRACK_ID  duplicate index (sanity check)

Column reference (tracks.csv multi-index → manifest flat names)
---------------------------------------------------------------
    ('set',    'split')      → split
    ('set',    'subset')     → subset
    ('track',  'genre_top')  → genre_top
    ('track',  'duration')   → duration_s   (integer seconds, from metadata)
    ('track',  'bit_rate')   → bit_rate      (-1 = unknown; many VBR non-standard values)
    ('artist', 'id')         → artist_id
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ── Paths (relative to this file) ─────────────────────────────────────────────
_SCRIPT_DIR    = Path(__file__).resolve().parent          # …/preprocessing
_MELCNN_DIR    = _SCRIPT_DIR.parent                       # …/MelCNN-MGR
_WORKSPACE     = _MELCNN_DIR.parent                       # …/machine-learning-1

DEFAULT_METADATA_ROOT = _WORKSPACE / "FMA" / "fma_metadata"
DEFAULT_OUT_DIR       = _MELCNN_DIR / "data" / "processed"
DEFAULT_SUBSET        = "small"   # one of "small", "medium", "large"
DEFAULT_SOURCE_NAME   = "fma"

DEFAULT_MIN_DURATION  = 30       # seconds — standard FMA clip window

# ── Multi-index column keys (tracks.csv uses a 2-level header) ────────────────
_COL_SPLIT     = ("set",    "split")
_COL_SUBSET    = ("set",    "subset")
_COL_GENRE_TOP = ("track",  "genre_top")
_COL_DURATION  = ("track",  "duration")
_COL_BIT_RATE  = ("track",  "bit_rate")
_COL_ARTIST_ID = ("artist", "id")

# List-valued columns that are stored as Python-literal strings in the CSV
_LIST_COLS = [
    ("track",  "genres"),
    ("track",  "genres_all"),
    ("track",  "tags"),
    ("album",  "tags"),
    ("artist", "tags"),
]

# Valid split labels used by FMA
_VALID_SPLITS = frozenset({"training", "validation", "test"})

# Labels intentionally excluded from train/val/test exports
_EXCLUDED_GENRE_TOP_LABELS = frozenset({"International"})
# _EXCLUDED_GENRE_TOP_LABELS = frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_text(value: object) -> str:
    """Return a stripped, case-folded string for robust text comparisons."""
    if pd.isna(value):
        return ""
    return str(value).strip().casefold()


def _is_excluded_genre_top(value: object) -> bool:
    """Return True if *value* is in the configured excluded-label set."""
    return _normalize_text(value) in _EXCLUDED_GENRE_TOP_LABELS

def get_audio_path(audio_dir: Path, track_id: int) -> Path:
    """Return the expected MP3 path for a given FMA track_id.

    FMA convention (from fma-repo/utils.py):
        tid_str = '{:06d}'.format(track_id)
        path    = audio_dir / tid_str[:3] / (tid_str + '.mp3')

    Examples
    --------
    >>> get_audio_path(Path('/data/fma_medium'), 2)
    PosixPath('/data/fma_medium/000/000002.mp3')
    >>> get_audio_path(Path('/data/fma_medium'), 1000)
    PosixPath('/data/fma_medium/001/001000.mp3')
    """
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


def make_sample_id(source_name: str, track_id: object) -> str:
    """Return a stable globally unique sample identifier."""
    return f"{source_name}:{track_id}"


def default_audio_root_for_subset(subset: str) -> Path:
    """Return the default FMA audio directory for a given subset label."""
    return _WORKSPACE / "FMA" / f"fma_{subset}"


# ─────────────────────────────────────────────────────────────────────────────
# Phase A — Collect candidates (metadata-first)
# ─────────────────────────────────────────────────────────────────────────────

def load_tracks(metadata_root: Path) -> pd.DataFrame:
    """Load ``tracks.csv`` with its two-level column header.

    The CSV uses a multi-index header (rows 0 and 1) and ``track_id`` as the
    index (column 0).  Several columns contain Python-literal list strings
    (e.g. ``"[21, 45]"``); those are parsed with ``ast.literal_eval`` so
    downstream code gets real Python lists.

    Returns a DataFrame with a MultiIndex column and ``track_id`` as index.
    """
    path = metadata_root / "tracks.csv"
    if not path.exists():
        raise FileNotFoundError(f"tracks.csv not found at {path}")

    logging.info("Loading %s", path)
    tracks = pd.read_csv(path, index_col=0, header=[0, 1])
    tracks.index.name = "track_id"

    # Parse list-valued columns that are stored as string literals
    for col in _LIST_COLS:
        if col in tracks.columns:
            tracks[col] = tracks[col].map(
                lambda v: ast.literal_eval(v) if isinstance(v, str) else v
            )

    logging.info("  Loaded %d total tracks", len(tracks))
    return tracks


def phase_a_collect(tracks: pd.DataFrame, subset: str, source_name: str) -> pd.DataFrame:
    """Extract the columns needed for the manifest and flag subset membership.

    Returns a flat DataFrame keyed by ``track_id``.

    Adds two flat identity columns for downstream multi-source manifests:
        sample_id    globally unique sample identifier
        source       dataset/source name
    """
    logging.info("Phase A: collecting candidates for subset='%s'", subset)

    needed = [_COL_SPLIT, _COL_SUBSET, _COL_GENRE_TOP,
              _COL_DURATION, _COL_BIT_RATE, _COL_ARTIST_ID]

    # Keep only the columns we need (all are present in every FMA tracks.csv)
    df = tracks[needed].copy()
    df.columns = ["split", "subset", "genre_top", "duration_s", "bit_rate", "artist_id"]
    df.insert(0, "source", source_name)
    df.insert(0, "sample_id", [make_sample_id(source_name, int(track_id)) for track_id in df.index])

    # Cast types
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df["bit_rate"]   = pd.to_numeric(df["bit_rate"],   errors="coerce")
    df["artist_id"]  = pd.to_numeric(df["artist_id"],  errors="coerce")

    # Flag subset membership (used by Phase C reason assignment)
    df["_in_target_subset"] = df["subset"] == subset

    # Sanity check: duplicate track_ids
    dupes = df.index.duplicated().sum()
    if dupes:
        logging.warning("  %d duplicate track_id(s) detected", dupes)

    sample_id_dupes = df["sample_id"].duplicated().sum()
    if sample_id_dupes:
        logging.warning("  %d duplicate sample_id(s) detected", sample_id_dupes)

    n_in_subset = df["_in_target_subset"].sum()
    logging.info("  Total rows: %d | In subset '%s': %d", len(df), subset, n_in_subset)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — Resolve filepaths (filesystem alignment)
# ─────────────────────────────────────────────────────────────────────────────

def phase_b_resolve(df: pd.DataFrame, audio_root: Path) -> pd.DataFrame:
    """Compute expected filepath for each track_id and stat the file.

    Adds columns:
        filepath        (str)  absolute path to the expected MP3
        audio_exists    (bool) whether the file is present
        filesize_bytes  (int | NaN) file size in bytes; NaN if missing
    """
    logging.info("Phase B: resolving filepaths under %s", audio_root)

    filepaths: dict[int, str] = {}
    exists_map: dict[int, bool] = {}
    sizes: dict[int, Optional[int]] = {}

    for tid in df.index:
        p = get_audio_path(audio_root, int(tid))
        filepaths[tid] = str(p)
        try:
            sizes[tid] = p.stat().st_size
            exists_map[tid] = True
        except FileNotFoundError:
            exists_map[tid] = False
            sizes[tid] = None

    df = df.copy()
    df["filepath"]       = pd.Series(filepaths,  index=df.index)
    df["audio_exists"]   = pd.Series(exists_map, index=df.index)
    df["filesize_bytes"] = pd.Series(sizes,       index=df.index)

    n_present = sum(exists_map.values())
    logging.info("  Audio files found: %d / %d", n_present, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Phase C — Apply rules (filtering + reason codes)
# ─────────────────────────────────────────────────────────────────────────────

def _assign_reason(row: pd.Series, min_duration_s: int) -> str:
    """Return the single reason_code that best explains a row's status.

    Priority (first rule that fires wins):
        NOT_IN_SUBSET > NO_AUDIO_FILE > NO_LABEL > EXCLUDED_LABEL > NO_SPLIT > TOO_SHORT > OK
    """
    if not row["_in_target_subset"]:
        return "NOT_IN_SUBSET"
    if not row["audio_exists"]:
        return "NO_AUDIO_FILE"
    if pd.isna(row["genre_top"]) or str(row["genre_top"]).strip() == "":
        return "NO_LABEL"
    if _is_excluded_genre_top(row["genre_top"]):
        return "EXCLUDED_LABEL"
    if pd.isna(row["split"]) or row["split"] not in _VALID_SPLITS:
        return "NO_SPLIT"
    if not pd.isna(row["duration_s"]) and row["duration_s"] < min_duration_s:
        return "TOO_SHORT"
    return "OK"


def phase_c_assign_reason(df: pd.DataFrame, min_duration_s: int) -> pd.DataFrame:
    """Assign exactly one ``reason_code`` per row (vectorised where possible).

    Rows marked ``EXCLUDED_LABEL`` remain in the full manifest for auditability
    but are excluded from the train/val/test parquet exports because only
    ``reason_code == 'OK'`` rows are written to those split files.
    """
    logging.info("Phase C: assigning reason codes (min_duration=%ds)", min_duration_s)
    df = df.copy()
    df["reason_code"] = df.apply(_assign_reason, axis=1, min_duration_s=min_duration_s)

    counts = df["reason_code"].value_counts()
    for code, cnt in counts.items():
        logging.info("  %-22s %d", code, cnt)
    return df


def phase_c_decode_probe(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to open every OK audio file and record sample_rate / channels.

    Files that cannot be opened are re-coded as DECODE_FAIL.
    Uses ``soundfile.info()`` — fast header-only read, no full decode.

    Adds columns (only populated for files that were successfully probed):
        sample_rate   (int | NaN)
        channels      (int | NaN)
    """
    try:
        import soundfile as sf
    except ImportError:
        logging.error("soundfile is required for --decode-probe. "
                      "Install it with: pip install soundfile")
        sys.exit(1)

    candidates = df[df["reason_code"] == "OK"]
    total = len(candidates)
    logging.info("Phase C (decode probe): probing %d files …", total)

    sample_rates: dict[int, Optional[int]] = {}
    channels_map: dict[int, Optional[int]] = {}
    fail_ids: list[int] = []

    for i, (tid, row) in enumerate(candidates.iterrows()):
        try:
            info = sf.info(row["filepath"])
            sample_rates[tid] = info.samplerate
            channels_map[tid] = info.channels
        except Exception as exc:
            fail_ids.append(tid)
            logging.warning("  DECODE_FAIL track_id=%d: %s", tid, exc)

        if (i + 1) % 1000 == 0:
            logging.info("  … probed %d / %d", i + 1, total)

    df = df.copy()
    if fail_ids:
        df.loc[fail_ids, "reason_code"] = "DECODE_FAIL"

    df["sample_rate"] = pd.Series(sample_rates, dtype="Int64")
    df["channels"]    = pd.Series(channels_map, dtype="Int64")

    logging.info("  DECODE_FAIL: %d", len(fail_ids))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def _write_report(df: pd.DataFrame, ok: pd.DataFrame, path: Path, subset: Optional[str] = None) -> None:
    """Write a human-readable quality-gate summary to *path*.

    If *subset* is provided, it will be shown in the report header so readers
    know which target subset (small/medium/large) was used to build the
    manifest.
    """
    lines: list[str] = []
    sep = "=" * 62

    in_subset = int((df["reason_code"] != "NOT_IN_SUBSET").sum())
    lines += [
        sep,
        "metadata_manifest build report",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"Target subset: {subset if subset is not None else 'N/A'}",
        sep, "",
        f"Total rows in tracks.csv  : {len(df):>7,}",
        f"In target subset          : {in_subset:>7,}",
        f"OK (usable)               : {len(ok):>7,}",
        f"Subset alignment rate     : {len(ok)/in_subset*100 if in_subset else 0:>6.1f}%"
        f"  (OK / in_subset)",
        "",
        "── Reason code counts ──────────────────────────────────",
    ]
    for code, cnt in df["reason_code"].value_counts().items():
        lines.append(f"  {code:<22} {cnt:>7,}")

    lines += ["", "── Split distribution (OK rows) ────────────────────────"]
    for split_name, cnt in ok["split"].value_counts().items():
        lines.append(f"  {split_name:<14} {cnt:>7,}")

    lines += ["", "── Genre distribution (OK rows) ────────────────────────"]
    for genre, cnt in ok["genre_top"].value_counts().items():
        lines.append(f"  {genre:<28} {cnt:>7,}")

    if "source" in ok.columns:
        lines += ["", "── Source distribution (OK rows) ───────────────────────"]
        for source, cnt in ok["source"].value_counts().items():
            lines.append(f"  {source:<28} {cnt:>7,}")

    lines += ["", "── Artist leakage check (OK rows) ──────────────────────"]
    splits = {
        s: set(ok[ok["split"] == s]["artist_id"].dropna())
        for s in ("training", "validation", "test")
    }
    for a, b in [("training", "validation"), ("training", "test"), ("validation", "test")]:
        overlap = splits[a] & splits[b]
        status = "OK — no overlap" if not overlap else f"!! {len(overlap)} shared artists"
        lines.append(f"  {a} ∩ {b}: {len(overlap)} ({status})")

    if "sample_rate" in ok.columns:
        lines += ["", "── Decode probe — sample rates (OK rows) ───────────────"]
        for sr, cnt in ok["sample_rate"].value_counts(dropna=False).items():
            lines.append(f"  {str(sr):<10} {cnt:>7,}")

    lines += ["", sep]
    path.write_text("\n".join(lines) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Phase D — Freeze outputs
# ─────────────────────────────────────────────────────────────────────────────

def phase_d_write(df: pd.DataFrame, out_dir: Path, config: dict) -> None:
    """Write parquet files, config snapshot, and build report.

    Outputs (all filenames include the ``subset`` tag from ``config['subset']``)
    ---------------------------------------------------------------------------
    metadata_manifest_{subset}.parquet     full table, all rows, stable sort by track_id
    train_{subset}.parquet                 OK + split==training
    val_{subset}.parquet                   OK + split==validation
    test_{subset}.parquet                  OK + split==test
    metadata_manifest_config_{subset}.json config snapshot
    metadata_manifest_report_{subset}.txt  quality-gate summary
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Phase D: writing outputs to %s", out_dir)

    # Drop internal helper column before writing
    df_out = df.drop(columns=["_in_target_subset"], errors="ignore")

    # Sort by sample_id when available so mixed-source manifests remain stable.
    if "sample_id" in df_out.columns:
        df_out = df_out.sort_values("sample_id", kind="stable")
    else:
        df_out = df_out.sort_index()

    # Determine subset suffix for filenames
    subset_name = config.get("subset") if isinstance(config, dict) else None
    subset_suffix = f"_{subset_name}" if subset_name else ""

    # Full manifest
    manifest_path = out_dir / f"metadata_manifest{subset_suffix}.parquet"
    df_out.to_parquet(manifest_path, engine="pyarrow", index=True)
    logging.info("  %-40s (%d rows)", str(manifest_path), len(df_out))

    # Per-split parquets (OK rows only) — include target subset in filename
    ok = df_out[df_out["reason_code"] == "OK"]

    split_files = [("train", "training"), ("val", "validation"), ("test", "test")]
    for fname, split_key in split_files:
        split_df = ok[ok["split"] == split_key]
        out_name = f"{fname}{subset_suffix}.parquet"
        path = out_dir / out_name
        split_df.to_parquet(path, engine="pyarrow", index=True)
        logging.info("  %-40s (%d rows)", str(path), len(split_df))

    # Config snapshot
    config_path = out_dir / f"metadata_manifest_config{subset_suffix}.json"
    with open(config_path, "w") as fh:
        json.dump(config, fh, indent=2)
    logging.info("  %s", config_path)

    # Build report
    report_path = out_dir / f"metadata_manifest_report{subset_suffix}.txt"
    _write_report(df_out, ok, report_path, subset_name)
    logging.info("  %s", report_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build metadata_manifest.parquet for FMA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--subset",
        default=DEFAULT_SUBSET,
        choices=["tiny", "small", "medium", "large"],
        help="FMA subset to include as OK candidates.",
    )
    p.add_argument(
        "--audio-root",
        default=None,
        help="Path to the audio folder. Defaults to FMA/fma_{subset} when omitted.",
    )
    p.add_argument(
        "--source-name",
        default=DEFAULT_SOURCE_NAME,
        help="Source label written to the manifest identity columns.",
    )
    p.add_argument(
        "--metadata-root",
        default=str(DEFAULT_METADATA_ROOT),
        help="Path to fma_metadata folder.",
    )
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Output directory for parquet files and report.",
    )
    p.add_argument(
        "--min-duration",
        type=int,
        default=DEFAULT_MIN_DURATION,
        dest="min_duration",
        help="Minimum track duration in seconds; shorter tracks get TOO_SHORT.",
    )
    p.add_argument(
        "--decode-probe",
        action="store_true",
        help="Open each OK audio file (soundfile.info) to catch decode errors "
             "and record sample_rate / channels in the manifest.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    resolved_audio_root = args.audio_root or str(default_audio_root_for_subset(args.subset))

    config = {
        "subset":         args.subset,
        "source_name":    args.source_name,
        "audio_root":     resolved_audio_root,
        "metadata_root":  args.metadata_root,
        "out_dir":        args.out_dir,
        "min_duration_s": args.min_duration,
        "excluded_genre_top_labels": sorted(_EXCLUDED_GENRE_TOP_LABELS),
        "decode_probe":   args.decode_probe,
        "generated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    audio_root    = Path(resolved_audio_root)
    metadata_root = Path(args.metadata_root)
    out_dir       = Path(args.out_dir)

    t0 = time.time()

    # ── Phase A ───────────────────────────────────────────────────────────────
    tracks = load_tracks(metadata_root)
    df = phase_a_collect(tracks, args.subset, args.source_name)

    # ── Phase B ───────────────────────────────────────────────────────────────
    df = phase_b_resolve(df, audio_root)

    # ── Phase C ───────────────────────────────────────────────────────────────
    df = phase_c_assign_reason(df, args.min_duration)
    if args.decode_probe:
        df = phase_c_decode_probe(df)

    # ── Phase D ───────────────────────────────────────────────────────────────
    phase_d_write(df, out_dir, config)

    ok_count = (df["reason_code"] == "OK").sum()
    elapsed  = time.time() - t0
    logging.info(
        "Done in %.1fs — %d usable samples out of %d candidates.",
        elapsed, ok_count, len(df),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
