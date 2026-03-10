#!/usr/bin/env python3
"""Build file-level and sample-level manifests across all supported datasets.

Outputs written to MelCNN-MGR/data/processed by default:
    manifest_all_datasets.parquet  one row per discovered audio file candidate
    manifest_all_samples.parquet   one row per fixed-length sample segment
    manifest_final_samples.parquet selected sample segments with final split labels

The script uses settings.data_sampling_settings in MelCNN-MGR/settings.json:
    target_genres
    sample_length_sec

Sampling rule
-------------
For each eligible audio file, the number of segments is:

    floor(duration_s / sample_length_sec)

Files shorter than sample_length_sec remain in manifest_all_datasets.parquet
with sampling_eligible=False, but they do not produce rows in
manifest_all_samples.parquet.
"""

from __future__ import annotations

import ast
import argparse
import json
import logging
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Iterable
import re

import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
_MELCNN_DIR = _SCRIPT_DIR.parent
_WORKSPACE = _MELCNN_DIR.parent

DEFAULT_SETTINGS_PATH = _MELCNN_DIR / "settings.json"
DEFAULT_PROCESSED_DIR = _MELCNN_DIR / "data" / "processed"
DEFAULT_MEDIUM_MANIFEST = DEFAULT_PROCESSED_DIR / "metadata_manifest_medium.parquet"
DEFAULT_ADDITIONAL_ROOT = _WORKSPACE / "additional_datasets" / "data"
DEFAULT_ALL_DATASETS_PATH = DEFAULT_PROCESSED_DIR / "manifest_all_datasets.parquet"
DEFAULT_ALL_SAMPLES_PATH = DEFAULT_PROCESSED_DIR / "manifest_all_samples.parquet"
DEFAULT_FINAL_SAMPLES_PATH = DEFAULT_PROCESSED_DIR / "manifest_final_samples.parquet"

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aif", ".aiff", ".au", ".m4a"}
_FFPROBE = shutil.which("ffprobe")
_SOUNDFILE_FIRST_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".au", ".ogg"}

# ── FMA-specific constants (ported from build_manifest.py) ────────────────────
DEFAULT_FMA_METADATA_ROOT = _WORKSPACE / "FMA" / "fma_metadata"
DEFAULT_FMA_AUDIO_ROOT = _WORKSPACE / "FMA" / "fma_medium"
DEFAULT_FMA_SUBSET = "medium"
DEFAULT_MIN_DURATION_DELTA = 0.001
DEFAULT_NUMBER_OF_SAMPLES_EXPECTED_EACH_GENRE = 1300
DEFAULT_TRAIN_RATIO_EACH_GENRE = 0.8
DEFAULT_SPLIT_SEED = 1337

_VALID_SPLITS = frozenset({"training", "validation", "test"})
_EXCLUDED_GENRE_TOP_LABELS = frozenset({"International"})
_FINAL_SPLIT_ORDER = ("training", "validation", "test")
_SEGMENT_SUFFIX_RE = re.compile(r":seg\d+$")

# Multi-index column keys used by FMA tracks.csv
_COL_SPLIT = ("set", "split")
_COL_SUBSET = ("set", "subset")
_COL_GENRE_TOP = ("track", "genre_top")
_COL_DURATION = ("track", "duration")
_COL_BIT_RATE = ("track", "bit_rate")
_COL_ARTIST_ID = ("artist", "id")

# List-valued columns that are stored as Python-literal strings in tracks.csv
_FMA_LIST_COLS = [
    ("track", "genres"),
    ("track", "genres_all"),
    ("track", "tags"),
    ("album", "tags"),
    ("artist", "tags"),
]

# Cached soundfile module (lazy-loaded once)
_sf_module = None


def _get_soundfile():
    """Lazy-load soundfile module once and cache it."""
    global _sf_module
    if _sf_module is None:
        try:
            import soundfile as sf
            _sf_module = sf
        except ImportError:
            _sf_module = False  # sentinel: tried and failed
    return _sf_module if _sf_module is not False else None


def derive_default_min_duration(sample_length_sec: float, min_duration_delta: float) -> float:
    """Derive FMA filter threshold from sample length.

    Use sample_length_sec - min_duration_delta when possible so a nominal clip
    clears the filter while sample segmentation still uses the exact
    sample_length_sec value.
    """
    if sample_length_sec > min_duration_delta:
        return sample_length_sec - min_duration_delta
    return sample_length_sec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all-datasets and all-samples manifests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS_PATH),
        help="Path to MelCNN-MGR/settings.json.",
    )
    parser.add_argument(
        "--medium-manifest",
        default=str(DEFAULT_MEDIUM_MANIFEST),
        help="Path to metadata_manifest_medium.parquet (used when available).",
    )
    parser.add_argument(
        "--fma-metadata-root",
        default=str(DEFAULT_FMA_METADATA_ROOT),
        help="Path to FMA metadata folder containing tracks.csv.",
    )
    parser.add_argument(
        "--fma-audio-root",
        default=str(DEFAULT_FMA_AUDIO_ROOT),
        help="Path to FMA audio folder (e.g. FMA/fma_medium).",
    )
    parser.add_argument(
        "--fma-subset",
        default=DEFAULT_FMA_SUBSET,
        choices=["small", "medium", "large"],
        help="FMA subset to use.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=None,
        dest="min_duration",
        help="Minimum track duration in seconds. Defaults to sample_length_sec - min_duration_delta when possible.",
    )
    parser.add_argument(
        "--force-rescan",
        action="store_true",
        help="Force FMA rebuild from tracks.csv even if manifest parquet exists.",
    )
    parser.add_argument(
        "--additional-root",
        default=str(DEFAULT_ADDITIONAL_ROOT),
        help="Path to additional_datasets/data.",
    )
    parser.add_argument(
        "--all-datasets-out",
        default=str(DEFAULT_ALL_DATASETS_PATH),
        help="Output path for manifest_all_datasets.parquet.",
    )
    parser.add_argument(
        "--all-samples-out",
        default=str(DEFAULT_ALL_SAMPLES_PATH),
        help="Output path for manifest_all_samples.parquet.",
    )
    parser.add_argument(
        "--final-samples-out",
        default=str(DEFAULT_FINAL_SAMPLES_PATH),
        help="Output path for manifest_final_samples.parquet.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="Random seed used for deterministic final split assignment.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _normalize_genre(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def load_data_sampling_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    raw_text = path.read_text()
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        try:
            payload = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError) as fallback_exc:
            raise ValueError(f"Invalid JSON in settings file: {path}") from fallback_exc

    config = payload.get("data_sampling_settings")
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected object at settings.data_sampling_settings in {path}"
        )

    target_genres = config.get("target_genres")
    if not isinstance(target_genres, list) or not target_genres or not all(
        isinstance(genre, str) and genre.strip() for genre in target_genres
    ):
        raise ValueError(
            f"Expected non-empty string list at settings.data_sampling_settings.target_genres in {path}"
        )

    sample_length_sec = config.get("sample_length_sec")
    if not isinstance(sample_length_sec, (int, float)) or sample_length_sec <= 0:
        raise ValueError(
            f"Expected positive number at settings.data_sampling_settings.sample_length_sec in {path}"
        )

    min_duration_delta = config.get("min_duration_delta", DEFAULT_MIN_DURATION_DELTA)
    if not isinstance(min_duration_delta, (int, float)) or min_duration_delta < 0:
        min_duration_delta = DEFAULT_MIN_DURATION_DELTA

    number_of_samples_expected_each_genre = config.get(
        "number_of_samples_expected_each_genre",
        DEFAULT_NUMBER_OF_SAMPLES_EXPECTED_EACH_GENRE,
    )
    if (
        not isinstance(number_of_samples_expected_each_genre, (int, float))
        or number_of_samples_expected_each_genre <= 0
    ):
        number_of_samples_expected_each_genre = DEFAULT_NUMBER_OF_SAMPLES_EXPECTED_EACH_GENRE

    train_ratio_each_genre = config.get(
        "train_n_val_test_split_ratio_each_genre",
        DEFAULT_TRAIN_RATIO_EACH_GENRE,
    )
    if (
        not isinstance(train_ratio_each_genre, (int, float))
        or train_ratio_each_genre <= 0
        or train_ratio_each_genre > 1
    ):
        train_ratio_each_genre = DEFAULT_TRAIN_RATIO_EACH_GENRE

    return {
        "target_genres": list(dict.fromkeys(genre.strip() for genre in target_genres)),
        "sample_length_sec": float(sample_length_sec),
        "min_duration_delta": float(min_duration_delta),
        "number_of_samples_expected_each_genre": int(number_of_samples_expected_each_genre),
        "train_n_val_test_split_ratio_each_genre": float(train_ratio_each_genre),
    }


def _relative_to_workspace(path: Path) -> str | None:
    """Return workspace-relative path using string ops (avoids slow stat/resolve on WSL)."""
    path_str = str(path)
    workspace_prefix = str(_WORKSPACE)
    # Handle both with and without trailing separator
    if path_str.startswith(workspace_prefix + "/"):
        return path_str[len(workspace_prefix) + 1:]
    if path_str.startswith(workspace_prefix + "\\"):
        return path_str[len(workspace_prefix) + 1:].replace("\\", "/")
    if path_str == workspace_prefix:
        return ""
    return None


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _probe_duration_ffprobe(path: Path) -> tuple[float | None, str | None]:
    if not _FFPROBE:
        return None, None

    cmd = [
        _FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    except subprocess.TimeoutExpired:
        return None, None
    if result.returncode != 0:
        return None, None

    text = result.stdout.strip()
    if not text:
        return None, None

    try:
        return float(text), "ffprobe"
    except ValueError:
        return None, None


def _probe_duration_soundfile(path: Path) -> tuple[float | None, str | None]:
    sf = _get_soundfile()
    if sf is None:
        return None, None

    try:
        info = sf.info(str(path))
    except Exception:
        return None, None

    if not info.samplerate:
        return None, None
    return float(info.frames) / float(info.samplerate), "soundfile"


def _normalize_audio_duration(duration_s: float | None) -> float | None:
    if duration_s is None:
        return None
    if duration_s <= 0:
        return float(duration_s)
    # Reduce measured durations to one-decimal precision before rounding up to
    # the whole-second value used for segmentation, so values like 29.900011s
    # and 29.988571s both normalize to 30s.
    return float(math.ceil(round(float(duration_s), 1) - 1e-6))


def probe_audio_duration(path: Path) -> tuple[float | None, str | None]:
    probes = [_probe_duration_ffprobe, _probe_duration_soundfile]
    if path.suffix.lower() in _SOUNDFILE_FIRST_EXTENSIONS:
        probes = [_probe_duration_soundfile, _probe_duration_ffprobe]

    for probe in probes:
        duration, source = probe(path)
        if duration is not None:
            return float(duration), source

    return None, None


def _probe_durations_for_frame(
    frame: pd.DataFrame,
    log_prefix: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Probe actual audio durations for rows with filepaths.

    Returns actual durations, normalized durations, and probe sources aligned to frame.index.
    """
    t0 = time.time()
    actual_durations: dict[object, float | None] = {}
    normalized_durations: dict[object, float | None] = {}
    duration_sources: dict[object, str] = {}
    total = len(frame)
    log_every = min(500, max(100, total // 20)) if total > 0 else 100
    logging.info(
        "%s Starting audio duration probe for %d file(s) (log every %d)",
        log_prefix,
        total,
        log_every,
    )
    for i, row in enumerate(frame.itertuples(index=True), start=1):
        duration_s, duration_source = probe_audio_duration(Path(str(row.filepath)))
        actual_durations[row.Index] = duration_s
        normalized_durations[row.Index] = _normalize_audio_duration(duration_s)
        duration_sources[row.Index] = duration_source or "unknown"
        if i % log_every == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0.0
            eta = (total - i) / rate if rate > 0 else 0.0
            logging.info(
                "%s Probed %d/%d audio durations (%.1f%% | %.1fs elapsed | ETA %.1fs)",
                log_prefix,
                i,
                total,
                (100.0 * i / total) if total else 100.0,
                elapsed,
                eta,
            )

    return (
        pd.Series(actual_durations, index=frame.index, dtype="float64"),
        pd.Series(normalized_durations, index=frame.index, dtype="float64"),
        pd.Series(duration_sources, index=frame.index, dtype="string"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FMA from-scratch loading (eliminates dependency on build_manifest.py)
# ─────────────────────────────────────────────────────────────────────────────

def _get_fma_audio_path(audio_dir: Path, track_id: int) -> Path:
    """Return the expected MP3 path for an FMA track_id.

    FMA convention: tid_str = '{:06d}'.format(track_id)
                    path    = audio_dir / tid_str[:3] / (tid_str + '.mp3')
    """
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


def _load_tracks_csv(metadata_root: Path) -> pd.DataFrame:
    """Load FMA tracks.csv with its two-level column header."""
    path = metadata_root / "tracks.csv"
    if not path.exists():
        raise FileNotFoundError(f"tracks.csv not found at {path}")

    logging.info("[FMA-rescan] Loading %s", path)
    t0 = time.time()
    tracks = pd.read_csv(path, index_col=0, header=[0, 1])
    tracks.index.name = "track_id"

    for col in _FMA_LIST_COLS:
        if col in tracks.columns:
            tracks[col] = tracks[col].map(
                lambda v: ast.literal_eval(v) if isinstance(v, str) else v
            )

    logging.info("[FMA-rescan] Loaded %d tracks in %.1fs", len(tracks), time.time() - t0)
    return tracks


def _assign_fma_reason(row: pd.Series, min_duration_s: float) -> str:
    """Return a single reason_code for an FMA candidate row.

    Priority: NO_AUDIO_FILE > AUDIO_READ_FAILED > NO_LABEL > EXCLUDED_LABEL > NO_SPLIT > TOO_SHORT > OK
    (NOT_IN_SUBSET is not needed because we pre-filter by subset.)
    """
    if not row["audio_exists"]:
        return "NO_AUDIO_FILE"
    if pd.isna(row["duration_s"]):
        return "AUDIO_READ_FAILED"
    if pd.isna(row["genre_top"]) or str(row["genre_top"]).strip() == "":
        return "NO_LABEL"
    genre_cf = str(row["genre_top"]).strip().casefold()
    if genre_cf in {g.casefold() for g in _EXCLUDED_GENRE_TOP_LABELS}:
        return "EXCLUDED_LABEL"
    if pd.isna(row["split"]) or row["split"] not in _VALID_SPLITS:
        return "NO_SPLIT"
    if not pd.isna(row["duration_s"]) and row["duration_s"] < min_duration_s:
        return "TOO_SHORT"
    return "OK"


def build_fma_candidates_from_scratch(
    metadata_root: Path,
    audio_root: Path,
    subset: str,
    target_genres: list[str],
    sample_length_sec: float,
    min_duration_s: float,
) -> pd.DataFrame:
    """Build FMA candidates directly from tracks.csv + filesystem.

    This replaces the need for a pre-built metadata_manifest_medium.parquet.
    Returns a DataFrame with the same schema as load_medium_candidates().
    """
    t0 = time.time()

    # Phase A: Load tracks.csv and extract needed columns
    tracks = _load_tracks_csv(metadata_root)

    needed = [_COL_SPLIT, _COL_SUBSET, _COL_GENRE_TOP,
              _COL_DURATION, _COL_BIT_RATE, _COL_ARTIST_ID]
    df = tracks[needed].copy()
    df.columns = ["split", "subset", "genre_top", "duration_s", "bit_rate", "artist_id"]
    df["duration_s"] = pd.to_numeric(df["duration_s"], errors="coerce")
    df["artist_id"] = pd.to_numeric(df["artist_id"], errors="coerce")

    logging.info("[FMA-rescan] Total tracks in CSV: %d", len(df))

    # Filter to target subset + genres early (avoids stat-ing 100K+ files)
    df = df[
        (df["subset"] == subset)
        & (df["genre_top"].isin(target_genres))
    ].copy()
    logging.info(
        "[FMA-rescan] After subset=%s + target_genres filter: %d tracks",
        subset, len(df),
    )

    if df.empty:
        logging.warning("[FMA-rescan] No tracks match filters — returning empty")
        return pd.DataFrame()

    # Phase B: Resolve filepaths and stat each file
    logging.info("[FMA-rescan] Resolving filepaths under %s ...", audio_root)
    filepaths: dict[int, str] = {}
    exists_map: dict[int, bool] = {}
    sizes: dict[int, int | None] = {}
    n_total = len(df)

    for i, tid in enumerate(df.index):
        p = _get_fma_audio_path(audio_root, int(tid))
        filepaths[tid] = str(p)
        try:
            sizes[tid] = p.stat().st_size
            exists_map[tid] = True
        except FileNotFoundError:
            exists_map[tid] = False
            sizes[tid] = None

        if (i + 1) % 2000 == 0 or (i + 1) == n_total:
            logging.info(
                "[FMA-rescan]   Resolved %d/%d filepaths (%.1fs)",
                i + 1, n_total, time.time() - t0,
            )

    df["filepath"] = pd.Series(filepaths, index=df.index)
    df["audio_exists"] = pd.Series(exists_map, index=df.index)
    df["filesize_bytes"] = pd.Series(sizes, index=df.index)

    n_present = sum(exists_map.values())
    logging.info("[FMA-rescan] Audio files found: %d / %d", n_present, n_total)

    # Keep only audio_exists == True (consistent with load_medium_candidates)
    n_before = len(df)
    df = df[df["audio_exists"] == True].copy()
    logging.info(
        "[FMA-rescan] Kept %d rows with audio_exists=True (dropped %d)",
        len(df), n_before - len(df),
    )

    if df.empty:
        return pd.DataFrame()

    # Map to unified schema
    source_label = f"fma-{subset}"
    df["source"] = source_label
    df["source_type"] = "fma"
    df["sample_id"] = [f"fma:{int(tid)}" for tid in df.index]
    df["source_track_id"] = df.index.astype(str)
    df["track_id"] = df.index.astype("Int64")
    df["relative_path"] = df["filepath"].map(
        lambda value: _relative_to_workspace(Path(str(value)))
    )
    df["file_ext"] = ".mp3"
    logging.info("[FMA-rescan] Probing actual audio durations ...")
    df["audio_duration_s"], df["duration_s"], df["duration_source"] = _probe_durations_for_frame(df, "[FMA-rescan]")
    df["manifest_origin"] = f"from_scratch:{metadata_root}"
    df["filesize_bytes"] = pd.to_numeric(
        df["filesize_bytes"], errors="coerce"
    ).astype("Int64")

    # Phase C: Assign reason codes using actual clip durations.
    logging.info("[FMA-rescan] Assigning reason codes (min_duration=%.2fs) ...", min_duration_s)
    df["reason_code"] = df.apply(_assign_fma_reason, axis=1, min_duration_s=min_duration_s)

    counts = df["reason_code"].value_counts()
    for code, cnt in counts.items():
        logging.info("[FMA-rescan]   %-22s %d", code, cnt)

    # Compute sampling eligibility
    logging.info("[FMA-rescan] Computing sampling eligibility ...")
    eligibility = []
    for reason_code, duration_s in zip(df["reason_code"], df["duration_s"]):
        normalized_duration = None if pd.isna(duration_s) else float(duration_s)
        eligibility.append(
            _apply_final_sampling_filter(
                str(reason_code),
                normalized_duration,
                sample_length_sec,
            )
        )
    df["sampling_eligible"] = [e[0] for e in eligibility]
    df["sampling_num_segments"] = pd.Series(
        [e[1] for e in eligibility], dtype="Int64"
    )
    df["sampling_exclusion_reason"] = [e[2] for e in eligibility]

    n_eligible = int(df["sampling_eligible"].sum())
    logging.info(
        "[FMA-rescan] Done: %d rows (%d eligible) in %.1fs",
        len(df), n_eligible, time.time() - t0,
    )
    genre_counts = df.groupby("genre_top").size().sort_values(ascending=False)
    if not genre_counts.empty:
        logging.info("[FMA-rescan] Audio/track counts by genre:\n%s", genre_counts.to_string())

    keep_columns = [
        "source", "source_type", "sample_id", "source_track_id",
        "track_id", "genre_top", "subset", "split", "artist_id", "filepath",
        "relative_path", "file_ext", "audio_exists", "filesize_bytes",
        "audio_duration_s", "duration_s", "duration_source", "reason_code", "sampling_eligible",
        "sampling_num_segments", "sampling_exclusion_reason", "manifest_origin",
    ]
    return df[keep_columns].reset_index(drop=True)


def build_fma_candidates(
    medium_manifest_path: Path,
    fma_metadata_root: Path,
    fma_audio_root: Path,
    fma_subset: str,
    target_genres: list[str],
    sample_length_sec: float,
    min_duration_s: float,
    force_rescan: bool = False,
) -> pd.DataFrame:
    """Smart dispatcher: use cached parquet if available, else build from scratch."""
    if not force_rescan and medium_manifest_path.exists():
        logging.info("[FMA] Using cached manifest: %s", medium_manifest_path)
        return load_medium_candidates(
            medium_manifest_path, fma_subset, target_genres, sample_length_sec, min_duration_s
        )

    if force_rescan:
        logging.info("[FMA] --force-rescan set, building from scratch")
    else:
        logging.info(
            "[FMA] Manifest not found at %s — building from scratch",
            medium_manifest_path,
        )
    return build_fma_candidates_from_scratch(
        fma_metadata_root, fma_audio_root, fma_subset,
        target_genres, sample_length_sec, min_duration_s,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_segment_count(duration_s: float | None, sample_length_sec: float) -> int:
    if duration_s is None or duration_s < sample_length_sec:
        return 0
    return int(math.floor(duration_s / sample_length_sec))


def _sampling_status(duration_s: float | None, sample_length_sec: float) -> tuple[bool, int, str | None]:
    if duration_s is None:
        return False, 0, "AUDIO_READ_FAILED"
    segment_count = _compute_segment_count(duration_s, sample_length_sec)
    if segment_count <= 0:
        return False, 0, "TOO_SHORT_FOR_SAMPLE_LENGTH"
    return True, segment_count, None


def _apply_final_sampling_filter(
    reason_code: str,
    duration_s: float | None,
    sample_length_sec: float,
) -> tuple[bool, int, str | None]:
    """Combine dataset-level filtering with segment-generation eligibility."""
    sampling_eligible, sampling_num_segments, sampling_exclusion_reason = _sampling_status(
        duration_s,
        sample_length_sec,
    )
    if reason_code != "OK":
        return False, 0, reason_code
    return sampling_eligible, sampling_num_segments, sampling_exclusion_reason


def _compute_final_split_targets(total_samples: int, train_ratio: float) -> dict[str, int]:
    """Compute per-split targets from a per-genre total and train ratio."""
    train_n = int(math.floor(total_samples * train_ratio))
    remaining = max(total_samples - train_n, 0)
    validation_n = remaining // 2
    test_n = remaining - validation_n
    return {
        "training": train_n,
        "validation": validation_n,
        "test": test_n,
    }


def _segment_group_id(sample_id: object) -> str:
    value = "" if sample_id is None else str(sample_id).strip()
    if not value or value.lower() == "nan":
        raise ValueError("Encountered missing sample_id while assigning final splits.")
    return _SEGMENT_SUFFIX_RE.sub("", value)


def assign_final_splits(
    all_samples: pd.DataFrame,
    number_of_samples_expected_each_genre: int,
    train_ratio_each_genre: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign final split labels by source sample identity and build the final samples manifest.

    All segments from the same source sample stay in the same split to avoid segment
    leakage across training / validation / test.
    """
    samples = all_samples.copy()
    if samples.empty:
        samples["final_split"] = pd.Series(dtype="string")
        samples["selected_for_final_manifest"] = pd.Series(dtype="bool")
        empty_summary = pd.DataFrame(
            columns=["genre_top", "training", "validation", "test", "selected_total", "target_total"]
        )
        return samples, samples.iloc[0:0].copy(), empty_summary

    samples["final_split"] = pd.Series([None] * len(samples), dtype="object")
    samples["selected_for_final_manifest"] = False
    samples["group_sample_id"] = samples["sample_id"].map(_segment_group_id)
    summary_rows: list[dict[str, object]] = []

    for genre_index, genre in enumerate(sorted(samples["genre_top"].dropna().unique())):
        genre_mask = samples["genre_top"] == genre
        genre_df = samples.loc[genre_mask].copy()
        target_total = min(number_of_samples_expected_each_genre, len(genre_df))
        targets = _compute_final_split_targets(target_total, train_ratio_each_genre)
        remaining = dict(targets)
        selected_counts = {split: 0 for split in _FINAL_SPLIT_ORDER}
        selected_total = 0

        grouped = (
            genre_df.groupby("group_sample_id", dropna=False)
            .agg(sample_count=("sample_id", "size"))
            .reset_index()
            .sample(frac=1.0, random_state=seed + genre_index)
            .reset_index(drop=True)
        )

        for group in grouped.itertuples(index=False):
            if selected_total >= target_total:
                break

            split_choice = max(
                _FINAL_SPLIT_ORDER,
                key=lambda split_name: (remaining[split_name], -_FINAL_SPLIT_ORDER.index(split_name)),
            )
            group_mask = genre_mask & (samples["group_sample_id"] == group.group_sample_id)
            samples.loc[group_mask, "final_split"] = split_choice
            samples.loc[group_mask, "selected_for_final_manifest"] = True

            selected_total += int(group.sample_count)
            selected_counts[split_choice] += int(group.sample_count)
            remaining[split_choice] -= int(group.sample_count)

        summary_rows.append(
            {
                "genre_top": genre,
                "training": selected_counts["training"],
                "validation": selected_counts["validation"],
                "test": selected_counts["test"],
                "selected_total": sum(selected_counts.values()),
                "target_total": target_total,
            }
        )

    final_samples = samples[samples["selected_for_final_manifest"] == True].copy()
    final_samples = final_samples.sort_values(
        ["genre_top", "final_split", "group_sample_id", "segment_index"],
        kind="stable",
    ).reset_index(drop=True)
    final_columns = [
        "sample_id", "source", "genre_top", "filepath", "track_id",
        "split", "sample_length_sec", "segment_index", "segment_start_sec",
        "segment_end_sec", "audio_duration_s", "reason_code",
        "final_split", "selected_for_final_manifest",
    ]
    samples = samples.drop(columns=["group_sample_id"])
    final_samples = final_samples[final_columns]
    split_summary = pd.DataFrame(summary_rows)
    return samples, final_samples, split_summary


def load_medium_candidates(
    path: Path,
    subset: str,
    target_genres: list[str],
    sample_length_sec: float,
    min_duration_s: float,
) -> pd.DataFrame:
    logging.info("[FMA-medium] Loading manifest from %s", path)
    t0 = time.time()
    if not path.exists():
        raise FileNotFoundError(f"Medium manifest not found: {path}")

    manifest = pd.read_parquet(path)
    logging.info("[FMA-medium] Loaded %d total rows in %.1fs", len(manifest), time.time() - t0)
    required_columns = {
        "sample_id", "source", "split", "subset", "genre_top", "duration_s",
        "artist_id", "filepath", "audio_exists", "filesize_bytes", "reason_code",
    }
    missing = sorted(required_columns.difference(manifest.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(missing)}")

    df = manifest[
        (manifest["subset"] == subset)
        & (manifest["genre_top"].isin(target_genres))
        & (manifest["audio_exists"] == True)
    ].copy()
    logging.info("[FMA-medium] Filtered to %d rows (subset=%s, target genres, audio_exists)", len(df), subset)

    if df.empty:
        return pd.DataFrame()

    logging.info("[FMA-medium] Computing derived columns ...")
    source_label = f"fma-{subset}"
    df["source"] = source_label
    df["source_type"] = "fma"
    df["sample_id"] = [f"fma:{int(track_id)}" for track_id in df.index.astype(int)]
    df["source_track_id"] = df.index.astype(str)
    df["track_id"] = df.index.astype("Int64")
    df["relative_path"] = df["filepath"].map(lambda value: _relative_to_workspace(Path(str(value))))
    df["file_ext"] = df["filepath"].map(lambda value: Path(str(value)).suffix.lower())
    logging.info("[FMA-medium] Probing actual audio durations ...")
    df["audio_duration_s"], df["duration_s"], df["duration_source"] = _probe_durations_for_frame(df, "[FMA-medium]")
    df["manifest_origin"] = str(path)
    df["filesize_bytes"] = pd.to_numeric(df["filesize_bytes"], errors="coerce").astype("Int64")
    df["reason_code"] = df.apply(_assign_fma_reason, axis=1, min_duration_s=min_duration_s)

    logging.info("[FMA-medium] Computing sampling eligibility ...")
    eligibility = []
    for reason_code, duration_s in zip(df["reason_code"], df["duration_s"]):
        normalized_duration = None if pd.isna(duration_s) else float(duration_s)
        eligibility.append(
            _apply_final_sampling_filter(
                str(reason_code),
                normalized_duration,
                sample_length_sec,
            )
        )
    df["sampling_eligible"] = [item[0] for item in eligibility]
    df["sampling_num_segments"] = pd.Series([item[1] for item in eligibility], dtype="Int64")
    df["sampling_exclusion_reason"] = [item[2] for item in eligibility]

    n_eligible = int(df["sampling_eligible"].sum())
    logging.info("[FMA-medium] Done: %d rows (%d eligible) in %.1fs", len(df), n_eligible, time.time() - t0)

    keep_columns = [
        "source", "source_type", "sample_id", "source_track_id", "track_id",
        "genre_top", "subset", "split", "artist_id", "filepath", "relative_path",
        "file_ext", "audio_exists", "filesize_bytes", "audio_duration_s", "duration_s", "duration_source",
        "reason_code", "sampling_eligible", "sampling_num_segments",
        "sampling_exclusion_reason", "manifest_origin",
    ]
    return df[keep_columns].reset_index(drop=True)


def _genre_alias_targets(folder_name: str, target_genres: list[str]) -> list[str]:
    normalized_targets = {_normalize_genre(genre): genre for genre in target_genres}
    folder_key = _normalize_genre(folder_name)
    aliases = {
        "hiphop": ["Hip-Hop"],
        "raphiphop": ["Hip-Hop"],
        "folkcountry": [genre for genre in ("Folk", "Country") if genre in target_genres],
    }

    if folder_key in aliases:
        return aliases[folder_key]

    mapped = normalized_targets.get(folder_key)
    return [mapped] if mapped else []


def _iter_audio_files(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS:
            yield path


def collect_additional_candidates(
    additional_root: Path,
    target_genres: list[str],
    sample_length_sec: float,
    min_duration_s: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if not additional_root.exists():
        logging.warning("Additional datasets root not found: %s", additional_root)
        return pd.DataFrame()

    t0 = time.time()
    logging.info("[Additional] Scanning %s ...", additional_root)
    total_files_probed = 0

    for source_dir in sorted(path for path in additional_root.iterdir() if path.is_dir()):
        logging.info("[Additional] Source: %s", source_dir.name)
        for genre_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            mapped_genres = _genre_alias_targets(genre_dir.name, target_genres)
            if not mapped_genres:
                logging.debug("[Additional]   Skipping %s (no genre match)", genre_dir.name)
                continue

            audio_files = list(_iter_audio_files(genre_dir))
            logging.info(
                "[Additional]   %s/%s → %s (%d files)",
                source_dir.name, genre_dir.name, mapped_genres, len(audio_files),
            )
            for i, audio_path in enumerate(audio_files):
                audio_duration_s, duration_source = probe_audio_duration(audio_path)
                duration_s = _normalize_audio_duration(audio_duration_s)
                if duration_s is None:
                    reason_code = "AUDIO_READ_FAILED"
                elif duration_s < min_duration_s:
                    reason_code = "TOO_SHORT"
                else:
                    reason_code = "OK"
                sampling_eligible, sampling_num_segments, sampling_exclusion_reason = _apply_final_sampling_filter(
                    reason_code,
                    duration_s,
                    sample_length_sec,
                )
                relative_path = _relative_to_workspace(audio_path)
                source_audio_id = relative_path or str(audio_path)
                total_files_probed += 1

                if (i + 1) % 50 == 0 or (i + 1) == len(audio_files):
                    logging.info(
                        "[Additional]     Probed %d/%d files in %s/%s (%.1fs elapsed)",
                        i + 1, len(audio_files), source_dir.name, genre_dir.name,
                        time.time() - t0,
                    )

                for genre in mapped_genres:
                    base_sample_id = f"{source_dir.name}:{genre}:{source_audio_id}"
                    rows.append(
                        {
                            "source": source_dir.name,
                            "source_type": "additional_dataset",
                            "sample_id": base_sample_id,
                            "source_track_id": None,
                            "track_id": None,
                            "genre_top": genre,
                            "subset": None,
                            "split": None,
                            "artist_id": None,
                            "filepath": str(audio_path),
                            "relative_path": relative_path,
                            "file_ext": audio_path.suffix.lower(),
                            "audio_exists": True,
                            "filesize_bytes": _safe_file_size(audio_path),
                            "audio_duration_s": audio_duration_s,
                            "duration_s": duration_s,
                            "duration_source": duration_source or "unknown",
                            "reason_code": reason_code,
                            "sampling_eligible": sampling_eligible,
                            "sampling_num_segments": sampling_num_segments,
                            "sampling_exclusion_reason": sampling_exclusion_reason,
                            "manifest_origin": str(source_dir),
                        }
                    )

    logging.info(
        "[Additional] Done: probed %d files, produced %d rows in %.1fs",
        total_files_probed, len(rows), time.time() - t0,
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["track_id"] = pd.Series(df["track_id"], dtype="Int64")
    df["artist_id"] = pd.Series(df["artist_id"], dtype="Int64")
    df["filesize_bytes"] = pd.Series(df["filesize_bytes"], dtype="Int64")
    df["sampling_num_segments"] = pd.Series(df["sampling_num_segments"], dtype="Int64")
    return df


def combine_dataset_manifest(*frames: pd.DataFrame) -> pd.DataFrame:
    logging.info("[Combine] Merging %d frame(s) ...", sum(1 for f in frames if f is not None and not f.empty))
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    columns = [
        "source", "source_type", "sample_id", "source_track_id", "track_id",
        "genre_top", "subset", "split", "artist_id", "filepath", "relative_path",
        "file_ext", "audio_exists", "filesize_bytes", "audio_duration_s", "duration_s", "duration_source",
        "reason_code", "sampling_eligible", "sampling_num_segments",
        "sampling_exclusion_reason", "manifest_origin",
    ]
    if not non_empty:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(non_empty, axis=0, ignore_index=True)
    before_dedup = len(combined)
    combined = combined[columns].drop_duplicates(subset=["sample_id"], keep="first")
    logging.info("[Combine] %d rows after concat, %d after dedup", before_dedup, len(combined))
    combined = combined.sort_values(["genre_top", "source", "sample_id"], kind="stable").reset_index(drop=True)
    combined["track_id"] = pd.Series(combined["track_id"], dtype="Int64")
    combined["artist_id"] = pd.Series(combined["artist_id"], dtype="Int64")
    combined["filesize_bytes"] = pd.Series(combined["filesize_bytes"], dtype="Int64")
    combined["sampling_num_segments"] = pd.Series(combined["sampling_num_segments"], dtype="Int64")
    return combined


def build_samples_manifest(all_datasets: pd.DataFrame, sample_length_sec: float) -> pd.DataFrame:
    logging.info("[Samples] Building sample segments (sample_length=%.1fs) ...", sample_length_sec)
    t0 = time.time()
    if all_datasets.empty:
        return pd.DataFrame(columns=[
            "sample_id", "source", "source_type", "genre_top", "filepath",
            "relative_path", "track_id", "subset", "split", "artist_id", "sample_length_sec",
            "segment_index", "segment_start_sec", "segment_end_sec", "total_segments_from_audio",
            "audio_duration_s", "file_ext", "reason_code", "final_split", "selected_for_final_manifest",
        ])

    rows: list[dict[str, object]] = []
    eligible = all_datasets[
        (all_datasets["reason_code"] == "OK")
        & (all_datasets["sampling_eligible"] == True)
        & (all_datasets["sampling_num_segments"].notna())
    ]
    n_eligible = len(eligible)
    logging.info("[Samples] %d eligible audio files to expand", n_eligible)
    for idx, record in enumerate(eligible.itertuples(index=False)):
        total_segments = int(record.sampling_num_segments)
        if total_segments <= 0:
            continue
        for segment_index in range(total_segments):
            start_sec = segment_index * sample_length_sec
            end_sec = start_sec + sample_length_sec
            rows.append(
                {
                    "sample_id": f"{record.sample_id}:seg{segment_index:04d}",
                    "source": record.source,
                    "source_type": record.source_type,
                    "genre_top": record.genre_top,
                    "filepath": record.filepath,
                    "relative_path": record.relative_path,
                    "track_id": record.track_id,
                    "subset": record.subset,
                    "split": record.split,
                    "artist_id": record.artist_id,
                    "sample_length_sec": sample_length_sec,
                    "segment_index": segment_index,
                    "segment_start_sec": float(start_sec),
                    "segment_end_sec": float(end_sec),
                    "total_segments_from_audio": total_segments,
                    "audio_duration_s": record.audio_duration_s,
                    "file_ext": record.file_ext,
                    "reason_code": record.reason_code,
                    "final_split": None,
                    "selected_for_final_manifest": False,
                }
            )

        if (idx + 1) % 2000 == 0 or (idx + 1) == n_eligible:
            logging.info("[Samples] Expanded %d/%d audio files → %d segments so far", idx + 1, n_eligible, len(rows))

    logging.info("[Samples] Expansion complete: %d segments from %d files in %.1fs", len(rows), n_eligible, time.time() - t0)
    samples = pd.DataFrame(rows)
    if samples.empty:
        return samples

    samples = samples.sort_values(["genre_top", "source", "sample_id", "segment_index"], kind="stable").reset_index(drop=True)
    samples["track_id"] = pd.Series(samples["track_id"], dtype="Int64")
    samples["artist_id"] = pd.Series(samples["artist_id"], dtype="Int64")
    samples["total_segments_from_audio"] = pd.Series(samples["total_segments_from_audio"], dtype="Int64")
    return samples


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)


def summarize_skipped_audio_paths(all_datasets: pd.DataFrame) -> pd.DataFrame:
    """Return one row per skipped audio filepath with aggregated metadata."""
    if all_datasets.empty or "filepath" not in all_datasets.columns:
        return pd.DataFrame(columns=["filepath", "source", "genre_top", "skip_reason"])

    skipped = all_datasets.copy()
    if "reason_code" in skipped.columns:
        skipped["skip_reason"] = skipped["reason_code"].astype("string")
    else:
        skipped["skip_reason"] = pd.Series([None] * len(skipped), dtype="string")

    if "sampling_exclusion_reason" in skipped.columns:
        fallback_reason = skipped["sampling_exclusion_reason"].astype("string")
        skipped["skip_reason"] = skipped["skip_reason"].where(
            skipped["skip_reason"] != "OK",
            fallback_reason,
        )

    skipped = skipped[
        skipped["skip_reason"].notna() & (skipped["skip_reason"].astype(str) != "OK")
    ].copy()
    if skipped.empty:
        return pd.DataFrame(columns=["filepath", "source", "genre_top", "skip_reason"])

    skipped["filepath"] = skipped["filepath"].astype(str)
    skipped["source"] = skipped["source"].astype(str)
    skipped["genre_top"] = skipped["genre_top"].astype("string")

    def _join_unique(values: pd.Series) -> str:
        items = sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()})
        return ", ".join(items)

    return (
        skipped.sort_values(["filepath", "skip_reason", "genre_top"], kind="stable")
        .groupby("filepath", as_index=False)
        .agg(
            source=("source", _join_unique),
            genre_top=("genre_top", _join_unique),
            skip_reason=("skip_reason", _join_unique),
        )
    )


def log_skipped_audio_paths(all_datasets: pd.DataFrame) -> int:
    """Print the pathname of each audio file skipped from the usable dataset."""
    summarized = summarize_skipped_audio_paths(all_datasets)
    logging.info("Skipped or unusable audio files: %d", len(summarized))
    for row in summarized.itertuples(index=False):
        genre_text = row.genre_top or "unknown"
        source_text = row.source or "unknown"
        logging.info(
            "[Skipped] reason=%s | source=%s | genre=%s | path=%s",
            row.skip_reason,
            source_text,
            genre_text,
            row.filepath,
        )
    return int(len(summarized))


def log_summary(
    all_datasets: pd.DataFrame,
    all_samples: pd.DataFrame,
    final_samples: pd.DataFrame | None = None,
) -> None:
    logging.info("All-datasets rows: %d", len(all_datasets))
    if not all_datasets.empty:
        logging.info(
            "Eligible dataset rows: %d | Ineligible dataset rows: %d",
            int(all_datasets["sampling_eligible"].sum()),
            int((~all_datasets["sampling_eligible"]).sum()),
        )
        logging.info("Dataset rows by source:\n%s", all_datasets.groupby("source").size().to_string())
        logging.info("Dataset rows by genre:\n%s", all_datasets.groupby("genre_top").size().sort_values(ascending=False).to_string())

    logging.info("All-samples rows: %d", len(all_samples))
    if not all_samples.empty:
        logging.info("Sample rows by source:\n%s", all_samples.groupby("source").size().to_string())
        logging.info("Sample rows by genre:\n%s", all_samples.groupby("genre_top").size().sort_values(ascending=False).to_string())
    if final_samples is not None:
        logging.info("Final-samples rows: %d", len(final_samples))
        if not final_samples.empty:
            logging.info("Final sample rows by split:\n%s", final_samples.groupby("final_split").size().to_string())
            logging.info("Final sample rows by genre:\n%s", final_samples.groupby("genre_top").size().sort_values(ascending=False).to_string())

    skipped_count = log_skipped_audio_paths(all_datasets)
    logging.info("Skipped audio file count: %d", skipped_count)


# ─────────────────────────────────────────────────────────────────────────────
# Report + config snapshot
# ─────────────────────────────────────────────────────────────────────────────

def write_build_report(
    all_datasets: pd.DataFrame,
    all_samples: pd.DataFrame,
    final_samples: pd.DataFrame,
    split_summary: pd.DataFrame,
    path: Path,
    config: dict,
) -> None:
    """Write a human-readable build report."""
    sep = "=" * 62
    lines: list[str] = [
        sep,
        "build_all_datasets_and_samples report",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        sep, "",
    ]

    # Config summary
    lines += ["── Configuration ───────────────────────────────────────"]
    for key, val in config.items():
        lines.append(f"  {key}: {val}")

    # Dataset summary
    lines += ["", "── Dataset manifest ────────────────────────────────────"]
    lines.append(f"  Total rows              : {len(all_datasets):>7,}")
    if not all_datasets.empty:
        n_eligible = int(all_datasets["sampling_eligible"].sum())
        skipped_count = len(summarize_skipped_audio_paths(all_datasets))
        lines.append(f"  Eligible for sampling   : {n_eligible:>7,}")
        lines.append(f"  Ineligible              : {len(all_datasets) - n_eligible:>7,}")
        lines.append(f"  Skipped audio files     : {skipped_count:>7,}")

        lines += ["", "  Rows by source:"]
        for src, cnt in all_datasets.groupby("source").size().items():
            lines.append(f"    {src:<28} {cnt:>7,}")

        lines += ["", "  Rows by genre:"]
        for genre, cnt in all_datasets.groupby("genre_top").size().sort_values(ascending=False).items():
            lines.append(f"    {genre:<28} {cnt:>7,}")

        lines += ["", "  Reason code breakdown:"]
        for code, cnt in all_datasets["reason_code"].value_counts().items():
            lines.append(f"    {code:<28} {cnt:>7,}")

        if "split" in all_datasets.columns:
            lines += ["", "  Split distribution (FMA rows with split):"]
            split_rows = all_datasets[all_datasets["split"].notna()]
            for split_name, cnt in split_rows["split"].value_counts().items():
                lines.append(f"    {split_name:<14} {cnt:>7,}")

        # Artist leakage check (FMA rows only)
        fma_ok = all_datasets[
            (all_datasets["source_type"] == "fma")
            & (all_datasets["reason_code"] == "OK")
            & (all_datasets["artist_id"].notna())
        ]
        if not fma_ok.empty and "split" in fma_ok.columns:
            lines += ["", "  Artist leakage check (FMA OK rows):"]
            splits = {
                s: set(fma_ok[fma_ok["split"] == s]["artist_id"])
                for s in ("training", "validation", "test")
            }
            for a, b in [("training", "validation"), ("training", "test"), ("validation", "test")]:
                overlap = splits.get(a, set()) & splits.get(b, set())
                status = "OK \u2014 no overlap" if not overlap else f"!! {len(overlap)} shared artists"
                lines.append(f"    {a} \u2229 {b}: {len(overlap)} ({status})")

    # Samples summary
    lines += ["", "── Samples manifest ────────────────────────────────────"]
    lines.append(f"  Total sample segments   : {len(all_samples):>7,}")
    if not all_samples.empty:
        lines += ["", "  Segments by source:"]
        for src, cnt in all_samples.groupby("source").size().items():
            lines.append(f"    {src:<28} {cnt:>7,}")

        lines += ["", "  Segments by genre:"]
        for genre, cnt in all_samples.groupby("genre_top").size().sort_values(ascending=False).items():
            lines.append(f"    {genre:<28} {cnt:>7,}")

    lines += ["", "── Final samples manifest ──────────────────────────────"]
    lines.append(f"  Total final segments    : {len(final_samples):>7,}")
    if not final_samples.empty:
        lines += ["", "  Final segments by split:"]
        for split_name, cnt in final_samples.groupby("final_split").size().items():
            lines.append(f"    {split_name:<28} {cnt:>7,}")

        lines += ["", "  Final segments by genre:"]
        for genre, cnt in final_samples.groupby("genre_top").size().sort_values(ascending=False).items():
            lines.append(f"    {genre:<28} {cnt:>7,}")

        if not split_summary.empty:
            lines += ["", "  Final per-genre split counts:"]
            lines.append(f"    {'Genre':<24} {'train':>7} {'val':>7} {'test':>7} {'total':>7} {'target':>7}")
            for row in split_summary.itertuples(index=False):
                lines.append(
                    f"    {row.genre_top:<24} {int(row.training):>7,} {int(row.validation):>7,}"
                    f" {int(row.test):>7,} {int(row.selected_total):>7,} {int(row.target_total):>7,}"
                )

    lines += ["", sep]
    path.write_text("\n".join(lines) + "\n")
    logging.info("Wrote report: %s", path)


def write_config_snapshot(config: dict, path: Path) -> None:
    """Write a JSON config snapshot for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)
    logging.info("Wrote config snapshot: %s", path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    settings_path = Path(args.settings)
    settings = load_data_sampling_settings(settings_path)
    target_genres = settings["target_genres"]
    sample_length_sec = float(settings["sample_length_sec"])
    min_duration_delta = float(settings["min_duration_delta"])
    number_of_samples_expected_each_genre = int(settings["number_of_samples_expected_each_genre"])
    train_ratio_each_genre = float(settings["train_n_val_test_split_ratio_each_genre"])
    min_duration_s = (
        float(args.min_duration)
        if args.min_duration is not None
        else derive_default_min_duration(sample_length_sec, min_duration_delta)
    )

    medium_manifest_path = Path(args.medium_manifest)
    if str(medium_manifest_path) == str(DEFAULT_MEDIUM_MANIFEST):
        medium_manifest_path = DEFAULT_PROCESSED_DIR / f"metadata_manifest_{args.fma_subset}.parquet"
    fma_metadata_root = Path(args.fma_metadata_root)
    fma_audio_root = Path(args.fma_audio_root)
    additional_root = Path(args.additional_root)
    all_datasets_out = Path(args.all_datasets_out)
    all_samples_out = Path(args.all_samples_out)
    final_samples_out = Path(args.final_samples_out)

    config = {
        "target_genres": target_genres,
        "sample_length_sec": sample_length_sec,
        "min_duration_delta": min_duration_delta,
        "number_of_samples_expected_each_genre": number_of_samples_expected_each_genre,
        "train_n_val_test_split_ratio_each_genre": train_ratio_each_genre,
        "split_seed": args.split_seed,
        "fma_subset": args.fma_subset,
        "min_duration_s": min_duration_s,
        "min_duration_source": "cli" if args.min_duration is not None else "sample_length_sec_minus_delta",
        "force_rescan": args.force_rescan,
        "medium_manifest": str(medium_manifest_path),
        "fma_metadata_root": str(fma_metadata_root),
        "fma_audio_root": str(fma_audio_root),
        "additional_root": str(additional_root),
        "all_datasets_out": str(all_datasets_out),
        "all_samples_out": str(all_samples_out),
        "final_samples_out": str(final_samples_out),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    logging.info("Target genres: %s", target_genres)
    logging.info("Sample length: %.3fs", sample_length_sec)
    logging.info("Min-duration delta: %.3fs", min_duration_delta)
    logging.info(
        "Per-genre final target: %d | train ratio: %.3f | split seed: %d",
        number_of_samples_expected_each_genre,
        train_ratio_each_genre,
        args.split_seed,
    )
    logging.info("FMA subset: %s | min_duration: %.3fs | force_rescan: %s",
                 args.fma_subset, min_duration_s, args.force_rescan)
    logging.info("=" * 60)

    t_start = time.time()

    # Step 1: FMA candidates (smart: parquet if available, else from-scratch)
    logging.info("Step 1/5: Building FMA candidates ...")
    fma_df = build_fma_candidates(
        medium_manifest_path=medium_manifest_path,
        fma_metadata_root=fma_metadata_root,
        fma_audio_root=fma_audio_root,
        fma_subset=args.fma_subset,
        target_genres=target_genres,
        sample_length_sec=sample_length_sec,
        min_duration_s=min_duration_s,
        force_rescan=args.force_rescan,
    )
    logging.info("Step 1/5 done (%.1fs)\n", time.time() - t_start)

    # Step 2: Additional dataset candidates
    t1 = time.time()
    logging.info("Step 2/5: Collecting additional dataset candidates ...")
    additional_df = collect_additional_candidates(
        additional_root, target_genres, sample_length_sec, min_duration_s,
    )
    logging.info("Step 2/5 done (%.1fs)\n", time.time() - t1)

    # Step 3: Combine
    t2 = time.time()
    logging.info("Step 3/5: Combining dataset manifest ...")
    all_datasets = combine_dataset_manifest(fma_df, additional_df)
    logging.info("Step 3/5 done (%.1fs)\n", time.time() - t2)

    # Step 4: Build sample segments
    t3 = time.time()
    logging.info("Step 4/5: Building sample segments manifest ...")
    all_samples = build_samples_manifest(all_datasets, sample_length_sec)
    logging.info("Step 4/5 done (%.1fs)\n", time.time() - t3)

    t4 = time.time()
    logging.info("Step 5/5: Assigning final train/validation/test splits ...")
    all_samples, final_samples, split_summary = assign_final_splits(
        all_samples,
        number_of_samples_expected_each_genre,
        train_ratio_each_genre,
        args.split_seed,
    )
    logging.info("Step 5/5 done (%.1fs)\n", time.time() - t4)

    # Write outputs
    write_parquet(all_datasets_out, all_datasets)
    write_parquet(all_samples_out, all_samples)
    write_parquet(final_samples_out, final_samples)
    log_summary(all_datasets, all_samples, final_samples)

    # Write report + config snapshot alongside the datasets parquet
    report_path = all_datasets_out.with_suffix(".report.txt")
    config_path = all_datasets_out.with_suffix(".config.json")
    write_build_report(all_datasets, all_samples, final_samples, split_summary, report_path, config)
    write_config_snapshot(config, config_path)

    logging.info("=" * 60)
    logging.info("Wrote %s (%d rows)", all_datasets_out, len(all_datasets))
    logging.info("Wrote %s (%d rows)", all_samples_out, len(all_samples))
    logging.info("Wrote %s (%d rows)", final_samples_out, len(final_samples))
    logging.info("Total time: %.1fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())