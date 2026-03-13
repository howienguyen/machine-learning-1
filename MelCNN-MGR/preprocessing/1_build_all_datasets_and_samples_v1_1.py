#!/usr/bin/env python3
"""Build file-level and sample-level manifests across all supported datasets.

Outputs written to MelCNN-MGR/data/processed by default:
    manifest_fma_datasets.parquet         one row per discovered FMA audio file candidate
    manifest_additional_datasets.parquet  one row per discovered additional-dataset audio file candidate
    manifest_fma_all_samples.parquet      one row per fixed-length FMA sample segment
    manifest_additional_all_samples.parquet one row per fixed-length additional-dataset sample segment
    manifest_final_samples.parquet        selected sample segments with final split labels

Logical pipeline stages:
    Stage 1a: generate manifest_fma_datasets.parquet and/or
              manifest_additional_datasets.parquet from selected source families
    Stage 1b: generate manifest_fma_all_samples.parquet and/or
              manifest_additional_all_samples.parquet from the corresponding
              Stage 1a dataset manifests
    Stage 2:  generate manifest_final_samples.parquet by combining both Stage 1b
              sample manifests for downstream model-training consumers

The CLI can execute Stage 1a only, Stage 1b only, Stage 1a+1b, Stage 2 only,
or the full pipeline in one run.
The parquet outputs define a natural boundary between intermediate manifest
generation and final training-manifest selection.

Artifact and sample identifiers are stored as deterministic 128-bit hex hashes.
The source-specific natural identity strings are still derived from stable
metadata, but the parquet-visible `artifact_id` and `sample_id` values use
their hashed forms instead of long raw path-based strings.

During Stage 2, the additional-source sample manifest is deterministically
shuffled with `--split-seed` before grouped final split assignment. Segment
grouping still keeps all segments from the same source audio in the same split.

Example commands
----------------
Run Stage 1a for FMA only:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage1a \
        --stage1a-sources fma

Run Stage 1a for additional datasets only:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage1a \
        --stage1a-sources additional

Run Stage 1b for FMA only:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage1b \
        --stage1b-sources fma

Run Stage 1b for additional datasets only:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage1b \
        --stage1b-sources additional

Run Stage 1b for both sources:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage1b \
        --stage1b-sources both

Run Stage 2 from the existing per-source sample manifests:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage2

Run the full pipeline in one command:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode both \
        --stage1a-sources both \
        --stage1b-sources both

Selected output parquet files are overwritten on rerun.

Stage 1a and Stage 1b source-selection modes
--------------------------------------------
Both stages can build from:

    fma
    additional
    both

Each selected parquet is rebuilt from scratch and overwritten on rerun.

The script uses settings.data_sampling_settings in MelCNN-MGR/settings.json:
    target_genres
    sample_length_sec

If `sample_length_sec` is missing or invalid inside an otherwise valid
`data_sampling_settings` object, the script falls back to 15.0 seconds.
Unreadable settings files and invalid `target_genres` remain hard errors.

Sampling rule
-------------
For each eligible audio file, the number of segments is:

    floor(duration_s / sample_length_sec)

Files shorter than sample_length_sec remain in their source-specific dataset
manifest with sampling_eligible=False, but they do not produce rows in the
corresponding source-specific sample manifest.
"""

from __future__ import annotations

import ast
import argparse
import hashlib
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
DEFAULT_FMA_DATASETS_PATH = DEFAULT_PROCESSED_DIR / "manifest_fma_datasets.parquet"
DEFAULT_ADDITIONAL_DATASETS_PATH = DEFAULT_PROCESSED_DIR / "manifest_additional_datasets.parquet"
DEFAULT_FMA_ALL_SAMPLES_PATH = DEFAULT_PROCESSED_DIR / "manifest_fma_all_samples.parquet"
DEFAULT_ADDITIONAL_ALL_SAMPLES_PATH = DEFAULT_PROCESSED_DIR / "manifest_additional_all_samples.parquet"
DEFAULT_FINAL_SAMPLES_PATH = DEFAULT_PROCESSED_DIR / "manifest_final_samples.parquet"

_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".aif", ".aiff", ".au", ".m4a"}
_FFPROBE = shutil.which("ffprobe")
_SOUNDFILE_FIRST_EXTENSIONS = {".wav", ".flac", ".aif", ".aiff", ".au", ".ogg"}
_DATASET_MANIFEST_COLUMNS = [
    "source", "artifact_id", "source_track_id", "track_id",
    "genre_top", "filepath", "audio_exists", "filesize_bytes", "actual_duration_s", "duration_s",
    "reason_code", "sampling_eligible", "sampling_num_segments",
    "sampling_exclusion_reason", "manifest_origin",
]
_SAMPLE_MANIFEST_COLUMNS = [
    "sample_id", "artifact_id", "source", "genre_top", "filepath", "track_id", "sample_length_sec",
    "segment_index", "segment_start_sec", "segment_end_sec", "total_segments_from_audio",
    "duration_s", "actual_duration_s", "reason_code",
]

# ── FMA-specific constants (ported from build_manifest.py) ────────────────────
DEFAULT_FMA_METADATA_ROOT = _WORKSPACE / "FMA" / "fma_metadata"
DEFAULT_FMA_AUDIO_ROOT = _WORKSPACE / "FMA" / "fma_medium"
DEFAULT_FMA_SUBSET = "medium"
DEFAULT_MIN_DURATION_DELTA = 0.001
DEFAULT_SAMPLE_LENGTH_SEC = 15.0
DEFAULT_NUMBER_OF_SAMPLES_EXPECTED_EACH_GENRE = 1300
DEFAULT_ADDITIONAL_SAMPLES_CONTRIBUTION_RATIO_EACH_GENRE = 0.0
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
        description="Build per-source dataset manifests, per-source sample manifests, and the final samples manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["stage1a", "stage1b", "stage1", "stage2", "both"],
        help="Execution mode: Stage 1a only, Stage 1b only, Stage 1a+1b, Stage 2 only, or the full pipeline.",
    )
    parser.add_argument(
        "--stage1a-sources",
        default="both",
        choices=["fma", "additional", "both"],
        help="Source family selection for Stage 1a dataset-manifest rebuilds.",
    )
    parser.add_argument(
        "--stage1b-sources",
        default="both",
        choices=["fma", "additional", "both"],
        help="Source family selection for Stage 1b sample-manifest rebuilds.",
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
        "--fma-datasets-out",
        default=str(DEFAULT_FMA_DATASETS_PATH),
        help="Output path for manifest_fma_datasets.parquet.",
    )
    parser.add_argument(
        "--additional-datasets-out",
        default=str(DEFAULT_ADDITIONAL_DATASETS_PATH),
        help="Output path for manifest_additional_datasets.parquet.",
    )
    parser.add_argument(
        "--fma-all-samples-out",
        default=str(DEFAULT_FMA_ALL_SAMPLES_PATH),
        help="Output path for manifest_fma_all_samples.parquet.",
    )
    parser.add_argument(
        "--additional-all-samples-out",
        default=str(DEFAULT_ADDITIONAL_ALL_SAMPLES_PATH),
        help="Output path for manifest_additional_all_samples.parquet.",
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


def _hash_id_string(value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Cannot hash an empty id string.")
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _artifact_id_string(
    source: object,
    genre_top: object,
    filepath: object,
    track_id: object,
) -> str:
    source_value = "" if source is None else str(source).strip()
    if _is_fma_source(source_value):
        if pd.isna(track_id):
            raise ValueError("FMA artifact id generation requires a track_id.")
        return f"fma:{int(track_id)}"

    filepath_obj = Path(str(filepath))
    relative_path = _relative_to_workspace(filepath_obj)
    source_audio_id = relative_path or str(filepath_obj)
    return f"{source_value}:{genre_top}:{source_audio_id}"


def _artifact_id(
    source: object,
    genre_top: object,
    filepath: object,
    track_id: object,
) -> str:
    return _hash_id_string(_artifact_id_string(source, genre_top, filepath, track_id))


def _sample_id_string(artifact_id_string: str, segment_index: int) -> str:
    return f"{artifact_id_string}:seg{segment_index:04d}"


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

    sample_length_sec_raw = config.get("sample_length_sec")
    sample_length_fallback_used = False
    if isinstance(sample_length_sec_raw, (int, float)) and sample_length_sec_raw > 0:
        sample_length_sec = float(sample_length_sec_raw)
    else:
        sample_length_sec = DEFAULT_SAMPLE_LENGTH_SEC
        sample_length_fallback_used = True

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

    additional_samples_contribution_ratio_each_genre = config.get(
        "additional_samples_contribution_ratio_expected_each_genre",
        config.get(
            "additional_samples_rate_expected_each_genre",
            DEFAULT_ADDITIONAL_SAMPLES_CONTRIBUTION_RATIO_EACH_GENRE,
        ),
    )
    if (
        not isinstance(additional_samples_contribution_ratio_each_genre, (int, float))
        or additional_samples_contribution_ratio_each_genre < 0
        or additional_samples_contribution_ratio_each_genre > 1
    ):
        additional_samples_contribution_ratio_each_genre = (
            DEFAULT_ADDITIONAL_SAMPLES_CONTRIBUTION_RATIO_EACH_GENRE
        )

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
        "sample_length_fallback_used": sample_length_fallback_used,
        "min_duration_delta": float(min_duration_delta),
        "number_of_samples_expected_each_genre": int(number_of_samples_expected_each_genre),
        "additional_samples_contribution_ratio_expected_each_genre": float(
            additional_samples_contribution_ratio_each_genre
        ),
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


def _is_fma_source(source: object) -> bool:
    value = "" if source is None else str(source).strip().casefold()
    return value.startswith("fma")


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
    df["artifact_id"] = [
        _artifact_id(source_label, None, None, int(track_id))
        for track_id in df.index.astype(int)
    ]
    df["source_track_id"] = df.index.astype(str)
    df["track_id"] = df.index.astype("Int64")
    logging.info("[FMA-rescan] Probing actual audio durations ...")
    df["actual_duration_s"], df["duration_s"], df["duration_source"] = _probe_durations_for_frame(df, "[FMA-rescan]")
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
        "source", "artifact_id", "source_track_id", "track_id", "genre_top",
        "filepath", "audio_exists", "filesize_bytes", "actual_duration_s", "duration_s",
        "reason_code", "sampling_eligible",
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


def _is_additional_source(source: object) -> bool:
    value = "" if source is None else str(source).strip().casefold()
    return not value.startswith("fma")


def _allocate_proportional_counts(total_count: int, weights: dict[str, int]) -> dict[str, int]:
    allocations = {split_name: 0 for split_name in _FINAL_SPLIT_ORDER}
    total_weight = sum(max(int(weights.get(split_name, 0)), 0) for split_name in _FINAL_SPLIT_ORDER)
    if total_count <= 0 or total_weight <= 0:
        return allocations

    remainders: list[tuple[float, int, str]] = []
    assigned = 0
    for split_index, split_name in enumerate(_FINAL_SPLIT_ORDER):
        raw = total_count * max(int(weights.get(split_name, 0)), 0) / total_weight
        base = int(math.floor(raw))
        allocations[split_name] = base
        assigned += base
        remainders.append((raw - base, -split_index, split_name))

    for _, _, split_name in sorted(remainders, reverse=True)[: max(0, total_count - assigned)]:
        allocations[split_name] += 1

    return allocations


def _compute_additional_total_target(
    total_target: int,
    ratio: float,
    additional_available: int,
    primary_available: int,
) -> int:
    if total_target <= 0 or additional_available <= 0:
        return 0

    ideal_target = int(round(total_target * ratio))
    minimum_needed = max(0, total_target - max(primary_available, 0))
    maximum_possible = min(total_target, max(additional_available, 0))
    return min(max(ideal_target, minimum_needed), maximum_possible)


def _select_group_assignments(
    grouped: pd.DataFrame,
    split_targets: dict[str, int],
) -> tuple[dict[str, str], dict[str, int]]:
    assignments: dict[str, str] = {}
    selected_counts = {split_name: 0 for split_name in _FINAL_SPLIT_ORDER}
    remaining = {split_name: int(split_targets.get(split_name, 0)) for split_name in _FINAL_SPLIT_ORDER}

    if grouped.empty or not any(value > 0 for value in remaining.values()):
        return assignments, selected_counts

    for group in grouped.itertuples(index=False):
        candidate_splits = [split_name for split_name in _FINAL_SPLIT_ORDER if remaining[split_name] > 0]
        if not candidate_splits:
            break

        sample_count = int(group.sample_count)
        split_choice = min(
            candidate_splits,
            key=lambda split_name: (
                max(sample_count - remaining[split_name], 0),
                abs(remaining[split_name] - sample_count),
                -remaining[split_name],
                _FINAL_SPLIT_ORDER.index(split_name),
            ),
        )

        assignments[str(group.group_sample_id)] = split_choice
        selected_counts[split_choice] += sample_count
        remaining[split_choice] -= sample_count

    return assignments, selected_counts


def _segment_group_id(sample_id: object) -> str:
    value = "" if sample_id is None else str(sample_id).strip()
    if not value or value.lower() == "nan":
        raise ValueError("Encountered missing sample_id while assigning final splits.")
    return _SEGMENT_SUFFIX_RE.sub("", value)


def assign_final_splits(
    all_samples: pd.DataFrame,
    number_of_samples_expected_each_genre: int,
    additional_samples_contribution_ratio_each_genre: float,
    train_ratio_each_genre: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign final split labels by source sample identity and build the final samples manifest.

    All segments from the same source sample stay in the same split to avoid segment
    leakage across training / validation / test.
    """
    samples = all_samples.copy()
    if samples.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "genre_top", "training", "validation", "test", "selected_total", "target_total",
                "additional_target_total", "additional_selected_total", "additional_selected_ratio_pct",
                "training_additional_target", "validation_additional_target", "test_additional_target",
                "training_additional_selected", "validation_additional_selected", "test_additional_selected",
            ]
        )
        return samples, samples.iloc[0:0].copy(), empty_summary

    working_samples = samples.copy()
    working_samples["final_split"] = pd.Series([None] * len(working_samples), dtype="object")
    working_samples["selected_for_final_manifest"] = False
    working_samples["group_sample_id"] = working_samples["sample_id"].map(_segment_group_id)
    working_samples["is_additional_source"] = working_samples["source"].map(_is_additional_source)
    summary_rows: list[dict[str, object]] = []

    for genre_index, genre in enumerate(sorted(working_samples["genre_top"].dropna().unique())):
        genre_mask = working_samples["genre_top"] == genre
        genre_df = working_samples.loc[genre_mask].copy()
        target_total = min(number_of_samples_expected_each_genre, len(genre_df))
        targets = _compute_final_split_targets(target_total, train_ratio_each_genre)

        grouped = (
            genre_df.groupby("group_sample_id", dropna=False, sort=False)
            .agg(
                sample_count=("sample_id", "size"),
                is_additional_source=("is_additional_source", "first"),
            )
            .reset_index()
        )

        additional_grouped = (
            grouped[grouped["is_additional_source"] == True]
            .reset_index(drop=True)
        )
        primary_grouped = (
            grouped[grouped["is_additional_source"] == False]
            .sample(frac=1.0, random_state=seed + 10_000 + genre_index)
            .reset_index(drop=True)
        )

        additional_available = int(additional_grouped["sample_count"].sum())
        primary_available = int(primary_grouped["sample_count"].sum())
        additional_target_total = _compute_additional_total_target(
            target_total,
            additional_samples_contribution_ratio_each_genre,
            additional_available,
            primary_available,
        )
        additional_targets = _allocate_proportional_counts(additional_target_total, targets)
        primary_targets = {
            split_name: max(targets[split_name] - additional_targets[split_name], 0)
            for split_name in _FINAL_SPLIT_ORDER
        }

        additional_assignments, additional_selected_counts = _select_group_assignments(
            additional_grouped,
            additional_targets,
        )
        primary_assignments, primary_selected_counts = _select_group_assignments(
            primary_grouped,
            primary_targets,
        )

        selected_assignments = {**additional_assignments, **primary_assignments}
        selected_counts = {
            split_name: additional_selected_counts[split_name] + primary_selected_counts[split_name]
            for split_name in _FINAL_SPLIT_ORDER
        }

        fallback_remaining = {
            split_name: max(targets[split_name] - selected_counts[split_name], 0)
            for split_name in _FINAL_SPLIT_ORDER
        }
        fallback_grouped = (
            grouped[~grouped["group_sample_id"].astype(str).isin(selected_assignments.keys())]
            .sample(frac=1.0, random_state=seed + 20_000 + genre_index)
            .reset_index(drop=True)
        )
        fallback_assignments, fallback_selected_counts = _select_group_assignments(
            fallback_grouped,
            fallback_remaining,
        )
        selected_assignments.update(fallback_assignments)
        selected_counts = {
            split_name: selected_counts[split_name] + fallback_selected_counts[split_name]
            for split_name in _FINAL_SPLIT_ORDER
        }

        for group_sample_id, split_choice in selected_assignments.items():
            group_mask = genre_mask & (working_samples["group_sample_id"] == group_sample_id)
            working_samples.loc[group_mask, "final_split"] = split_choice
            working_samples.loc[group_mask, "selected_for_final_manifest"] = True

        selected_grouped = grouped[grouped["group_sample_id"].astype(str).isin(selected_assignments.keys())].copy()
        selected_grouped["assigned_split"] = selected_grouped["group_sample_id"].astype(str).map(selected_assignments)
        selected_total = int(selected_grouped["sample_count"].sum())
        additional_selected_total = int(
            selected_grouped.loc[selected_grouped["is_additional_source"] == True, "sample_count"].sum()
        )
        additional_selected_by_split = (
            selected_grouped[selected_grouped["is_additional_source"] == True]
            .groupby("assigned_split")["sample_count"]
            .sum()
            .to_dict()
        )

        summary_rows.append(
            {
                "genre_top": genre,
                "training": selected_counts["training"],
                "validation": selected_counts["validation"],
                "test": selected_counts["test"],
                "selected_total": selected_total,
                "target_total": target_total,
                "additional_target_total": additional_target_total,
                "additional_selected_total": additional_selected_total,
                "additional_selected_ratio_pct": round(
                    (100.0 * additional_selected_total / selected_total) if selected_total else 0.0,
                    2,
                ),
                "training_additional_target": additional_targets["training"],
                "validation_additional_target": additional_targets["validation"],
                "test_additional_target": additional_targets["test"],
                "training_additional_selected": int(additional_selected_by_split.get("training", 0)),
                "validation_additional_selected": int(additional_selected_by_split.get("validation", 0)),
                "test_additional_selected": int(additional_selected_by_split.get("test", 0)),
            }
        )

    final_samples = working_samples[working_samples["selected_for_final_manifest"] == True].copy()
    final_samples = final_samples.sort_values(
        ["genre_top", "final_split", "group_sample_id", "segment_index"],
        kind="stable",
    ).reset_index(drop=True)
    final_columns = [
        "sample_id", "source", "genre_top", "filepath", "track_id",
        "sample_length_sec", "segment_index", "segment_start_sec",
        "segment_end_sec", "total_segments_from_audio", "duration_s", "actual_duration_s", "reason_code",
        "final_split",
    ]
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
    df["artifact_id"] = [
        _artifact_id(source_label, None, None, int(track_id))
        for track_id in df.index.astype(int)
    ]
    df["source_track_id"] = df.index.astype(str)
    df["track_id"] = df.index.astype("Int64")
    logging.info("[FMA-medium] Probing actual audio durations ...")
    df["actual_duration_s"], df["duration_s"], df["duration_source"] = _probe_durations_for_frame(df, "[FMA-medium]")
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
        "source", "artifact_id", "source_track_id", "track_id",
        "genre_top", "filepath", "audio_exists", "filesize_bytes", "actual_duration_s", "duration_s",
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
                actual_duration_s, duration_source = probe_audio_duration(audio_path)
                duration_s = _normalize_audio_duration(actual_duration_s)
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
                    artifact_id = _artifact_id(
                        source=source_dir.name,
                        genre_top=genre,
                        filepath=audio_path,
                        track_id=None,
                    )
                    rows.append(
                        {
                            "source": source_dir.name,
                            "artifact_id": artifact_id,
                            "source_track_id": None,
                            "track_id": None,
                            "genre_top": genre,
                            "filepath": str(audio_path),
                            "audio_exists": True,
                            "filesize_bytes": _safe_file_size(audio_path),
                            "actual_duration_s": actual_duration_s,
                            "duration_s": duration_s,
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
    df["filesize_bytes"] = pd.Series(df["filesize_bytes"], dtype="Int64")
    df["sampling_num_segments"] = pd.Series(df["sampling_num_segments"], dtype="Int64")
    return df


def combine_dataset_manifest(*frames: pd.DataFrame) -> pd.DataFrame:
    logging.info("[Combine] Merging %d frame(s) ...", sum(1 for f in frames if f is not None and not f.empty))
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    columns = list(_DATASET_MANIFEST_COLUMNS)
    if not non_empty:
        return pd.DataFrame(columns=columns)

    combined = pd.concat(non_empty, axis=0, ignore_index=True)
    before_dedup = len(combined)
    combined = combined[columns].drop_duplicates(subset=["artifact_id"], keep="first")
    logging.info("[Combine] %d rows after concat, %d after dedup", before_dedup, len(combined))
    combined = combined.sort_values(["genre_top", "source", "artifact_id"], kind="stable").reset_index(drop=True)
    combined["track_id"] = pd.Series(combined["track_id"], dtype="Int64")
    combined["filesize_bytes"] = pd.Series(combined["filesize_bytes"], dtype="Int64")
    combined["sampling_num_segments"] = pd.Series(combined["sampling_num_segments"], dtype="Int64")
    return combined


def load_dataset_manifest(path: Path, description: str) -> pd.DataFrame:
    """Load a dataset manifest used as input for Stage 1b."""
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 1b requires an existing {description}: {path}"
        )

    datasets = pd.read_parquet(path)
    missing = sorted(set(_DATASET_MANIFEST_COLUMNS).difference(datasets.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {description} {path}: {', '.join(missing)}"
        )

    datasets = datasets.copy()
    datasets["track_id"] = pd.Series(datasets["track_id"], dtype="Int64")
    datasets["filesize_bytes"] = pd.Series(datasets["filesize_bytes"], dtype="Int64")
    datasets["sampling_num_segments"] = pd.Series(datasets["sampling_num_segments"], dtype="Int64")
    return datasets


def build_stage1a_dataset_manifests(
    stage1a_sources: str,
    medium_manifest_path: Path,
    fma_metadata_root: Path,
    fma_audio_root: Path,
    fma_subset: str,
    additional_root: Path,
    target_genres: list[str],
    sample_length_sec: float,
    min_duration_s: float,
    force_rescan: bool,
    fma_datasets_out: Path,
    additional_datasets_out: Path,
) -> dict[str, pd.DataFrame]:
    """Build selected Stage 1a dataset manifests and overwrite their outputs."""
    manifests: dict[str, pd.DataFrame] = {}

    if stage1a_sources in {"fma", "both"}:
        logging.info("Stage 1a/FMA: Building FMA candidates ...")
        fma_df = build_fma_candidates(
            medium_manifest_path=medium_manifest_path,
            fma_metadata_root=fma_metadata_root,
            fma_audio_root=fma_audio_root,
            fma_subset=fma_subset,
            target_genres=target_genres,
            sample_length_sec=sample_length_sec,
            min_duration_s=min_duration_s,
            force_rescan=force_rescan,
        )
        write_parquet(fma_datasets_out, fma_df)
        logging.info("Stage 1a/FMA: Wrote %s (%d rows)", fma_datasets_out, len(fma_df))
        manifests["fma"] = fma_df

    if stage1a_sources in {"additional", "both"}:
        logging.info("Stage 1a/Additional: Collecting additional dataset candidates ...")
        additional_df = collect_additional_candidates(
            additional_root,
            target_genres,
            sample_length_sec,
            min_duration_s,
        )
        write_parquet(additional_datasets_out, additional_df)
        logging.info("Stage 1a/Additional: Wrote %s (%d rows)", additional_datasets_out, len(additional_df))
        manifests["additional"] = additional_df

    return manifests


def build_samples_manifest(all_datasets: pd.DataFrame, sample_length_sec: float) -> pd.DataFrame:
    logging.info("[Samples] Building sample segments (sample_length=%.1fs) ...", sample_length_sec)
    t0 = time.time()
    if all_datasets.empty:
        return pd.DataFrame(columns=list(_SAMPLE_MANIFEST_COLUMNS))

    rows: list[dict[str, object]] = []
    eligible = all_datasets[
        (all_datasets["reason_code"] == "OK")
        & (all_datasets["duration_s"].notna())
    ]
    n_eligible = len(eligible)
    logging.info("[Samples] %d eligible audio files to expand", n_eligible)
    for idx, record in enumerate(eligible.itertuples(index=False)):
        total_segments = _compute_segment_count(float(record.duration_s), sample_length_sec)
        if total_segments <= 0:
            continue
        for segment_index in range(total_segments):
            start_sec = segment_index * sample_length_sec
            end_sec = start_sec + sample_length_sec
            rows.append(
                {
                    "sample_id": _sample_id_string(str(record.artifact_id), segment_index),
                    "artifact_id": record.artifact_id,
                    "source": record.source,
                    "genre_top": record.genre_top,
                    "filepath": record.filepath,
                    "track_id": record.track_id,
                    "sample_length_sec": sample_length_sec,
                    "segment_index": segment_index,
                    "segment_start_sec": float(start_sec),
                    "segment_end_sec": float(end_sec),
                    "total_segments_from_audio": total_segments,
                    "duration_s": record.duration_s,
                    "actual_duration_s": record.actual_duration_s,
                    "reason_code": record.reason_code,
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
    samples["total_segments_from_audio"] = pd.Series(samples["total_segments_from_audio"], dtype="Int64")
    return samples


def combine_sample_manifests(*frames: pd.DataFrame, sort_output: bool = True) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame(columns=list(_SAMPLE_MANIFEST_COLUMNS))

    combined = pd.concat(non_empty, axis=0, ignore_index=True)
    combined = combined[list(_SAMPLE_MANIFEST_COLUMNS)].drop_duplicates(subset=["sample_id"], keep="first")
    if sort_output:
        combined = combined.sort_values(["genre_top", "source", "sample_id", "segment_index"], kind="stable")
    combined = combined.reset_index(drop=True)
    combined["track_id"] = pd.Series(combined["track_id"], dtype="Int64")
    combined["total_segments_from_audio"] = pd.Series(combined["total_segments_from_audio"], dtype="Int64")
    return combined


def shuffle_sample_manifest(frame: pd.DataFrame, random_state: int, description: str) -> pd.DataFrame:
    """Return a deterministically shuffled sample manifest for Stage 2 candidate ordering."""
    if frame is None or frame.empty:
        return frame

    logging.info("Stage 2: Shuffling %s with split seed %d", description, random_state)
    return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)


def load_samples_manifest(path: Path, description: str) -> pd.DataFrame:
    """Load a sample manifest used as input for Stage 2."""
    if not path.exists():
        raise FileNotFoundError(
            f"Stage 2 requires an existing {description}: {path}"
        )

    samples = pd.read_parquet(path)
    required_columns = {
        "sample_id", "source", "genre_top", "filepath", "track_id",
        "sample_length_sec", "segment_index", "segment_start_sec", "segment_end_sec",
        "total_segments_from_audio", "duration_s", "actual_duration_s", "reason_code",
    }
    missing = sorted(required_columns.difference(samples.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {description} {path}: {', '.join(missing)}"
        )

    samples = samples.copy()
    samples["track_id"] = pd.Series(samples["track_id"], dtype="Int64")
    samples["total_segments_from_audio"] = pd.Series(samples["total_segments_from_audio"], dtype="Int64")
    return samples


def build_stage1b_sample_manifests(
    stage1b_sources: str,
    sample_length_sec: float,
    fma_datasets_path: Path,
    additional_datasets_path: Path,
    fma_all_samples_out: Path,
    additional_all_samples_out: Path,
) -> dict[str, pd.DataFrame]:
    """Build selected Stage 1b sample manifests and overwrite their outputs."""
    manifests: dict[str, pd.DataFrame] = {}

    if stage1b_sources in {"fma", "both"}:
        logging.info("Stage 1b/FMA: Loading dataset manifest from %s", fma_datasets_path)
        fma_datasets = load_dataset_manifest(fma_datasets_path, "Stage 1a FMA dataset manifest")
        fma_samples = build_samples_manifest(fma_datasets, sample_length_sec)
        write_parquet(fma_all_samples_out, fma_samples)
        logging.info("Stage 1b/FMA: Wrote %s (%d rows)", fma_all_samples_out, len(fma_samples))
        manifests["fma"] = fma_samples

    if stage1b_sources in {"additional", "both"}:
        logging.info("Stage 1b/Additional: Loading dataset manifest from %s", additional_datasets_path)
        additional_datasets = load_dataset_manifest(
            additional_datasets_path,
            "Stage 1a additional dataset manifest",
        )
        additional_samples = build_samples_manifest(additional_datasets, sample_length_sec)
        write_parquet(additional_all_samples_out, additional_samples)
        logging.info("Stage 1b/Additional: Wrote %s (%d rows)", additional_all_samples_out, len(additional_samples))
        manifests["additional"] = additional_samples

    return manifests


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
    all_datasets: pd.DataFrame | None = None,
    all_samples: pd.DataFrame | None = None,
    final_samples: pd.DataFrame | None = None,
) -> None:
    if all_datasets is not None:
        logging.info("All-datasets rows: %d", len(all_datasets))
    if all_datasets is not None and not all_datasets.empty:
        logging.info(
            "Eligible dataset rows: %d | Ineligible dataset rows: %d",
            int(all_datasets["sampling_eligible"].sum()),
            int((~all_datasets["sampling_eligible"]).sum()),
        )
        logging.info("Dataset rows by source:\n%s", all_datasets.groupby("source").size().to_string())
        logging.info("Dataset rows by genre:\n%s", all_datasets.groupby("genre_top").size().sort_values(ascending=False).to_string())

    if all_samples is not None:
        logging.info("All-samples rows: %d", len(all_samples))
    if all_samples is not None and not all_samples.empty:
        logging.info("Sample rows by source:\n%s", all_samples.groupby("source").size().to_string())
        logging.info("Sample rows by genre:\n%s", all_samples.groupby("genre_top").size().sort_values(ascending=False).to_string())
    if final_samples is not None:
        logging.info("Final-samples rows: %d", len(final_samples))
        if not final_samples.empty:
            logging.info("Final sample rows by split:\n%s", final_samples.groupby("final_split").size().to_string())
            logging.info("Final sample rows by source:\n%s", final_samples.groupby("source").size().to_string())
            logging.info("Final sample rows by genre:\n%s", final_samples.groupby("genre_top").size().sort_values(ascending=False).to_string())

    if all_datasets is not None:
        skipped_count = log_skipped_audio_paths(all_datasets)
        logging.info("Skipped audio file count: %d", skipped_count)


# ─────────────────────────────────────────────────────────────────────────────
# Report + config snapshot
# ─────────────────────────────────────────────────────────────────────────────

def write_build_report(
    all_datasets: pd.DataFrame | None,
    all_samples: pd.DataFrame | None,
    final_samples: pd.DataFrame | None,
    split_summary: pd.DataFrame | None,
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

    if all_datasets is not None:
        lines += ["", "── Dataset manifest ────────────────────────────────────"]
        lines.append(f"  Total rows              : {len(all_datasets):>7,}")
    if all_datasets is not None and not all_datasets.empty:
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

    if all_samples is not None:
        lines += ["", "── Samples manifest ────────────────────────────────────"]
        lines.append(f"  Total sample segments   : {len(all_samples):>7,}")
    if all_samples is not None and not all_samples.empty:
        lines += ["", "  Segments by source:"]
        for src, cnt in all_samples.groupby("source").size().items():
            lines.append(f"    {src:<28} {cnt:>7,}")

        lines += ["", "  Segments by genre:"]
        for genre, cnt in all_samples.groupby("genre_top").size().sort_values(ascending=False).items():
            lines.append(f"    {genre:<28} {cnt:>7,}")

    if final_samples is not None:
        lines += ["", "── Final samples manifest ──────────────────────────────"]
        lines.append(f"  Total final segments    : {len(final_samples):>7,}")
    if final_samples is not None and not final_samples.empty:
        lines += ["", "  Final segments by split:"]
        for split_name, cnt in final_samples.groupby("final_split").size().items():
            lines.append(f"    {split_name:<28} {cnt:>7,}")

        lines += ["", "  Final segments by genre:"]
        for genre, cnt in final_samples.groupby("genre_top").size().sort_values(ascending=False).items():
            lines.append(f"    {genre:<28} {cnt:>7,}")

        if split_summary is not None and not split_summary.empty:
            lines += ["", "  Final per-genre split counts:"]
            lines.append(f"    {'Genre':<24} {'train':>7} {'val':>7} {'test':>7} {'total':>7} {'target':>7}")
            for row in split_summary.itertuples(index=False):
                lines.append(
                    f"    {row.genre_top:<24} {int(row.training):>7,} {int(row.validation):>7,}"
                    f" {int(row.test):>7,} {int(row.selected_total):>7,} {int(row.target_total):>7,}"
                )

            lines += ["", "  Final per-genre additional-source contribution:"]
            lines.append(
                f"    {'Genre':<24} {'add_tgt':>7} {'add_sel':>7} {'add_%':>7} {'tr_add':>7} {'va_add':>7} {'te_add':>7}"
            )
            for row in split_summary.itertuples(index=False):
                lines.append(
                    f"    {row.genre_top:<24} {int(row.additional_target_total):>7,}"
                    f" {int(row.additional_selected_total):>7,} {float(row.additional_selected_ratio_pct):>6.2f}%"
                    f" {int(row.training_additional_selected):>7,} {int(row.validation_additional_selected):>7,}"
                    f" {int(row.test_additional_selected):>7,}"
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
    sample_length_fallback_used = bool(settings.get("sample_length_fallback_used", False))
    min_duration_delta = float(settings["min_duration_delta"])
    number_of_samples_expected_each_genre = int(settings["number_of_samples_expected_each_genre"])
    additional_samples_contribution_ratio_each_genre = float(
        settings["additional_samples_contribution_ratio_expected_each_genre"]
    )
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
    fma_datasets_out = Path(args.fma_datasets_out)
    additional_datasets_out = Path(args.additional_datasets_out)
    fma_all_samples_out = Path(args.fma_all_samples_out)
    additional_all_samples_out = Path(args.additional_all_samples_out)
    final_samples_out = Path(args.final_samples_out)

    config = {
        "mode": args.mode,
        "stage1a_sources": args.stage1a_sources,
        "stage1b_sources": args.stage1b_sources,
        "target_genres": target_genres,
        "sample_length_sec": sample_length_sec,
        "min_duration_delta": min_duration_delta,
        "number_of_samples_expected_each_genre": number_of_samples_expected_each_genre,
        "additional_samples_contribution_ratio_expected_each_genre": additional_samples_contribution_ratio_each_genre,
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
        "fma_datasets_out": str(fma_datasets_out),
        "additional_datasets_out": str(additional_datasets_out),
        "fma_all_samples_out": str(fma_all_samples_out),
        "additional_all_samples_out": str(additional_all_samples_out),
        "final_samples_out": str(final_samples_out),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    logging.info("Target genres: %s", target_genres)
    logging.info("Sample length: %.3fs", sample_length_sec)
    if sample_length_fallback_used:
        logging.warning(
            "settings.data_sampling_settings.sample_length_sec was missing or invalid in %s; using default %.3fs",
            settings_path,
            DEFAULT_SAMPLE_LENGTH_SEC,
        )
    logging.info("Min-duration delta: %.3fs", min_duration_delta)
    logging.info(
        "Per-genre final target: %d | additional contribution ratio: %.3f | train ratio: %.3f | split seed: %d",
        number_of_samples_expected_each_genre,
        additional_samples_contribution_ratio_each_genre,
        train_ratio_each_genre,
        args.split_seed,
    )
    logging.info("FMA subset: %s | min_duration: %.3fs | force_rescan: %s",
                 args.fma_subset, min_duration_s, args.force_rescan)
    logging.info("Stage 1a sources: %s", args.stage1a_sources)
    logging.info("Stage 1b sources: %s", args.stage1b_sources)
    logging.info("Execution mode: %s", args.mode)
    logging.info("=" * 60)

    t_start = time.time()

    stage1a_outputs: dict[str, pd.DataFrame] = {}
    stage1b_outputs: dict[str, pd.DataFrame] = {}
    all_datasets: pd.DataFrame | None = None
    all_samples: pd.DataFrame | None = None
    final_samples: pd.DataFrame | None = None
    split_summary: pd.DataFrame | None = None

    if args.mode in {"stage1a", "stage1", "both"}:
        t_stage1a = time.time()
        logging.info("Step 1a: Building dataset manifests ...")
        stage1a_outputs = build_stage1a_dataset_manifests(
            stage1a_sources=args.stage1a_sources,
            medium_manifest_path=medium_manifest_path,
            fma_metadata_root=fma_metadata_root,
            fma_audio_root=fma_audio_root,
            fma_subset=args.fma_subset,
            additional_root=additional_root,
            target_genres=target_genres,
            sample_length_sec=sample_length_sec,
            min_duration_s=min_duration_s,
            force_rescan=args.force_rescan,
            fma_datasets_out=fma_datasets_out,
            additional_datasets_out=additional_datasets_out,
        )
        all_datasets = combine_dataset_manifest(*stage1a_outputs.values())
        logging.info("Step 1a done (%.1fs)\n", time.time() - t_stage1a)

    if args.mode == "stage1b":
        logging.info("Stage 1b: using source selection %s", args.stage1b_sources)

    if args.mode in {"stage1b", "stage1", "both"}:
        t_stage1b = time.time()
        logging.info("Step 1b: Building sample manifests ...")
        stage1b_outputs = build_stage1b_sample_manifests(
            stage1b_sources=args.stage1b_sources,
            sample_length_sec=sample_length_sec,
            fma_datasets_path=fma_datasets_out,
            additional_datasets_path=additional_datasets_out,
            fma_all_samples_out=fma_all_samples_out,
            additional_all_samples_out=additional_all_samples_out,
        )
        all_samples = combine_sample_manifests(*stage1b_outputs.values())
        logging.info("Step 1b done (%.1fs)\n", time.time() - t_stage1b)

    if args.mode == "stage2":
        logging.info(
            "Stage 2 input: loading FMA and additional sample manifests from %s and %s",
            fma_all_samples_out,
            additional_all_samples_out,
        )

    if args.mode in {"stage2", "both"}:
        stage2_sample_frames: list[pd.DataFrame] = []
        if "fma" in stage1b_outputs:
            stage2_sample_frames.append(stage1b_outputs["fma"])
        else:
            stage2_sample_frames.append(
                load_samples_manifest(fma_all_samples_out, "Stage 1b FMA sample manifest")
            )

        if "additional" in stage1b_outputs:
            stage2_sample_frames.append(
                shuffle_sample_manifest(
                    stage1b_outputs["additional"],
                    args.split_seed + 30_000,
                    "in-memory Stage 1b additional sample manifest",
                )
            )
        else:
            stage2_sample_frames.append(
                shuffle_sample_manifest(
                    load_samples_manifest(
                        additional_all_samples_out,
                        "Stage 1b additional sample manifest",
                    ),
                    args.split_seed + 30_000,
                    "loaded Stage 1b additional sample manifest",
                )
            )

        all_samples = combine_sample_manifests(*stage2_sample_frames, sort_output=False)
        t_stage2 = time.time()
        logging.info("Step 2: Assigning final train/validation/test splits ...")
        all_samples, final_samples, split_summary = assign_final_splits(
            all_samples,
            number_of_samples_expected_each_genre,
            additional_samples_contribution_ratio_each_genre,
            train_ratio_each_genre,
            args.split_seed,
        )
        logging.info("Step 2 done (%.1fs)\n", time.time() - t_stage2)
        write_parquet(final_samples_out, final_samples)

    log_summary(all_datasets, all_samples, final_samples)

    if final_samples is not None:
        report_base = final_samples_out
    elif all_samples is not None:
        report_base = additional_all_samples_out if args.stage1b_sources == "additional" else fma_all_samples_out
    else:
        report_base = additional_datasets_out if args.stage1a_sources == "additional" else fma_datasets_out
    report_path = report_base.with_suffix(".report.txt")
    config_path = report_base.with_suffix(".config.json")
    write_build_report(all_datasets, all_samples, final_samples, split_summary, report_path, config)
    write_config_snapshot(config, config_path)

    logging.info("=" * 60)
    if "fma" in stage1a_outputs:
        logging.info("Wrote %s (%d rows)", fma_datasets_out, len(stage1a_outputs["fma"]))
    if "additional" in stage1a_outputs:
        logging.info("Wrote %s (%d rows)", additional_datasets_out, len(stage1a_outputs["additional"]))
    if "fma" in stage1b_outputs:
        logging.info("Wrote %s (%d rows)", fma_all_samples_out, len(stage1b_outputs["fma"]))
    if "additional" in stage1b_outputs:
        logging.info("Wrote %s (%d rows)", additional_all_samples_out, len(stage1b_outputs["additional"]))
    if final_samples is not None:
        logging.info("Wrote %s (%d rows)", final_samples_out, len(final_samples))
    logging.info("Total time: %.1fs", time.time() - t_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())