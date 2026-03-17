#!/usr/bin/env python3
"""Build split-grouped log-mel .npy files from the final sample manifest.

This script turns sample-level manifest rows into deterministic log-mel feature
files grouped by split directories:

- train/
- val/
- test/

It also writes parquet indexes that training scripts can consume directly,
instead of re-decoding audio and rebuilding log-mels inside each model script.

Upstream manifests provide deterministic hashed sample identifiers. Segment-level
sample ids keep the `:segNNNN` suffix so grouped audio identity remains
recoverable by stripping the suffix when needed.

Each manifest row is decoded using its segment start time and the target
`sample_length_sec`. If the decoded waveform is shorter than the target length
because the source audio ends early, the builder pads the waveform with zeros
(silence) so every sample still reaches the exact target duration. If the
decoded waveform is longer than expected, it is truncated to the target length.
The final log-mel array is also written to a fixed `(n_mels, n_frames)` shape,
with any missing tail frames zero-filled.

Defaults mirror the active baseline reference in baseline_logmel_cnn_v21.py:
- sample_rate = 22050
- n_mels = 192
- n_fft = 512
- hop_length = 256

The default output-root suffix follows
`settings.data_sampling_settings.sample_length_sec` when available and falls
back to 15 seconds if the settings file cannot be read. The actual extraction
length remains manifest-driven unless `--sample-length-sec` is provided.

Manifest rows are filtered by `settings.data_sampling_settings.target_genres`
before log-mel generation so only currently configured genres are written.

Preferred upstream/downstream run profile:

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage1 \
        --stage1a-sources fma \
        --stage1b-sources fma

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage1 \
        --stage1a-sources additional \
        --stage1b-sources additional

    python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py \
        --mode stage2

    python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py

    python MelCNN-MGR/model_training/logmel_cnn_v2_2.py
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures as futures
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import warnings
import wave
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MELCNN_DIR = SCRIPT_DIR.parent
WORKSPACE = MELCNN_DIR.parent

DEFAULT_MANIFEST_PATH = MELCNN_DIR / "data" / "processed" / "manifest_final_samples.parquet"
DEFAULT_SETTINGS_PATH = MELCNN_DIR / "settings.json"
DEFAULT_SAMPLE_LENGTH_SEC = 15.0
TMP_BUILD_LOGMEL_DIR = (MELCNN_DIR / "data" / "tmp_build_log_mel").resolve()
MAX_SAVED_DEBUG_CLIPS = 10
SAVE_LOADED_AUDIO_DEBUG_CLIPS = True


def _load_data_sampling_settings(settings_path: Path) -> dict[str, object] | None:
    try:
        raw_text = settings_path.read_text()
    except Exception:
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            return None

    config = payload.get("data_sampling_settings")
    if not isinstance(config, dict):
        return None

    return config


def load_default_sample_length_from_settings(settings_path: Path) -> float:
    config = _load_data_sampling_settings(settings_path)
    if config is None:
        return DEFAULT_SAMPLE_LENGTH_SEC

    sample_length_sec = config.get("sample_length_sec")
    if not isinstance(sample_length_sec, (int, float)) or sample_length_sec <= 0:
        return DEFAULT_SAMPLE_LENGTH_SEC

    return float(sample_length_sec)


def load_target_genres_from_settings(settings_path: Path) -> list[str]:
    config = _load_data_sampling_settings(settings_path)
    if config is None:
        raise ValueError(
            f"Could not read settings.data_sampling_settings from settings file: {settings_path}"
        )

    target_genres = config.get("target_genres")
    if not isinstance(target_genres, list) or not target_genres:
        raise ValueError(
            f"Expected non-empty list at settings.data_sampling_settings.target_genres in {settings_path}"
        )

    cleaned_target_genres: list[str] = []
    for genre in target_genres:
        if not isinstance(genre, str) or not genre.strip():
            raise ValueError(
                f"Expected non-empty string values in settings.data_sampling_settings.target_genres in {settings_path}"
            )
        cleaned_target_genres.append(genre.strip())

    return list(dict.fromkeys(cleaned_target_genres))

DEFAULT_SAMPLE_LENGTH_FROM_SETTINGS = load_default_sample_length_from_settings(DEFAULT_SETTINGS_PATH)

# DEFAULT_OUT_ROOT = MELCNN_DIR / "cache" / f"logmel_dataset_{DEFAULT_SAMPLE_LENGTH_FROM_SETTINGS:g}s"
DEFAULT_OUT_ROOT = Path("/home/hsnguyen") / "model-training-data-cache" / f"logmel_dataset_{DEFAULT_SAMPLE_LENGTH_FROM_SETTINGS:g}s"

DEFAULT_NUM_WORKERS = min(4, (os.cpu_count() or 6))

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_MELS = 192
DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 256
DEFAULT_MIN_SECONDS = 1.0

ALL_INDEX_NAME = "logmel_manifest_all.parquet"
CONFIG_NAME = "logmel_config.json"
REPORT_NAME = "logmel_build_report.txt"

_SPLIT_DIR_MAP = {
    "training": "train",
    "train": "train",
    "validation": "val",
    "val": "val",
    "test": "test",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Path to sample-level parquet manifest. Defaults to manifest_final_samples.parquet.",
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS_PATH),
        help="Path to MelCNN-MGR/settings.json used for target_genres filtering.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="Output directory for split-grouped .npy files and parquet indexes.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate for log-mel extraction.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=DEFAULT_N_MELS,
        help="Number of mel bins.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=DEFAULT_N_FFT,
        help="FFT window size.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Hop length for the mel spectrogram.",
    )
    parser.add_argument(
        "--sample-length-sec",
        type=float,
        default=None,
        help="Override manifest sample_length_sec. If omitted, a single uniform manifest value is required.",
    )
    parser.add_argument(
        "--audio-backend",
        choices=("auto", "ffmpeg", "librosa"),
        default="auto",
        help="Decode backend. 'auto' prefers ffmpeg when available, otherwise librosa.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Parallel worker count for feature extraction.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
        help="Progress log interval in processed samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests or partial builds.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Remove the output root before rebuilding all .npy files and indexes.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def _normalize_split_label(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _SPLIT_DIR_MAP:
        raise ValueError(f"Unsupported split label: {value!r}")
    return _SPLIT_DIR_MAP[key]


def _sanitize_component(text: object) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    value = value.strip("._")
    return value or "unknown"


def _sample_cache_stem(sample_id: str) -> str:
    return hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:20]


def _build_output_path(out_root: Path, split_dir: str, genre_top: str, sample_id: str) -> Path:
    return out_root / split_dir / _sanitize_component(genre_top) / f"{_sample_cache_stem(sample_id)}.npy"


def _detect_audio_backend(requested: str) -> str:
    if requested != "auto":
        if requested == "ffmpeg" and shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg was requested but is not installed or not on PATH.")
        return requested
    return "ffmpeg" if shutil.which("ffmpeg") is not None else "librosa"


def _ensure_librosa_available() -> None:
    try:
        import librosa  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "librosa is required for log-mel extraction but could not be imported. "
            "Install dependencies from requirements.txt."
        ) from exc


def _sanitize_debug_component(text: object) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text).strip())
    value = value.strip("._")
    return value or "clip"


def _trim_saved_debug_clips(directory: Path, keep_latest: int) -> None:
    wav_files = sorted(directory.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale_path in wav_files[keep_latest:]:
        stale_path.unlink(missing_ok=True)


def _save_loaded_audio_debug_clip(
    waveform: np.ndarray,
    sample_rate: int,
    filepath: Path,
    start_sec: float,
    duration_sec: float,
    loader_tag: str,
) -> Path | None:
    if not SAVE_LOADED_AUDIO_DEBUG_CLIPS:
        return None
    if waveform is None or waveform.size == 0:
        return None

    TMP_BUILD_LOGMEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    millis = int((time.time() % 1) * 1000)
    clip_name = (
        f"{timestamp}-{millis:03d}-pid{os.getpid()}-"
        f"{_sanitize_debug_component(loader_tag)}-"
        f"{_sanitize_debug_component(filepath.stem)}-"
        f"{int(round(start_sec * 1000)):07d}ms-"
        f"{int(round(duration_sec * 1000)):07d}ms.wav"
    )
    clip_path = TMP_BUILD_LOGMEL_DIR / clip_name

    pcm16 = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype("<i2")
    with wave.open(str(clip_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm16.tobytes())

    _trim_saved_debug_clips(TMP_BUILD_LOGMEL_DIR, MAX_SAVED_DEBUG_CLIPS)
    logging.debug(
        "Saved decoded debug clip path=%s sample_rate=%s seconds=%.3f loader=%s",
        clip_path,
        sample_rate,
        len(pcm16) / sample_rate,
        loader_tag,
    )
    return clip_path


def _load_audio_ffmpeg(
    filepath: Path,
    sr: int,
    start_sec: float,
    duration_sec: float,
    mono: bool = True,
) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(filepath),
        "-ss",
        f"{start_sec:.6f}",
        "-t",
        f"{duration_sec:.6f}",
        "-vn",
        "-sn",
        "-dn",
        "-ar",
        str(sr),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-f", "f32le", "pipe:1"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed: {err}")

    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError("ffmpeg produced empty audio output")
    _save_loaded_audio_debug_clip(
        y,
        sample_rate=sr,
        filepath=filepath,
        start_sec=start_sec,
        duration_sec=duration_sec,
        loader_tag="ffmpeg",
    )
    return y


def _load_audio_librosa(
    filepath: Path,
    sr: int,
    start_sec: float,
    duration_sec: float,
    mono: bool = True,
) -> np.ndarray:
    import librosa

    y, _ = librosa.load(
        str(filepath),
        sr=sr,
        mono=mono,
        offset=float(start_sec),
        duration=float(duration_sec),
    )
    y = y.astype(np.float32, copy=False)
    _save_loaded_audio_debug_clip(
        y,
        sample_rate=sr,
        filepath=filepath,
        start_sec=start_sec,
        duration_sec=duration_sec,
        loader_tag="librosa",
    )
    return y


def _load_audio_segment(
    filepath: Path,
    sr: int,
    start_sec: float,
    duration_sec: float,
    audio_backend: str,
) -> np.ndarray:
    if audio_backend == "ffmpeg":
        return _load_audio_ffmpeg(filepath, sr=sr, start_sec=start_sec, duration_sec=duration_sec, mono=True)
    return _load_audio_librosa(filepath, sr=sr, start_sec=start_sec, duration_sec=duration_sec, mono=True)


def _normalize_to_fixed_duration(y: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    target_len = int(round(target_sec * sr))
    n = len(y)
    if n == target_len:
        return y.astype(np.float32, copy=False)
    if n > target_len:
        return y[:target_len].astype(np.float32, copy=False)

    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32, copy=False)


def _sanity_check_audio(y: np.ndarray, sr: int) -> tuple[bool, str]:
    if y is None or len(y) == 0:
        return False, "empty_audio"
    if not np.isfinite(y).all():
        return False, "non_finite_samples"
    if len(y) < int(DEFAULT_MIN_SECONDS * sr):
        return False, f"too_short_decoded(<{DEFAULT_MIN_SECONDS}s)"
    return True, ""


def _logmel_fixed_shape(
    y: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    n_frames: int,
) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Empty filters detected in mel frequency basis.*",
            category=UserWarning,
        )
        spec = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
        )
    logmel = np.log1p(spec)
    out = np.zeros((n_mels, n_frames), dtype=np.float32)
    frames = min(logmel.shape[1], n_frames)
    out[:, :frames] = logmel[:, :frames].astype(np.float32, copy=False)
    return out


def _is_valid_npy(path: Path, expected_shape: tuple[int, int]) -> bool:
    try:
        arr = np.load(str(path), mmap_mode="r")
        return tuple(arr.shape) == expected_shape
    except Exception:
        return False


def _load_manifest(path: Path, row_limit: int | None, target_genres: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest parquet not found: {path}")

    frame = pd.read_parquet(path)
    required = {
        "sample_id",
        "filepath",
        "genre_top",
        "segment_index",
        "segment_start_sec",
        "sample_length_sec",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    split_column = "final_split" if "final_split" in frame.columns else "split"
    if split_column not in frame.columns:
        raise ValueError("Manifest must contain either 'final_split' or 'split'.")

    if "selected_for_final_manifest" in frame.columns:
        frame = frame[frame["selected_for_final_manifest"] == True].copy()
    else:
        frame = frame.copy()

    if "reason_code" in frame.columns:
        frame = frame[frame["reason_code"].astype(str) == "OK"].copy()

    allowed_genres = {genre.strip() for genre in target_genres if genre.strip()}
    before_genre_filter = len(frame)
    frame = frame[frame["genre_top"].astype(str).isin(allowed_genres)].copy()
    logging.info(
        "Applied target_genres filter from settings: kept %d/%d manifest rows across genres=%s",
        len(frame),
        before_genre_filter,
        sorted(allowed_genres),
    )

    frame = frame[frame[split_column].notna()].copy()
    frame["split_dir"] = frame[split_column].map(_normalize_split_label)
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["filepath"] = frame["filepath"].astype(str)
    frame = frame.sort_values(["split_dir", "genre_top", "sample_id", "segment_index"], kind="stable")
    frame = frame.reset_index(drop=True)
    if row_limit is not None:
        frame = frame.head(row_limit).copy()
    frame["_row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def _resolve_sample_length(frame: pd.DataFrame, override_value: float | None) -> float:
    if override_value is not None:
        values = frame["sample_length_sec"].dropna().astype(float).unique().tolist()
        if values and any(abs(float(v) - float(override_value)) > 1e-6 for v in values):
            raise ValueError(
                "Manifest sample_length_sec values do not match --sample-length-sec override."
            )
        return float(override_value)

    values = sorted({round(float(v), 6) for v in frame["sample_length_sec"].dropna().tolist()})
    if not values:
        raise ValueError("Manifest contains no usable sample_length_sec values.")
    if len(values) != 1:
        raise ValueError(
            f"Expected a single uniform sample_length_sec value, found: {values}. "
            "Pass --sample-length-sec only if the manifest should be coerced to one length."
        )
    return float(values[0])


def _process_one_sample(task: tuple) -> dict[str, object]:
    (
        row_id,
        sample_id,
        filepath_str,
        genre_top,
        split_dir,
        start_sec,
        target_sec,
        out_root_str,
        sample_rate,
        n_mels,
        n_fft,
        hop_length,
        n_frames,
        audio_backend,
    ) = task

    filepath = Path(filepath_str)
    out_root = Path(out_root_str)
    output_path = _build_output_path(out_root, split_dir, genre_top, sample_id)
    expected_shape = (n_mels, n_frames)

    if output_path.exists():
        if _is_valid_npy(output_path, expected_shape):
            return {
                "_row_id": row_id,
                "logmel_path": str(output_path),
                "logmel_relpath": str(output_path.relative_to(out_root)),
                "logmel_status": "cached",
                "logmel_reason": "",
            }
        try:
            output_path.unlink()
        except Exception:
            pass

    if not filepath.exists():
        return {
            "_row_id": row_id,
            "logmel_path": "",
            "logmel_relpath": "",
            "logmel_status": "skipped",
            "logmel_reason": "missing_audio_file",
        }

    try:
        y = _load_audio_segment(
            filepath=filepath,
            sr=sample_rate,
            start_sec=float(start_sec),
            duration_sec=float(target_sec),
            audio_backend=audio_backend,
        )
    except Exception as exc:
        return {
            "_row_id": row_id,
            "logmel_path": "",
            "logmel_relpath": "",
            "logmel_status": "skipped",
            "logmel_reason": f"decode_fail:{type(exc).__name__}",
        }

    ok, reason = _sanity_check_audio(y, sample_rate)
    if not ok:
        return {
            "_row_id": row_id,
            "logmel_path": "",
            "logmel_relpath": "",
            "logmel_status": "skipped",
            "logmel_reason": reason,
        }

    try:
        y = _normalize_to_fixed_duration(y, sample_rate, target_sec)
        logmel = _logmel_fixed_shape(
            y=y,
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            n_frames=n_frames,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, logmel)
    except Exception as exc:
        return {
            "_row_id": row_id,
            "logmel_path": "",
            "logmel_relpath": "",
            "logmel_status": "skipped",
            "logmel_reason": f"logmel_fail:{type(exc).__name__}",
        }

    return {
        "_row_id": row_id,
        "logmel_path": str(output_path),
        "logmel_relpath": str(output_path.relative_to(out_root)),
        "logmel_status": "ok",
        "logmel_reason": "",
    }


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)


def _write_report(frame: pd.DataFrame, out_root: Path, config: dict[str, object]) -> None:
    lines: list[str] = []
    sep = "=" * 62
    lines.extend([
        sep,
        "2_build_log_mel_dataset report",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        sep,
        "",
        "Configuration:",
    ])
    for key, value in config.items():
        lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "Dataset summary:",
        f"  Total manifest rows : {len(frame):,}",
        f"  Usable log-mels     : {int(frame['logmel_usable'].sum()):,}",
        f"  Skipped rows        : {int((~frame['logmel_usable']).sum()):,}",
        "",
        "Rows by split/status:",
    ])
    if not frame.empty:
        split_status = frame.groupby(["split_dir", "logmel_status"]).size().sort_index()
        for (split_dir, status), count in split_status.items():
            lines.append(f"  {split_dir:<8} {status:<8} {count:>7,}")

        lines.extend(["", "Usable rows by split/genre:"])
        usable = frame[frame["logmel_usable"] == True]
        if not usable.empty:
            counts = usable.groupby(["split_dir", "genre_top"]).size().sort_index()
            for (split_dir, genre_top), count in counts.items():
                lines.append(f"  {split_dir:<8} {genre_top:<20} {count:>7,}")

        failed = frame[frame["logmel_usable"] == False]
        if not failed.empty:
            lines.extend(["", "Failure reasons:"])
            for reason, count in failed["logmel_reason"].value_counts().items():
                lines.append(f"  {reason:<28} {count:>7,}")

    lines.extend(["", sep, ""])
    (out_root / REPORT_NAME).write_text("\n".join(lines))


def _build_logmel_dataset(
    manifest: pd.DataFrame,
    out_root: Path,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    target_sec: float,
    audio_backend: str,
    num_workers: int,
    log_every: int,
) -> pd.DataFrame:
    n_frames = int(target_sec * sample_rate / hop_length)
    if n_frames <= 0:
        raise ValueError("Computed n_frames must be positive.")

    task_columns = [
        "_row_id",
        "sample_id",
        "filepath",
        "genre_top",
        "split_dir",
        "segment_start_sec",
    ]
    tasks = [
        (
            int(row_id),
            str(sample_id),
            str(filepath),
            str(genre_top),
            str(split_dir),
            float(segment_start_sec),
            float(target_sec),
            str(out_root),
            int(sample_rate),
            int(n_mels),
            int(n_fft),
            int(hop_length),
            int(n_frames),
            str(audio_backend),
        )
        for row_id, sample_id, filepath, genre_top, split_dir, segment_start_sec in manifest[task_columns].itertuples(index=False, name=None)
    ]

    logging.info(
        "[LogMel] Building %d sample segments | backend=%s | shape=(%d, %d) | workers=%d",
        len(tasks),
        audio_backend,
        n_mels,
        n_frames,
        num_workers,
    )

    results: list[dict[str, object]] = []
    n_ok = 0
    n_cached = 0
    n_skipped = 0
    t0 = time.time()
    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(_process_one_sample, tasks, chunksize=32), start=1):
            results.append(result)
            status = result["logmel_status"]
            if status == "ok":
                n_ok += 1
            elif status == "cached":
                n_cached += 1
            else:
                n_skipped += 1
            if i % log_every == 0 or i == len(tasks):
                logging.info(
                    "[LogMel] %d/%d processed | ok=%d | cached=%d | skipped=%d | elapsed=%.1fs",
                    i,
                    len(tasks),
                    n_ok,
                    n_cached,
                    n_skipped,
                    time.time() - t0,
                )

    result_df = pd.DataFrame(results)
    merged = manifest.merge(result_df, on="_row_id", how="left")
    merged["logmel_usable"] = merged["logmel_status"].isin(["ok", "cached"])
    merged["logmel_shape"] = f"{n_mels}x{n_frames}"
    merged["sample_rate"] = int(sample_rate)
    merged["n_mels"] = int(n_mels)
    merged["n_fft"] = int(n_fft)
    merged["hop_length"] = int(hop_length)
    return merged


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    manifest_path = Path(args.manifest)
    settings_path = Path(args.settings)
    out_root = Path(args.out_root)
    audio_backend = _detect_audio_backend(args.audio_backend)
    _ensure_librosa_available()
    target_genres = load_target_genres_from_settings(settings_path)

    logging.info("Manifest: %s", manifest_path)
    logging.info("Settings: %s", settings_path)
    logging.info("Output root: %s", out_root)
    logging.info("Audio backend: %s", audio_backend)
    logging.info("Target genres from settings: %s", target_genres)
    logging.info(
        "Default sample length from settings/fallback: %.3fs",
        DEFAULT_SAMPLE_LENGTH_FROM_SETTINGS,
    )

    manifest = _load_manifest(manifest_path, args.limit, target_genres)
    target_sec = _resolve_sample_length(manifest, args.sample_length_sec)
    n_frames = int(target_sec * args.sample_rate / args.hop_length)
    logging.info(
        "Actual target sample length used for extraction: %.3fs (%s)",
        target_sec,
        "cli override" if args.sample_length_sec is not None else "manifest",
    )

    if args.clear_cache and out_root.exists():
        logging.info("Removing existing output root: %s", out_root)
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    config = {
        "manifest": str(manifest_path),
        "settings": str(settings_path),
        "out_root": str(out_root),
        "rows_requested": len(manifest),
        "target_genres": target_genres,
        "sample_rate": args.sample_rate,
        "sample_length_sec": target_sec,
        "n_mels": args.n_mels,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_frames": n_frames,
        "audio_backend": audio_backend,
        "num_workers": args.num_workers,
        "limit": args.limit,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_root / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")

    t0 = time.time()
    index_df = _build_logmel_dataset(
        manifest=manifest,
        out_root=out_root,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        target_sec=target_sec,
        audio_backend=audio_backend,
        num_workers=args.num_workers,
        log_every=args.log_every,
    )
    elapsed = time.time() - t0

    index_df = index_df.drop(columns=["_row_id"])
    all_index_path = out_root / ALL_INDEX_NAME
    _write_parquet(all_index_path, index_df)

    for split_dir in ("train", "val", "test"):
        split_df = index_df[(index_df["split_dir"] == split_dir) & (index_df["logmel_usable"] == True)].copy()
        _write_parquet(out_root / f"logmel_manifest_{split_dir}.parquet", split_df)

    _write_report(index_df, out_root, config)

    logging.info("Wrote %s (%d rows)", all_index_path, len(index_df))
    for split_dir in ("train", "val", "test"):
        usable_count = int(((index_df["split_dir"] == split_dir) & (index_df["logmel_usable"] == True)).sum())
        logging.info("Usable %s rows: %d", split_dir, usable_count)
    logging.info("Completed in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        import sys as _sys
        if _sys.platform.startswith("linux"):
            subprocess.run(["stty", "sane"], check=False)