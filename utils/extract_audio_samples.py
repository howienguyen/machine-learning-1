"""Extract fixed-length audio segments from genre_downloads.

Verbose mode (default):

    python utils/extract_audio_samples.py

Quiet mode for less terminal stress:

    python utils/extract_audio_samples.py --quiet

Terminal note: (If you don't use --quiet)

    On some VS Code integrated terminals, very noisy multiprocessing output can
    temporarily corrupt terminal rendering so typed characters appear invisible
    even though input still works. If that happens after the script exits, run:

        stty sane

    The default logging prints per-file messages. Use --quiet if your terminal
    rendering becomes slow or glitchy.

Segment count selection per eligible file:
    - duration <= min-duration-seconds: skipped
    - min-duration-seconds < duration <= 30s: extract 1 segment
    - 30s < duration <= 60s: extract 2 segments when max-num-segments > 1
    - duration > 60s: extract up to max-num-segments segments, capped by how
      many full non-overlapping segments fit in the file duration

Segment placement behavior:
        - when possible, segment starts avoid the first and last edge-buffer-seconds
        - if the file is too short to preserve both edge buffers, the extractor keeps
            as much leading buffer as possible and sacrifices the trailing buffer first
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_INPUT_DIR = Path("genre_downloads/my-collection")
DEFAULT_OUTPUT_DIR = Path("genre_downloads/cropped-audio/my-collection")
SUPPORTED_EXTENSIONS = {".mp3", ".wav"}
_FFMPEG = shutil.which("ffmpeg")
_FFPROBE = shutil.which("ffprobe")
_librosa_module = None


def _get_librosa():
    global _librosa_module
    if _librosa_module is None:
        import librosa

        _librosa_module = librosa
    return _librosa_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed-length samples from eligible audio files in genre_downloads "
            "subfolders, using per-file duration rules to reduce the number of segments "
            "for shorter clips."
        ),
        epilog=(
            "Logging behavior:\n"
            "  default   show everything (per-file OK/SKIP, worker summaries, etc.)\n"
            "  --quiet   show failures, worker summaries, and final summary only\n\n"
            "If your integrated terminal finishes in a strange state where typed\n"
            "characters become invisible, use --quiet or run: stty sane"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input root containing per-genre subfolders with mp3/wav files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root where extracted wav segments are written in mirrored subfolders.",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=31.0,
        help="Length of each extracted segment in seconds.",
    )
    parser.add_argument(
        "--max-num-segments",
        type=int,
        default=240,
        help=(
            "Maximum number of segments to extract from a single file. "
            "Files longer than min-duration-seconds but <= 30s use 1 segment; "
            "files > 30s and <= 60s use 2 segments when this value is greater than 1; "
            "files > 60s are capped by how many full non-overlapping segments fit in the audio."
        ),
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=31.0,
        help="Skip files with duration less than or equal to this threshold.",
    )
    parser.add_argument(
        "--edge-buffer-seconds",
        type=float,
        default=20.0,
        help=(
            "Preferred buffer to avoid at the beginning and end when placing segment start times. "
            "If both buffers do not fit, the extractor preserves as much leading buffer as possible."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output segment files if they already exist.",
    )
    parser.add_argument(
        "--backend",
        choices=["ffmpeg", "librosa"],
        default="ffmpeg",
        help=(
            "Audio extraction backend. Defaults to ffmpeg when ffmpeg/ffprobe are available; "
            "otherwise falls back to librosa."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help=(
            "Number of worker processes used to handle top-level subfolders in parallel. "
            "Use 1 to disable multiprocessing."
        ),
    )
    parser.add_argument(
        "--parallel-by",
        choices=["subdir", "file"],
        default="file",
        help=(
            "How work is divided across multiple processes. "
            "'subdir' assigns one top-level input subfolder per worker; "
            "'file' assigns individual audio files to workers."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Disable per-file skip/success messages. By default, all files are logged. "
            "Use this if your terminal rendering is slow or glitchy."
        ),
    )
    return parser.parse_args()


def _log(message: str, *, flush: bool = False) -> None:
    print(message, flush=flush)


def _log_verbose(quiet: bool, message: str, *, flush: bool = False) -> None:
    if not quiet:
        print(message, flush=flush)


def _log_progress(quiet: bool, message: str, *, done: bool = False) -> None:
    if quiet:
        return
    print(message, flush=True)


def find_input_subdirs(input_dir: Path) -> list[Path]:
    return [child for child in sorted(input_dir.iterdir()) if child.is_dir()]


def find_audio_files(input_dir: Path) -> list[Path]:
    audio_files: list[Path] = []

    for child in find_input_subdirs(input_dir):
        for path in sorted(child.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(path)

    return audio_files


def find_audio_files_in_subdir(folder: Path) -> list[Path]:
    return [path for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]


def choose_start_times(
    duration_seconds: float,
    segment_seconds: float,
    num_segments: int,
    edge_buffer_seconds: float,
) -> list[float]:
    if duration_seconds < segment_seconds or num_segments < 1:
        return []

    max_start = duration_seconds - segment_seconds
    if max_start <= 0:
        return [0.0]

    start_min = min(edge_buffer_seconds, max_start)

    if duration_seconds >= segment_seconds + (2.0 * edge_buffer_seconds):
        start_max = duration_seconds - edge_buffer_seconds - segment_seconds
    else:
        start_max = max_start

    if num_segments == 1:
        return [start_min]

    return np.linspace(start_min, start_max, num=num_segments).tolist()


def choose_num_segments(
    duration_seconds: float,
    segment_seconds: float,
    max_num_segments: int,
) -> int:
    if max_num_segments <= 0:
        return 0

    if max_num_segments == 1:
        return 1

    if duration_seconds <= 30.0:
        return 1

    if duration_seconds <= 60.0:
        return 2

    max_segments_by_duration = max(
        0,
        math.floor((duration_seconds + 1e-9) / max(1e-9, segment_seconds)),
    )

    return min(max_num_segments, max_segments_by_duration)


def resolve_backend(requested_backend: str | None) -> str:
    if requested_backend == "ffmpeg":
        if not _FFMPEG or not _FFPROBE:
            raise RuntimeError(
                "ffmpeg backend requested, but ffmpeg and/or ffprobe are not available on PATH."
            )
        return "ffmpeg"

    if requested_backend == "librosa":
        _get_librosa()
        return "librosa"

    if _FFMPEG and _FFPROBE:
        return "ffmpeg"

    print("[WARN] ffmpeg/ffprobe not found on PATH; falling back to librosa backend.")
    _get_librosa()
    return "librosa"


def probe_duration_ffprobe(path: Path) -> float:
    if not _FFPROBE:
        raise RuntimeError("ffprobe is not available on PATH.")

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
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed to read duration")

    text = result.stdout.strip()
    if not text:
        raise RuntimeError("ffprobe returned an empty duration value")

    return float(text)


def extract_segment_ffmpeg(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    segment_seconds: float,
    overwrite: bool,
) -> None:
    if not _FFMPEG:
        raise RuntimeError("ffmpeg is not available on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-ss",
        f"{start_seconds:.6f}",
        "-t",
        f"{segment_seconds:.6f}",
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed to extract audio segment")


def slice_audio(
    waveform: np.ndarray,
    sample_rate: int,
    start_seconds: float,
    segment_seconds: float,
) -> np.ndarray:
    start_sample = int(round(start_seconds * sample_rate))
    segment_samples = int(round(segment_seconds * sample_rate))
    end_sample = start_sample + segment_samples

    if waveform.ndim == 1:
        return waveform[start_sample:end_sample]

    return waveform[:, start_sample:end_sample]


def write_audio(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.ndim == 1:
        sf.write(path, waveform, sample_rate)
        return

    sf.write(path, waveform.T, sample_rate)


def process_audio_file(
    audio_path: Path,
    input_dir: Path,
    output_dir: Path,
    backend: str,
    segment_seconds: float,
    max_num_segments: int,
    min_duration_seconds: float,
    edge_buffer_seconds: float,
    overwrite: bool,
    quiet: bool,
) -> dict[str, int]:
    written_segments = 0

    try:
        if backend == "ffmpeg":
            duration_seconds = probe_duration_ffprobe(audio_path)
            waveform = None
            sample_rate = None
        else:
            librosa = _get_librosa()
            waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
            waveform = np.asarray(waveform, dtype=np.float32)
            duration_seconds = waveform.shape[-1] / float(sample_rate)

        source = audio_path.relative_to(input_dir)
        source_label = f"[file:{source}]"

        if duration_seconds <= min_duration_seconds:
            _log_progress(
                quiet,
                f"{source_label} SKIP duration={duration_seconds:.2f}s",
            )
            return {
                "audio_files_found": 1,
                "processed_files": 0,
                "skipped_short": 1,
                "failed_files": 0,
                "written_segments": 0,
            }

        num_segments = choose_num_segments(
            duration_seconds=duration_seconds,
            segment_seconds=segment_seconds,
            max_num_segments=max_num_segments,
        )

        if num_segments < 1:
            _log_progress(
                quiet,
                f"{source_label} SKIP duration={duration_seconds:.2f}s segment_seconds={segment_seconds:.2f}s",
            )
            return {
                "audio_files_found": 1,
                "processed_files": 0,
                "skipped_short": 1,
                "failed_files": 0,
                "written_segments": 0,
            }

        start_times = choose_start_times(
            duration_seconds=duration_seconds,
            segment_seconds=segment_seconds,
            num_segments=num_segments,
            edge_buffer_seconds=edge_buffer_seconds,
        )
        if len(start_times) != num_segments:
            _log(f"FAIL start-time generation: {audio_path}", flush=True)
            return {
                "audio_files_found": 1,
                "processed_files": 0,
                "skipped_short": 0,
                "failed_files": 1,
                "written_segments": 0,
            }

        relative_parent = audio_path.parent.relative_to(input_dir)
        output_parent = output_dir / relative_parent

        for index, start_seconds in enumerate(start_times, start=1):
            output_name = f"{audio_path.stem}__seg{index:02d}_start{start_seconds:06.2f}s.wav"
            output_path = output_parent / output_name

            if output_path.exists() and not overwrite:
                continue

            if backend == "ffmpeg":
                extract_segment_ffmpeg(
                    input_path=audio_path,
                    output_path=output_path,
                    start_seconds=start_seconds,
                    segment_seconds=segment_seconds,
                    overwrite=overwrite,
                )
            else:
                segment = slice_audio(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    start_seconds=start_seconds,
                    segment_seconds=segment_seconds,
                )

                expected_samples = int(round(segment_seconds * sample_rate))
                if segment.shape[-1] < expected_samples:
                    _log(
                        f"FAIL short slice: {audio_path} seg{index:02d} got {segment.shape[-1]} samples, "
                        f"expected {expected_samples}",
                        flush=True,
                    )
                    return {
                        "audio_files_found": 1,
                        "processed_files": 0,
                        "skipped_short": 0,
                        "failed_files": 1,
                        "written_segments": written_segments,
                    }

                write_audio(output_path, segment, sample_rate)

            written_segments += 1
            _log_progress(
                quiet,
                f"{source_label} WRITE -> {output_path.relative_to(output_dir)} seg={index:02d} start={start_seconds:.2f}s",
            )

        _log_progress(
            quiet,
            f"{source_label} OK segments={num_segments} starts={[round(value, 2) for value in start_times]}",
        )
        return {
            "audio_files_found": 1,
            "processed_files": 1,
            "skipped_short": 0,
            "failed_files": 0,
            "written_segments": written_segments,
        }

    except Exception as exc:
        _log(f"FAIL: {audio_path} ({exc})", flush=True)
        return {
            "audio_files_found": 1,
            "processed_files": 0,
            "skipped_short": 0,
            "failed_files": 1,
            "written_segments": written_segments,
        }


def process_subfolder(
    subdir: Path,
    input_dir: Path,
    output_dir: Path,
    backend: str,
    segment_seconds: float,
    max_num_segments: int,
    min_duration_seconds: float,
    edge_buffer_seconds: float,
    overwrite: bool,
    quiet: bool,
) -> dict[str, int]:
    audio_files = find_audio_files_in_subdir(subdir)
    total_audio_files = len(audio_files)
    estimated_total_segments = max(1, total_audio_files * max_num_segments)
    processed_files = 0
    skipped_short = 0
    failed_files = 0
    written_segments = 0

    _log(f"[worker:{subdir.name}] audio_files_found={total_audio_files}", flush=True)

    for file_index, audio_path in enumerate(audio_files, start=1):
        try:
            progress_pct = 100.0 * written_segments / estimated_total_segments
            if backend == "ffmpeg":
                duration_seconds = probe_duration_ffprobe(audio_path)
                waveform = None
                sample_rate = None
            else:
                librosa = _get_librosa()
                waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
                waveform = np.asarray(waveform, dtype=np.float32)
                duration_seconds = waveform.shape[-1] / float(sample_rate)

            if duration_seconds <= min_duration_seconds:
                skipped_short += 1
                _log_progress(
                    quiet,
                    "[worker:{worker}] est={done:04d}/{total:04d} ({pct:5.1f}%) SKIP {source} duration={duration:.2f}s".format(
                        worker=subdir.name,
                        done=written_segments,
                        total=estimated_total_segments,
                        pct=progress_pct,
                        source=audio_path.relative_to(input_dir),
                        duration=duration_seconds,
                    ),
                )
                continue

            num_segments = choose_num_segments(
                duration_seconds=duration_seconds,
                segment_seconds=segment_seconds,
                max_num_segments=max_num_segments,
            )

            if num_segments < 1:
                skipped_short += 1
                _log_progress(
                    quiet,
                    "[worker:{worker}] est={done:04d}/{total:04d} ({pct:5.1f}%) SKIP {source} duration={duration:.2f}s segment_seconds={segment_seconds:.2f}s".format(
                        worker=subdir.name,
                        done=written_segments,
                        total=estimated_total_segments,
                        pct=progress_pct,
                        source=audio_path.relative_to(input_dir),
                        duration=duration_seconds,
                        segment_seconds=segment_seconds,
                    ),
                )
                continue

            start_times = choose_start_times(
                duration_seconds=duration_seconds,
                segment_seconds=segment_seconds,
                num_segments=num_segments,
                edge_buffer_seconds=edge_buffer_seconds,
            )
            if len(start_times) != num_segments:
                failed_files += 1
                _log(f"FAIL start-time generation: {audio_path}", flush=True)
                continue

            relative_parent = audio_path.parent.relative_to(input_dir)
            output_parent = output_dir / relative_parent

            for index, start_seconds in enumerate(start_times, start=1):
                output_name = f"{audio_path.stem}__seg{index:02d}_start{start_seconds:06.2f}s.wav"
                output_path = output_parent / output_name

                if output_path.exists() and not overwrite:
                    continue

                if backend == "ffmpeg":
                    extract_segment_ffmpeg(
                        input_path=audio_path,
                        output_path=output_path,
                        start_seconds=start_seconds,
                        segment_seconds=segment_seconds,
                        overwrite=overwrite,
                    )
                else:
                    segment = slice_audio(
                        waveform=waveform,
                        sample_rate=sample_rate,
                        start_seconds=start_seconds,
                        segment_seconds=segment_seconds,
                    )

                    expected_samples = int(round(segment_seconds * sample_rate))
                    if segment.shape[-1] < expected_samples:
                        failed_files += 1
                        _log(
                            f"FAIL short slice: {audio_path} seg{index:02d} got {segment.shape[-1]} samples, "
                            f"expected {expected_samples}",
                            flush=True,
                        )
                        continue

                    write_audio(output_path, segment, sample_rate)
                written_segments += 1
                progress_pct = 100.0 * written_segments / estimated_total_segments
                _log_progress(
                    quiet,
                    "[worker:{worker}] est={done:04d}/{total:04d} ({pct:5.1f}%) WRITE -> {output} seg={index:02d} start={start:.2f}s".format(
                        worker=subdir.name,
                        done=written_segments,
                        total=estimated_total_segments,
                        pct=progress_pct,
                        output=output_path.relative_to(output_dir),
                        index=index,
                        start=start_seconds,
                    ),
                )

            processed_files += 1
            _log_progress(
                quiet,
                "[worker:{worker}] est={done:04d}/{total:04d} ({pct:5.1f}%) OK file={current}/{file_total} segments={num_segments} starts={starts}".format(
                    worker=subdir.name,
                    done=written_segments,
                    total=estimated_total_segments,
                    pct=progress_pct,
                    current=file_index,
                    file_total=total_audio_files,
                    num_segments=num_segments,
                    starts=[round(value, 2) for value in start_times],
                ),
            )

        except Exception as exc:
            failed_files += 1
            _log(f"FAIL: {audio_path} ({exc})", flush=True)

    _log_progress(
        quiet,
        "[worker:{worker}] est={done:04d}/{total:04d} ({pct:5.1f}%) completed {processed}/{file_total} files | skipped_short={skipped} failed={failed} written_segments={written}".format(
            worker=subdir.name,
            done=written_segments,
            total=estimated_total_segments,
            pct=(100.0 * written_segments / estimated_total_segments),
            processed=processed_files,
            file_total=total_audio_files,
            skipped=skipped_short,
            failed=failed_files,
            written=written_segments,
        ),
        done=True,
    )

    _log(
        "[worker:{name}] processed={processed} skipped_short={skipped} failed={failed} written_segments={written}".format(
            name=subdir.name,
            processed=processed_files,
            skipped=skipped_short,
            failed=failed_files,
            written=written_segments,
        ),
        flush=True,
    )

    return {
        "audio_files_found": len(audio_files),
        "processed_files": processed_files,
        "skipped_short": skipped_short,
        "failed_files": failed_files,
        "written_segments": written_segments,
    }


def main() -> None:
    args = parse_args()
    backend = resolve_backend(args.backend)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    subdirs = find_input_subdirs(input_dir)
    audio_files = find_audio_files(input_dir)
    if not subdirs or not audio_files:
        print(f"No mp3/wav files found in subfolders of {input_dir}")
        return

    print(f"backend: {backend}")
    requested_workers = max(1, args.num_workers)
    parallel_items = subdirs if args.parallel_by == "subdir" else audio_files
    max_workers = min(requested_workers, len(parallel_items), os.cpu_count() or 1)
    print(f"num_workers: {max_workers}")
    print(f"parallel_by: {args.parallel_by}")

    processed_files = 0
    skipped_short = 0
    failed_files = 0
    written_segments = 0

    if max_workers == 1:
        for subdir in subdirs:
            result = process_subfolder(
                subdir=subdir,
                input_dir=input_dir,
                output_dir=output_dir,
                backend=backend,
                segment_seconds=args.segment_seconds,
                max_num_segments=args.max_num_segments,
                min_duration_seconds=args.min_duration_seconds,
                edge_buffer_seconds=args.edge_buffer_seconds,
                overwrite=args.overwrite,
                quiet=args.quiet,
            )
            processed_files += result["processed_files"]
            skipped_short += result["skipped_short"]
            failed_files += result["failed_files"]
            written_segments += result["written_segments"]
    elif args.parallel_by == "file":
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_audio_file,
                    audio_path,
                    input_dir,
                    output_dir,
                    backend,
                    args.segment_seconds,
                    args.max_num_segments,
                    args.min_duration_seconds,
                    args.edge_buffer_seconds,
                    args.overwrite,
                    args.quiet,
                ): audio_path
                for audio_path in audio_files
            }

            for future in as_completed(futures):
                audio_path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_files += 1
                    _log(f"FAIL worker file: {audio_path} ({exc})")
                    continue

                processed_files += result["processed_files"]
                skipped_short += result["skipped_short"]
                failed_files += result["failed_files"]
                written_segments += result["written_segments"]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_subfolder,
                    subdir,
                    input_dir,
                    output_dir,
                    backend,
                    args.segment_seconds,
                    args.max_num_segments,
                    args.min_duration_seconds,
                    args.edge_buffer_seconds,
                    args.overwrite,
                    args.quiet,
                ): subdir
                for subdir in subdirs
            }

            for future in as_completed(futures):
                subdir = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failed_files += len(find_audio_files_in_subdir(subdir))
                    _log(f"FAIL worker: {subdir} ({exc})")
                    continue

                processed_files += result["processed_files"]
                skipped_short += result["skipped_short"]
                failed_files += result["failed_files"]
                written_segments += result["written_segments"]

    print("\nSummary")
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"audio_files_found: {len(audio_files)}")
    print(f"processed_files: {processed_files}")
    print(f"processed_pct: {(100.0 * processed_files / len(audio_files)) if audio_files else 0.0:.1f}%")
    print(f"skipped_short: {skipped_short}")
    print(f"failed_files: {failed_files}")
    print(f"written_segments: {written_segments}")


if __name__ == "__main__":
    try:
        main()
    finally:
        import sys as _sys
        if _sys.platform.startswith("linux"):
            subprocess.run(["stty", "sane"], check=False)