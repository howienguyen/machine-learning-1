"""Extract fixed-length audio segments from genre_downloads.

README-style usage example:

    python extract_mtg_processed_samples.py \
        --input-dir genre_downloads \
        --output-dir mtg-processed-samples \
        --segment-seconds 11 \
        --max-num-segments 3 \
        --min-duration-seconds 30 \
        --edge-buffer-seconds 20

Segment count selection per eligible file:
    - duration <= min-duration-seconds: skipped
    - min-duration-seconds < duration <= 30s: extract 1 segment
    - 30s < duration <= 60s: extract 2 segments when max-num-segments > 1
    - duration > 60s: extract up to max-num-segments segments
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


DEFAULT_INPUT_DIR = Path("genre_downloads/metal")
DEFAULT_OUTPUT_DIR = Path("mtg-processed-samples/metal")
SUPPORTED_EXTENSIONS = {".mp3", ".wav"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed-length samples from eligible audio files in genre_downloads "
            "subfolders, using per-file duration rules to reduce the number of segments "
            "for shorter clips."
        )
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
        default=11.0,
        help="Length of each extracted segment in seconds.",
    )
    parser.add_argument(
        "--max-num-segments",
        type=int,
        default=600,
        help=(
            "Maximum number of segments to extract from a single file. "
            "Files longer than min-duration-seconds but <= 30s use 1 segment; "
            "files > 30s and <= 60s use 2 segments when this value is greater than 1."
        ),
    )
    parser.add_argument(
        "--min-duration-seconds",
        type=float,
        default=30.0,
        help="Skip files with duration less than or equal to this threshold.",
    )
    parser.add_argument(
        "--edge-buffer-seconds",
        type=float,
        default=20.0,
        help="Preferred buffer to avoid at the beginning and end when placing segment start times.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output segment files if they already exist.",
    )
    return parser.parse_args()


def find_audio_files(input_dir: Path) -> list[Path]:
    audio_files: list[Path] = []

    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue

        for path in sorted(child.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                audio_files.append(path)

    return audio_files


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

    if duration_seconds >= segment_seconds + (2.0 * edge_buffer_seconds):
        start_min = edge_buffer_seconds
        start_max = duration_seconds - edge_buffer_seconds - segment_seconds
    else:
        start_min = 0.0
        start_max = max_start

    if start_max < start_min:
        start_min = 0.0
        start_max = max_start

    if num_segments == 1:
        return [start_min]

    return np.linspace(start_min, start_max, num=num_segments).tolist()


def choose_num_segments(duration_seconds: float, max_num_segments: int) -> int:
    if max_num_segments <= 1:
        return max_num_segments

    if duration_seconds <= 30.0:
        return 1

    if duration_seconds <= 60.0:
        return 2

    return max_num_segments


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


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"No mp3/wav files found in subfolders of {input_dir}")
        return

    processed_files = 0
    skipped_short = 0
    failed_files = 0
    written_segments = 0

    for audio_path in audio_files:
        try:
            waveform, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
            waveform = np.asarray(waveform, dtype=np.float32)
            duration_seconds = waveform.shape[-1] / float(sample_rate)

            if duration_seconds <= args.min_duration_seconds:
                skipped_short += 1
                print(
                    f"SKIP short: {audio_path} ({duration_seconds:.2f}s <= {args.min_duration_seconds:.2f}s)"
                )
                continue

            num_segments = choose_num_segments(
                duration_seconds=duration_seconds,
                max_num_segments=args.max_num_segments,
            )

            start_times = choose_start_times(
                duration_seconds=duration_seconds,
                segment_seconds=args.segment_seconds,
                num_segments=num_segments,
                edge_buffer_seconds=args.edge_buffer_seconds,
            )
            if len(start_times) != num_segments:
                failed_files += 1
                print(f"FAIL start-time generation: {audio_path}")
                continue

            relative_parent = audio_path.parent.relative_to(input_dir)
            output_parent = output_dir / relative_parent

            for index, start_seconds in enumerate(start_times, start=1):
                segment = slice_audio(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    start_seconds=start_seconds,
                    segment_seconds=args.segment_seconds,
                )

                expected_samples = int(round(args.segment_seconds * sample_rate))
                if segment.shape[-1] < expected_samples:
                    failed_files += 1
                    print(
                        f"FAIL short slice: {audio_path} seg{index:02d} got {segment.shape[-1]} samples, "
                        f"expected {expected_samples}"
                    )
                    continue

                output_name = f"{audio_path.stem}__seg{index:02d}_start{start_seconds:06.2f}s.wav"
                output_path = output_parent / output_name

                if output_path.exists() and not args.overwrite:
                    continue

                write_audio(output_path, segment, sample_rate)
                written_segments += 1

            processed_files += 1
            print(
                f"OK: {audio_path} -> {num_segments} segment(s) "
                f"at {[round(value, 2) for value in start_times]}"
            )

        except Exception as exc:
            failed_files += 1
            print(f"FAIL: {audio_path} ({exc})")

    print("\nSummary")
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"audio_files_found: {len(audio_files)}")
    print(f"processed_files: {processed_files}")
    print(f"skipped_short: {skipped_short}")
    print(f"failed_files: {failed_files}")
    print(f"written_segments: {written_segments}")


if __name__ == "__main__":
    main()