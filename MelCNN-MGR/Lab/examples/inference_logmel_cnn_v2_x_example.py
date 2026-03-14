#!/usr/bin/env python3
"""Example: Log-Mel CNN v2.x family genre inference.

This example runs inference on every supported file in a chosen audio
directory using a model directory passed explicitly to
MelCNN-MGR/model_inference/inference_logmel_cnn_v2_x.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MELCNN_DIR = _SCRIPT_DIR.parents[1]
_WORKSPACE_DIR = _MELCNN_DIR.parent
sys.path.insert(0, str(_MELCNN_DIR))

from model_inference.inference_logmel_cnn_v2_x import (
    AUDIO_BACKEND,
    LogMelCNNV2XInference,
    PredictionResult,
)


DEFAULT_AUDIO_DIR = _WORKSPACE_DIR / "MelCNN-MGR/resources/audio_demo"
DEFAULT_INFERENCE_MODE = "three_crop"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Log-Mel CNN v2.x example inference over an audio directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("MelCNN-MGR/demo-models/logmel-cnn-v2_1-20260313-081401"),
        help="Path to the model directory to use, for example MelCNN-MGR/demo-models/logmel-cnn-v2_1-20260313-081401.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory containing audio files to classify.",
    )
    parser.add_argument(
        "--mode",
        choices=["single_crop", "three_crop"],
        default=DEFAULT_INFERENCE_MODE,
        help="Inference mode to use for all files.",
    )
    return parser.parse_args()


def _find_audio_files(audio_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(audio_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _print_result(result: PredictionResult) -> None:
    name = Path(result.file).name

    print(f"\n{'─' * 60}")
    print(f"  File       : {name}")
    print(f"  Predicted  : {result.genre}  ({result.confidence:.2%})")
    print(f"  Mode       : {result.mode}")

    if result.crops:
        for idx, crop in enumerate(result.crops, 1):
            print(f"    Crop {idx}: {crop.genre} ({crop.confidence:.2%})")

    print("  Top-3      : ", end="")
    for genre, prob in result.top_k(3):
        print(f"{genre} ({prob:.1%})  ", end="")
    print()


def main() -> None:
    args = _parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Configured model directory does not exist: {model_dir}")
    audio_dir = args.audio_dir.expanduser().resolve()
    if not audio_dir.exists():
        raise FileNotFoundError(f"Configured audio directory does not exist: {audio_dir}")

    audio_files = _find_audio_files(audio_dir)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in: {audio_dir}")

    print(f"Loading model from: {model_dir}")
    engine = LogMelCNNV2XInference(model_dir)
    print(f"  Genre classes ({engine.n_classes}): {engine.genre_classes}")
    print(f"  Model file   : {engine.model_path.name}")
    print(f"  Audio backend: {AUDIO_BACKEND}")
    print(f"  Sample rate  : {engine.sample_rate}")
    print(f"  Clip duration: {engine.clip_duration}s")
    print(f"  Log-mel shape: {engine.logmel_shape}")

    print(f"\nRunning inference (mode={args.mode}) on {len(audio_files)} file(s) from {audio_dir} ...\n")

    failed = 0
    for audio_path in audio_files:
        try:
            result = engine.predict(audio_path, mode=args.mode)
            _print_result(result)
        except Exception as exc:
            failed += 1
            print(f"\n[ERROR] {audio_path.name}: {exc}", file=sys.stderr)

    print(f"\n{'═' * 60}")
    print(f"  Completed : {len(audio_files) - failed}/{len(audio_files)} succeeded")
    print(f"  Failed    : {failed}")
    print(f"{'═' * 60}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()