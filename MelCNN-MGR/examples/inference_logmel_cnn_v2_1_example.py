#!/usr/bin/env python3
"""Example: Log-Mel CNN v2.1 Family Genre Inference.

This example uses hardcoded settings and predicts every audio file inside
audio_demo using the config-driven v2-family inference module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MELCNN_DIR = _SCRIPT_DIR.parent
_WORKSPACE_DIR = _MELCNN_DIR.parent
sys.path.insert(0, str(_MELCNN_DIR))

from inference_logmel_cnn_v2_1 import AUDIO_BACKEND, LogMelCNNV21Inference, PredictionResult


# Hardcoded example settings.
RUN_DIR = _MELCNN_DIR / "models" / "logmel-cnn-v2-20260311-171117"
AUDIO_DIR = _WORKSPACE_DIR / "audio_demo"
INFERENCE_MODE = "three_crop"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


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
    if not RUN_DIR.exists():
        raise FileNotFoundError(
            f"Configured RUN_DIR does not exist: {RUN_DIR}\n"
            "Update RUN_DIR in this example to point at a valid v2-family training run."
        )
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(f"Configured AUDIO_DIR does not exist: {AUDIO_DIR}")

    audio_files = _find_audio_files(AUDIO_DIR)
    if not audio_files:
        raise FileNotFoundError(f"No supported audio files found in: {AUDIO_DIR}")

    print(f"Loading model from: {RUN_DIR}")
    engine = LogMelCNNV21Inference(RUN_DIR)
    print(f"  Genre classes ({engine.n_classes}): {engine.genre_classes}")
    print(f"  Model file   : {engine.model_path.name}")
    print(f"  Audio backend: {AUDIO_BACKEND}")
    print(f"  Sample rate  : {engine.sample_rate}")
    print(f"  Clip duration: {engine.clip_duration}s")
    print(f"  Log-mel shape: {engine.logmel_shape}")

    print(f"\nRunning inference (mode={INFERENCE_MODE}) on {len(audio_files)} file(s) from {AUDIO_DIR} ...\n")

    failed = 0
    for audio_path in audio_files:
        try:
            result = engine.predict(audio_path, mode=INFERENCE_MODE)
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