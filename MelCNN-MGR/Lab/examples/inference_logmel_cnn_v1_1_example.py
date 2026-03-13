#!/usr/bin/env python3
"""
Example: Log-Mel CNN v1.1 Genre Inference
=========================================
Demonstrates how to use the v1.1 inference module to classify audio files by
music genre.

Usage
-----
    # Predict on specific files
    python MelCNN-MGR/examples/inference_logmel_cnn_v1_1_example.py \
        --run-dir MelCNN-MGR/models/logmel-cnn-v1_1-20260311-120000 \
        --files song1.mp3 song2.mp3

    # Auto-select random samples from a manifest
    python MelCNN-MGR/examples/inference_logmel_cnn_v1_1_example.py \
        --run-dir MelCNN-MGR/models/logmel-cnn-v1_1-20260311-120000 \
        --subset small --random 5

    # Single-crop mode
    python MelCNN-MGR/examples/inference_logmel_cnn_v1_1_example.py \
        --run-dir MelCNN-MGR/models/logmel-cnn-v1_1-20260311-120000 \
        --subset small --random 3 --mode single_crop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_MELCNN_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_MELCNN_DIR))

from inference_logmel_cnn_v1_1 import AUDIO_BACKEND, LogMelCNNV11Inference, PredictionResult


def _random_test_files(subset: str, n: int) -> list[tuple[str, str]]:
    """Pick ``n`` random test files and return ``(filepath, true_genre)`` pairs."""
    import numpy as np
    import pandas as pd

    processed_dir = _MELCNN_DIR / "data" / "processed"
    test_path = processed_dir / f"test_{subset}.parquet"
    if not test_path.exists():
        sys.exit(f"Test manifest not found: {test_path}")

    df = pd.read_parquet(test_path)
    n = min(n, len(df))
    indices = np.random.choice(len(df), size=n, replace=False)
    return [(df.iloc[i]["filepath"], df.iloc[i]["genre_top"]) for i in indices]


def _print_result(result: PredictionResult, true_genre: str = "?") -> None:
    name = Path(result.file).name
    match = "✓" if result.genre == true_genre else ("✗" if true_genre != "?" else " ")

    print(f"\n{'─' * 60}")
    print(f"  File       : {name}")
    print(f"  Predicted  : {result.genre}  ({result.confidence:.2%})")
    print(f"  True genre : {true_genre}  {match}")
    print(f"  Mode       : {result.mode}")

    if result.crops:
        for idx, crop in enumerate(result.crops, 1):
            print(f"    Crop {idx}: {crop.genre} ({crop.confidence:.2%})")

    print("  Top-3      : ", end="")
    for genre, prob in result.top_k(3):
        print(f"{genre} ({prob:.1%})  ", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run genre inference using a trained Log-Mel CNN v1.1 model.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to the training run directory (contains .keras + norm_stats.npz).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        type=Path,
        default=[],
        help="Audio files to classify.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="FMA subset (tiny/small/medium) used with --random.",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        metavar="N",
        help="Pick N random samples from the test manifest of --subset.",
    )
    parser.add_argument(
        "--mode",
        choices=["single_crop", "three_crop"],
        default="three_crop",
        help="Inference mode: three_crop (default) or single_crop.",
    )
    args = parser.parse_args()

    if not args.files and args.random == 0:
        parser.error("Provide --files and/or --random N (with --subset).")
    if args.random > 0 and args.subset is None:
        parser.error("--random requires --subset (e.g. --subset small).")

    print(f"Loading model from: {args.run_dir}")
    engine = LogMelCNNV11Inference(args.run_dir)
    print(f"  Genre classes ({engine.n_classes}): {engine.genre_classes}")
    print(f"  Model file   : {engine.model_path.name}")
    print(f"  Audio backend: {AUDIO_BACKEND}")

    items: list[tuple[str, str]] = []
    for filepath in args.files:
        items.append((str(filepath), "?"))

    if args.random > 0:
        items.extend(_random_test_files(args.subset, args.random))

    print(f"\nRunning inference (mode={args.mode}) on {len(items)} file(s) ...\n")

    correct = 0
    total_with_label = 0
    for filepath, true_genre in items:
        try:
            result = engine.predict(filepath, mode=args.mode)
            _print_result(result, true_genre)
            if true_genre != "?":
                total_with_label += 1
                if result.genre == true_genre:
                    correct += 1
        except Exception as exc:
            print(f"\n[ERROR] {Path(filepath).name}: {exc}", file=sys.stderr)

    if total_with_label > 0:
        acc = correct / total_with_label
        print(f"\n{'═' * 60}")
        print(f"  Accuracy: {correct}/{total_with_label} = {acc:.2%}")
        print(f"{'═' * 60}")


if __name__ == "__main__":
    main()