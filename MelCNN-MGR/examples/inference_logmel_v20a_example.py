#!/usr/bin/env python3
"""
Example: MelCNN-MGR Genre Inference (Log-Mel CNN v20a)
=====================================================
Demonstrates how to use the inference module to classify audio files
by music genre using a trained Log-Mel CNN v20a model.

Usage
-----
    # Predict on specific files
    python MelCNN-MGR/examples/inference_logmel_v20a_example.py \\
        --run-dir MelCNN-MGR/models/logmel-cnn-v20a-20260308-013615 \\
        --files song1.mp3 song2.mp3 song3.mp3

    # Auto-select random samples from a manifest (for quick testing)
    python MelCNN-MGR/examples/inference_logmel_v20a_example.py \
        --run-dir MelCNN-MGR/models/logmel-cnn-v20a-20260308-013615 \
        --subset small --random 5

    # Single-crop mode (faster, slightly less accurate)
    python MelCNN-MGR/examples/inference_logmel_v20a_example.py \
        --run-dir MelCNN-MGR/models/logmel-cnn-v20a-20260308-013615 \\
        --subset small --random 3 --mode single_crop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Resolve project root so imports work when invoked from anywhere ───────────
_SCRIPT_DIR = Path(__file__).resolve().parent      # .../MelCNN-MGR/examples
_MELCNN_DIR = _SCRIPT_DIR.parent                   # .../MelCNN-MGR
_WORKSPACE  = _MELCNN_DIR.parent                   # .../machine-learning-1
sys.path.insert(0, str(_MELCNN_DIR))

from inference_logmel_v20a import MelCNNInference, PredictionResult


def _random_test_files(subset: str, n: int) -> list[tuple[str, str]]:
    """Pick *n* random files from the test manifest.  Returns [(path, true_genre), ...]."""
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
    """Pretty-print a single prediction result."""
    name = Path(result.file).name
    match = "✓" if result.genre == true_genre else ("✗" if true_genre != "?" else " ")

    print(f"\n{'─' * 60}")
    print(f"  File       : {name}")
    print(f"  Predicted  : {result.genre}  ({result.confidence:.2%})")
    print(f"  True genre : {true_genre}  {match}")
    print(f"  Mode       : {result.mode}")

    if result.crops:
        for i, crop in enumerate(result.crops, 1):
            print(f"    Crop {i}: {crop.genre} ({crop.confidence:.2%})")

    print(f"  Top-3      : ", end="")
    for genre, prob in result.top_k(3):
        print(f"{genre} ({prob:.1%})  ", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run genre inference using a trained MelCNN-MGR v20a model.",
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Path to the training run directory (contains .keras + norm_stats.npz).",
    )
    parser.add_argument(
        "--files", nargs="*", type=Path, default=[],
        help="Audio files to classify.",
    )
    parser.add_argument(
        "--subset", type=str, default=None,
        help="FMA subset (tiny/small/medium) — used with --random.",
    )
    parser.add_argument(
        "--random", type=int, default=0, metavar="N",
        help="Pick N random samples from the test manifest of --subset.",
    )
    parser.add_argument(
        "--mode", choices=["three_crop", "single_crop"], default="three_crop",
        help="Inference mode: three_crop (default, more accurate) or single_crop (faster).",
    )
    args = parser.parse_args()

    if not args.files and args.random == 0:
        parser.error("Provide --files and/or --random N (with --subset).")
    if args.random > 0 and args.subset is None:
        parser.error("--random requires --subset (e.g. --subset small).")

    # ── Load the inference engine ─────────────────────────────────────────
    print(f"Loading model from: {args.run_dir}")
    engine = MelCNNInference(args.run_dir)
    print(f"  Genre classes ({engine.n_classes}): {engine.genre_classes}")
    print(f"  Audio backend: {engine._tf is not None and 'tf+' or ''}{MelCNNInference.__module__}")

    # ── Collect files to predict ──────────────────────────────────────────
    items: list[tuple[str, str]] = []   # (filepath, true_genre)

    for f in args.files:
        items.append((str(f), "?"))

    if args.random > 0:
        items.extend(_random_test_files(args.subset, args.random))

    # ── Run predictions ───────────────────────────────────────────────────
    correct = 0
    total   = 0
    for filepath, true_genre in items:
        try:
            result = engine.predict(filepath, mode=args.mode)
            _print_result(result, true_genre)
            if true_genre != "?":
                total += 1
                if result.genre == true_genre:
                    correct += 1
        except Exception as exc:
            print(f"\n[ERROR] {Path(filepath).name}: {exc}", file=sys.stderr)

    # ── Summary ───────────────────────────────────────────────────────────
    if total > 0:
        print(f"\n{'=' * 60}")
        print(f"  Accuracy on {total} samples: {correct}/{total} = {correct/total:.1%}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
