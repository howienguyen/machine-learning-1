"""
Log-Mel CNN v1.1 Inference Module
=================================
Reusable inference engine for Log-Mel CNN v1.1 genre classification.

Loads a trained model, normalization stats, and genre labels from a run
directory produced by ``MelCNN-MGR/model_training/logmel_cnn_v1_1.py``.
Supports both single-crop (center 10 s) and 3-crop (early/middle/late)
inference.

Key behavior:
  - Loads ``best_model_macro_f1.keras`` by default.
  - Falls back to ``logmel_cnn_v1_1.keras`` then any other ``.keras`` file.
  - Registers ``CosineAnnealingWithWarmup`` so the saved model can be
    deserialized correctly.
  - Uses the same deterministic center-crop logic as validation/test time.

Run directory layout (produced by the training script)::

    logmel-cnn-v1_1-YYYYMMDD-HHMMSS/
        best_model_macro_f1.keras      # preferred checkpoint
        logmel_cnn_v1_1.keras          # final saved model
        norm_stats.npz                 # mu, std, genre_classes
        run_report_logmel_cnn_v1_1.json

Usage
-----
    from inference_logmel_cnn_v1_1 import LogMelCNNV11Inference

    engine = LogMelCNNV11Inference(
        "MelCNN-MGR/models/logmel-cnn-v1_1-20260311-120000"
    )
    result = engine.predict("path/to/song.mp3")
    print(result.genre, result.confidence)

Used by
-------
    - ``MelCNN-MGR/examples/inference_logmel_cnn_v1_1_example.py``
    - ``MelCNN-MGR/inference_web_service/app.py``
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
AUDIO_BACKEND = "ffmpeg" if FFMPEG_AVAILABLE else "librosa"

SAMPLE_RATE = 22050
N_MELS = 192
N_FFT = 512
HOP_LENGTH = 256
CLIP_DURATION = 10.0
N_FRAMES = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH)
LOGMEL_SHAPE = (N_MELS, N_FRAMES)


def _register_cosine_schedule() -> None:
    """Register the custom LR schedule used during training."""
    import tensorflow as tf

    @tf.keras.saving.register_keras_serializable(package="MelCNN")
    class CosineAnnealingWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, warmup_steps, total_steps, lr_max, lr_min):
            super().__init__()
            self.warmup_steps = float(warmup_steps)
            self.total_steps = float(total_steps)
            self.lr_max = lr_max
            self.lr_min = lr_min

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            warmup_lr = self.lr_min + (self.lr_max - self.lr_min) * (
                step / tf.maximum(self.warmup_steps, 1.0)
            )
            progress = (step - self.warmup_steps) / tf.maximum(
                self.total_steps - self.warmup_steps, 1.0
            )
            cosine_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0 + tf.cos(math.pi * progress)
            )
            return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

        def get_config(self):
            return {
                "warmup_steps": self.warmup_steps,
                "total_steps": self.total_steps,
                "lr_max": self.lr_max,
                "lr_min": self.lr_min,
            }


@dataclass
class CropDetail:
    genre: str
    confidence: float
    probs: list[float]


@dataclass
class PredictionResult:
    file: str
    genre: str
    confidence: float
    probs: list[float]
    genre_classes: list[str]
    mode: str
    crops: list[CropDetail] = field(default_factory=list)

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        indexed = list(enumerate(self.probs))
        indexed.sort(key=lambda item: item[1], reverse=True)
        return [(self.genre_classes[i], p) for i, p in indexed[:k]]


def _load_audio_ffmpeg(
    filepath: str,
    sr: int,
    mono: bool = True,
    duration: float | None = None,
) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", str(filepath), "-vn", "-sn", "-dn"]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ar", str(sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-f", "f32le", "pipe:1"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg decode failed: {err}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise RuntimeError("ffmpeg produced empty output")
    return y


def load_audio(
    filepath: str | Path,
    sr: int = SAMPLE_RATE,
    mono: bool = True,
    duration: float | None = None,
) -> np.ndarray:
    filepath = str(filepath)
    if AUDIO_BACKEND == "ffmpeg":
        return _load_audio_ffmpeg(filepath, sr=sr, mono=mono, duration=duration)
    y, _ = librosa.load(filepath, sr=sr, mono=mono, duration=duration)
    return y.astype(np.float32, copy=False)


def normalize_to_fixed_duration(y: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Center-crop or center-pad a waveform to the target duration."""
    target_len = int(round(target_sec * sr))
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        mid = n // 2
        half = target_len // 2
        start = max(0, mid - half)
        if start + target_len > n:
            start = n - target_len
        return y[start : start + target_len]

    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32)


def extract_logmel(y: np.ndarray) -> np.ndarray:
    """Compute a fixed-shape ``(192, 861)`` log-mel spectrogram."""
    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    logmel = np.log1p(S)
    out = np.zeros(LOGMEL_SHAPE, dtype=np.float32)
    n = min(logmel.shape[1], N_FRAMES)
    out[:, :n] = logmel[:, :n].astype(np.float32, copy=False)
    return out


def extract_three_crops(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    target_sec: float = CLIP_DURATION,
) -> list[np.ndarray]:
    """Extract 3 deterministic waveform crops: early / middle / late."""
    target_len = int(round(target_sec * sr))
    n = len(y)

    if n <= target_len:
        padded = normalize_to_fixed_duration(y, sr, target_sec)
        return [padded, padded, padded]

    def _crop_at(center: int) -> np.ndarray:
        half = target_len // 2
        start = center - half
        if start < 0:
            start = 0
        if start + target_len > n:
            start = n - target_len
        return y[start : start + target_len]

    if n >= 3 * target_len:
        return [_crop_at(n // 4), _crop_at(n // 2), _crop_at(3 * n // 4)]
    return [
        y[:target_len],
        _crop_at(n // 2),
        y[n - target_len :],
    ]


class LogMelCNNV11Inference:
    """Load a trained v1.1 model and run genre inference on audio files."""

    _MODEL_CANDIDATES = [
        "best_model_macro_f1.keras",
        "logmel_cnn_v1_1.keras",
    ]

    def __init__(self, run_dir: str | Path, prefer_macro_f1: bool = True) -> None:
        import tensorflow as tf

        _register_cosine_schedule()

        self._tf = tf
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        candidates = self._MODEL_CANDIDATES if prefer_macro_f1 else list(reversed(self._MODEL_CANDIDATES))
        keras_files = sorted(run_dir.glob("*.keras"))
        fallbacks = [p.name for p in keras_files if p.name not in candidates]

        model_path: Path | None = None
        for name in candidates + fallbacks:
            path = run_dir / name
            if path.exists():
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                f"No .keras model found in run directory: {run_dir}\n"
                "Re-run the training script to generate model checkpoints."
            )

        print(f"[LogMelCNNV11Inference] Loading model: {model_path.name}")
        self.model = tf.keras.models.load_model(str(model_path))
        self.model_path = model_path

        stats_path = run_dir / "norm_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {stats_path}\n"
                "Re-run the training script to generate norm_stats.npz."
            )
        data = np.load(str(stats_path), allow_pickle=True)
        self.mu: np.ndarray = data["mu"]
        self.std: np.ndarray = data["std"]
        self.genre_classes: list[str] = data["genre_classes"].tolist()
        self.n_classes: int = len(self.genre_classes)

        report_path = run_dir / "run_report_logmel_cnn_v1_1.json"
        self.run_report: dict | None = None
        if report_path.exists():
            self.run_report = json.loads(report_path.read_text())

        self.run_dir = run_dir

    def _normalize_logmel(self, logmel: np.ndarray) -> np.ndarray:
        x = logmel[np.newaxis, ..., np.newaxis]
        x = ((x - self.mu) / self.std).astype(np.float32)
        return x

    def _predict_logmel(self, logmel: np.ndarray) -> np.ndarray:
        x = self._normalize_logmel(logmel)
        return self.model.predict(x, verbose=0)[0]

    def predict_waveform(
        self,
        y: np.ndarray,
        sr: int = SAMPLE_RATE,
        mode: str = "three_crop",
        source_name: str = "<waveform>",
    ) -> PredictionResult:
        """Predict from an in-memory waveform.

        This is used by streaming or service layers that already have decoded PCM
        audio and do not want to round-trip through a temporary file.
        """
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            raise ValueError("Waveform is empty.")

        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32, copy=False)

        if mode == "single_crop":
            y_clip = normalize_to_fixed_duration(y, SAMPLE_RATE, CLIP_DURATION)
            logmel = extract_logmel(y_clip)
            probs = self._predict_logmel(logmel)
            pred_idx = int(np.argmax(probs))
            return PredictionResult(
                file=source_name,
                genre=self.genre_classes[pred_idx],
                confidence=float(probs[pred_idx]),
                probs=[float(p) for p in probs],
                genre_classes=self.genre_classes,
                mode="single_crop",
            )

        if mode != "three_crop":
            raise ValueError(f"Unsupported inference mode: {mode}")

        crops = extract_three_crops(y, SAMPLE_RATE, CLIP_DURATION)
        crop_probs = [self._predict_logmel(extract_logmel(crop)) for crop in crops]
        avg_probs = np.mean(crop_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))

        crop_details = []
        for cp in crop_probs:
            ci = int(np.argmax(cp))
            crop_details.append(
                CropDetail(
                    genre=self.genre_classes[ci],
                    confidence=float(cp[ci]),
                    probs=[float(p) for p in cp],
                )
            )

        return PredictionResult(
            file=source_name,
            genre=self.genre_classes[pred_idx],
            confidence=float(avg_probs[pred_idx]),
            probs=[float(p) for p in avg_probs],
            genre_classes=self.genre_classes,
            mode="three_crop",
            crops=crop_details,
        )

    def predict(self, audio_path: str | Path, mode: str = "three_crop") -> PredictionResult:
        """Predict the genre of an audio file using single- or three-crop inference."""
        audio_path = Path(audio_path)
        y = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)
        return self.predict_waveform(y, sr=SAMPLE_RATE, mode=mode, source_name=str(audio_path))

    def predict_batch(
        self,
        audio_paths: list[str | Path],
        mode: str = "three_crop",
    ) -> list[PredictionResult]:
        import sys

        results = []
        for path in audio_paths:
            try:
                results.append(self.predict(path, mode=mode))
            except Exception as exc:
                print(f"[WARN] Skipped {path}: {exc}", file=sys.stderr)
        return results


def _print_result(result: PredictionResult) -> None:
    print(f"\n{'-' * 60}")
    print(f"File      : {Path(result.file).name}")
    print(f"Predicted : {result.genre} ({result.confidence:.2%})")
    print(f"Mode      : {result.mode}")
    if result.crops:
        for idx, crop in enumerate(result.crops, start=1):
            print(f"  Crop {idx}: {crop.genre} ({crop.confidence:.2%})")
    print("Top-3     : ", end="")
    for genre, prob in result.top_k(3):
        print(f"{genre} ({prob:.1%})  ", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run direct smoke-test inference with a trained Log-Mel CNN v1.1 model.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        default=Path("MelCNN-MGR/models/logmel-cnn-demo"),
        type=Path,
        help="Training run directory containing .keras files and norm_stats.npz.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        default=[Path("audio_demo/Blues-Chris Stapleton-Tennessee Whiskey.mp3"),
                 Path("audio_demo/Metal-Black Sabbath-Black Sabbath.mp3"),
                 Path("audio_demo/Hip-Hop-Kendrick Lamar-HUMBLE.mp3")],
        type=Path,
        help="One or more audio files to classify.",
    )
    parser.add_argument(
        "--mode",
        choices=["single_crop", "three_crop"],
        default="three_crop",
        help="Inference mode to use for all files.",
    )
    parser.add_argument(
        "--final-model",
        action="store_true",
        help="Prefer the final saved model over the best macro-F1 checkpoint.",
    )
    args = parser.parse_args()

    engine = LogMelCNNV11Inference(args.run_dir, prefer_macro_f1=not args.final_model)
    print(f"Run dir     : {engine.run_dir}")
    print(f"Model file  : {engine.model_path.name}")
    print(f"Backend     : {AUDIO_BACKEND}")
    print(f"Classes     : {engine.n_classes}")

    failed = 0
    for audio_path in args.files:
        try:
            result = engine.predict(audio_path, mode=args.mode)
            _print_result(result)
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {audio_path}: {exc}", file=sys.stderr)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()