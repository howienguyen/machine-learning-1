"""Config-driven inference module for the Log-Mel CNN v2 family.

Supports run directories produced by:
  - MelCNN-MGR/model_training/logmel_cnn_v2.py
  - MelCNN-MGR/model_training/logmel_cnn_v2_1.py
  - MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py

Unlike the older v1.1 inference module, this loader reads feature settings from
the run artifacts so it can match the training-time log-mel shape.

Key behavior:
  - Loads best_model_macro_f1.keras by default.
  - Falls back to the final saved model, then any other .keras file.
  - Loads mu/std/genre_classes from norm_stats.npz.
  - Reads feature_config from run_report_*.json when available.
  - Validates the resolved log-mel input shape against model.input_shape.
  - Supports single-crop and three-crop inference.
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

DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_MELS = 192
DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 256


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
    sr: int,
    mono: bool = True,
    duration: float | None = None,
) -> np.ndarray:
    filepath = str(filepath)
    if AUDIO_BACKEND == "ffmpeg":
        return _load_audio_ffmpeg(filepath, sr=sr, mono=mono, duration=duration)
    y, _ = librosa.load(filepath, sr=sr, mono=mono, duration=duration)
    return y.astype(np.float32, copy=False)


def _normalize_to_fixed_duration(y: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
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


def _extract_three_crops(y: np.ndarray, sr: int, target_sec: float) -> list[np.ndarray]:
    target_len = int(round(target_sec * sr))
    n = len(y)

    if n <= target_len:
        padded = _normalize_to_fixed_duration(y, sr, target_sec)
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


class LogMelCNNV21Inference:
    """Load a trained v2-family model and run genre inference on audio files."""

    _MODEL_CANDIDATES = [
        "best_model_macro_f1.keras",
        "logmel_cnn_v2_1.keras",
        "logmel_cnn_v2_1_exp.keras",
        "logmel_cnn_v2.keras",
    ]

    def __init__(self, run_dir: str | Path, prefer_macro_f1: bool = True) -> None:
        import tensorflow as tf

        _register_cosine_schedule()

        self._tf = tf
        self.run_dir = Path(run_dir).expanduser().resolve()
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        model_candidates = self._MODEL_CANDIDATES if prefer_macro_f1 else list(reversed(self._MODEL_CANDIDATES))
        keras_files = sorted(self.run_dir.glob("*.keras"))
        fallbacks = [p.name for p in keras_files if p.name not in model_candidates]

        self.model_path: Path | None = None
        for name in model_candidates + fallbacks:
            path = self.run_dir / name
            if path.exists():
                self.model_path = path
                break

        if self.model_path is None:
            raise FileNotFoundError(
                f"No .keras model found in run directory: {self.run_dir}\n"
                "Re-run the training script to generate model checkpoints."
            )

        print(f"[LogMelCNNV21Inference] Loading model: {self.model_path.name}")
        self.model = tf.keras.models.load_model(str(self.model_path))

        stats_path = self.run_dir / "norm_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {stats_path}\n"
                "Re-run the training script to generate norm_stats.npz."
            )
        data = np.load(str(stats_path), allow_pickle=True)
        self.mu: np.ndarray = data["mu"]
        self.std: np.ndarray = data["std"]
        self.genre_classes: list[str] = [str(item) for item in data["genre_classes"].tolist()]
        self.n_classes: int = len(self.genre_classes)

        self.run_report_path, self.run_report = self._load_run_report()
        feature_config = self._resolve_feature_config()
        self.sample_rate = int(feature_config["sample_rate"])
        self.n_mels = int(feature_config["n_mels"])
        self.n_fft = int(feature_config["n_fft"])
        self.hop_length = int(feature_config["hop_length"])
        self.clip_duration = float(feature_config["clip_duration_sec"])
        self.logmel_shape = tuple(int(v) for v in feature_config["logmel_shape"])

        self._validate_feature_shape()

    def _load_run_report(self) -> tuple[Path | None, dict[str, object] | None]:
        report_candidates = sorted(self.run_dir.glob("run_report_*.json"))
        if not report_candidates:
            return None, None

        report_path = report_candidates[0]
        try:
            return report_path, json.loads(report_path.read_text())
        except Exception as exc:
            print(f"[WARN] Failed to read run report from {report_path}: {exc}", file=sys.stderr)
            return report_path, None

    def _resolve_feature_config(self) -> dict[str, object]:
        report_feature_config = None
        if isinstance(self.run_report, dict):
            candidate = self.run_report.get("feature_config")
            if isinstance(candidate, dict):
                report_feature_config = candidate

        model_input_shape = tuple(self.model.input_shape)
        if len(model_input_shape) != 4 or model_input_shape[-1] != 1:
            raise ValueError(
                "Unsupported model input shape for log-mel inference. "
                f"Expected (None, n_mels, n_frames, 1), got {model_input_shape}."
            )

        inferred_n_mels = int(model_input_shape[1])
        inferred_n_frames = int(model_input_shape[2])

        sample_rate = int(report_feature_config.get("sample_rate", DEFAULT_SAMPLE_RATE)) if report_feature_config else DEFAULT_SAMPLE_RATE
        n_mels = int(report_feature_config.get("n_mels", inferred_n_mels)) if report_feature_config else inferred_n_mels
        n_fft = int(report_feature_config.get("n_fft", DEFAULT_N_FFT)) if report_feature_config else DEFAULT_N_FFT
        hop_length = int(report_feature_config.get("hop_length", DEFAULT_HOP_LENGTH)) if report_feature_config else DEFAULT_HOP_LENGTH

        if report_feature_config and "clip_duration_sec" in report_feature_config:
            clip_duration_sec = float(report_feature_config["clip_duration_sec"])
        else:
            clip_duration_sec = float(inferred_n_frames * hop_length / sample_rate)

        if report_feature_config and "logmel_shape" in report_feature_config:
            logmel_shape = tuple(int(v) for v in report_feature_config["logmel_shape"])
        else:
            logmel_shape = (n_mels, inferred_n_frames)

        return {
            "sample_rate": sample_rate,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "clip_duration_sec": clip_duration_sec,
            "logmel_shape": logmel_shape,
        }

    def _validate_feature_shape(self) -> None:
        expected_input_shape = (None, *self.logmel_shape, 1)
        actual_input_shape = tuple(self.model.input_shape)
        if actual_input_shape != expected_input_shape:
            raise ValueError(
                "Inference feature shape does not match the loaded model input shape. "
                f"Expected {expected_input_shape}, got {actual_input_shape}."
            )

        if self.mu.shape != (1, self.n_mels, 1, 1):
            raise ValueError(
                f"mu has unexpected shape {self.mu.shape}; expected (1, {self.n_mels}, 1, 1)."
            )
        if self.std.shape != (1, self.n_mels, 1, 1):
            raise ValueError(
                f"std has unexpected shape {self.std.shape}; expected (1, {self.n_mels}, 1, 1)."
            )

    def extract_logmel(self, y: np.ndarray) -> np.ndarray:
        """Compute a fixed-shape log-mel spectrogram that matches the trained model."""
        spectrogram = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        logmel = np.log1p(spectrogram)
        out = np.zeros(self.logmel_shape, dtype=np.float32)
        frame_count = min(logmel.shape[1], self.logmel_shape[1])
        out[:, :frame_count] = logmel[:, :frame_count].astype(np.float32, copy=False)
        return out

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
        sr: int | None = None,
        mode: str = "three_crop",
        source_name: str = "<waveform>",
    ) -> PredictionResult:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.size == 0:
            raise ValueError("Waveform is empty.")

        source_sr = int(sr or self.sample_rate)
        if source_sr != self.sample_rate:
            y = librosa.resample(y, orig_sr=source_sr, target_sr=self.sample_rate).astype(np.float32, copy=False)

        if mode == "single_crop":
            y_clip = _normalize_to_fixed_duration(y, self.sample_rate, self.clip_duration)
            logmel = self.extract_logmel(y_clip)
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

        crops = _extract_three_crops(y, self.sample_rate, self.clip_duration)
        crop_probs = [self._predict_logmel(self.extract_logmel(crop)) for crop in crops]
        avg_probs = np.mean(crop_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))

        crop_details = []
        for cp in crop_probs:
            crop_idx = int(np.argmax(cp))
            crop_details.append(
                CropDetail(
                    genre=self.genre_classes[crop_idx],
                    confidence=float(cp[crop_idx]),
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
        audio_path = Path(audio_path)
        y = load_audio(audio_path, sr=self.sample_rate, mono=True)
        return self.predict_waveform(y, sr=self.sample_rate, mode=mode, source_name=str(audio_path))

    def predict_batch(self, audio_paths: list[str | Path], mode: str = "three_crop") -> list[PredictionResult]:
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
        description="Run direct inference with a trained Log-Mel CNN v2 family model.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Training run directory containing .keras files and norm_stats.npz.",
    )
    parser.add_argument(
        "files",
        nargs="+",
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
        help="Prefer the final saved model over the best Macro-F1 checkpoint.",
    )
    args = parser.parse_args()

    engine = LogMelCNNV21Inference(args.run_dir, prefer_macro_f1=not args.final_model)
    print(f"Run dir       : {engine.run_dir}")
    print(f"Model file    : {engine.model_path.name}")
    print(f"Backend       : {AUDIO_BACKEND}")
    print(f"Classes       : {engine.n_classes}")
    print(f"Sample rate   : {engine.sample_rate}")
    print(f"Clip duration : {engine.clip_duration}s")
    print(f"Log-mel shape : {engine.logmel_shape}")

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