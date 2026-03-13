"""
MelCNN-MGR Inference Module — v20a1
=====================================
Reusable inference engine for Log-Mel CNN v20a1 genre classification.

Loads a trained model, normalization stats, and genre labels from a run
directory produced by ``baseline_logmel_cnn_v20a1.py``.  Supports both
single-crop (center 10 s) and 3-crop (early/middle/late) inference.

Key differences from v20a inference:
  - Loads ``best_model_macro_f1.keras`` by default (v20a1 saves best macro-F1
    checkpoint in addition to EarlyStopping-restored weights).
  - Falls back to ``best_model_val_loss.keras`` then any ``.keras`` file if the
    preferred checkpoint is absent.
  - Registers ``CosineAnnealingWithWarmup`` with Keras serialization so the
    custom LR schedule deserializes correctly when loading the saved model.
  - ``normalize_to_fixed_duration`` uses center-crop for inference (same as
    val/test preprocessing in training — random crop is train-only).

Run directory layout (produced by the training script)::

    logmel-cnn-v20a1-YYYYMMDD-HHMMSS/
        best_model_macro_f1.keras          # best macro-F1 checkpoint (preferred)
        best_model_val_loss.keras          # best val-loss checkpoint (fallback)
        norm_stats.npz                     # mu, std, genre_classes
        run_report_<subset>.json           # training report (optional)

Usage
-----
    from MelCNN_MGR.inference_logmel_v20a1 import MelCNNInference

    engine = MelCNNInference("MelCNN-MGR/models/logmel-cnn-v20a1-20260308-172507")
    result = engine.predict("path/to/song.mp3")
    print(result.genre, result.confidence)

    # Top-3 predictions
    for genre, prob in result.top_k(3):
        print(f"  {genre}: {prob:.3f}")

    # Batch prediction
    results = engine.predict_batch(["a.mp3", "b.mp3", "c.mp3"])
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np


# ── Audio backend detection ────────────────────────────────────────────────────
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
AUDIO_BACKEND = "ffmpeg" if FFMPEG_AVAILABLE else "librosa"

# ── Audio / spectrogram constants (must match training) ────────────────────────
SAMPLE_RATE   = 22_050
N_MELS        = 128
N_FFT         = 512
HOP_LENGTH    = 256
CLIP_DURATION = 10.0                                           # seconds
N_FRAMES      = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH)  # 861
LOGMEL_SHAPE  = (N_MELS, N_FRAMES)                             # (128, 861)


# ── Custom LR schedule — must be registered before load_model ─────────────────

def _register_cosine_schedule() -> None:
    """Register CosineAnnealingWithWarmup so Keras can deserialize saved models."""
    import tensorflow as tf
    import math

    @tf.keras.saving.register_keras_serializable(package="MelCNN")
    class CosineAnnealingWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
        """Linear warmup followed by cosine decay to min_lr (mirrors training script)."""

        def __init__(self, warmup_steps, total_steps, lr_max, lr_min):
            super().__init__()
            self.warmup_steps = float(warmup_steps)
            self.total_steps  = float(total_steps)
            self.lr_max       = lr_max
            self.lr_min       = lr_min

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
                "total_steps":  self.total_steps,
                "lr_max":       self.lr_max,
                "lr_min":       self.lr_min,
            }


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class CropDetail:
    """Per-crop prediction detail."""
    genre: str
    confidence: float
    probs: list[float]


@dataclass
class PredictionResult:
    """Result returned by ``MelCNNInference.predict()``."""
    file: str
    genre: str
    confidence: float
    probs: list[float]
    genre_classes: list[str]
    mode: str                                   # "single_crop" | "three_crop"
    crops: list[CropDetail] = field(default_factory=list)

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (genre, probability) pairs, sorted descending."""
        indexed = list(enumerate(self.probs))
        indexed.sort(key=lambda t: t[1], reverse=True)
        return [(self.genre_classes[i], p) for i, p in indexed[:k]]


# ── Audio helpers ──────────────────────────────────────────────────────────────

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
    """Load an audio file as float32 waveform, using ffmpeg if available."""
    filepath = str(filepath)
    if AUDIO_BACKEND == "ffmpeg":
        return _load_audio_ffmpeg(filepath, sr=sr, mono=mono, duration=duration)
    y, _ = librosa.load(filepath, sr=sr, mono=mono, duration=duration)
    return y.astype(np.float32, copy=False)


# ── Waveform / spectrogram helpers ─────────────────────────────────────────────

def normalize_to_fixed_duration(
    y: np.ndarray, sr: int, target_sec: float,
) -> np.ndarray:
    """Center-crop or center-pad *y* to exactly *target_sec* seconds.

    Inference always uses center-crop (deterministic).  Random crop is a
    train-time augmentation only; val/test and inference use the center window.
    """
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
    # n < target_len → center-pad with silence
    pad_total = target_len - n
    pad_left  = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32)


def extract_logmel(y: np.ndarray) -> np.ndarray:
    """Compute a fixed-shape ``(128, 861)`` log-mel spectrogram.

    Matches the ``_logmel_fixed_shape`` function in the training script:
    log1p-compressed mel spectrogram, zero-padded / truncated to N_FRAMES.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
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
    """Extract 3 deterministic 10-second waveform crops (early / middle / late).

    - clips ≤ target_sec: 3 copies of center-padded clip
    - clips between 1× and 3× target: start / center / end crops
    - clips ≥ 3× target: crops centered at 25 %, 50 %, 75 % of duration
    """
    target_len = int(round(target_sec * sr))
    n = len(y)

    if n <= target_len:
        padded = normalize_to_fixed_duration(y, sr, target_sec)
        return [padded, padded, padded]

    def _crop_at(center: int) -> np.ndarray:
        half  = target_len // 2
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


# ── Inference engine ───────────────────────────────────────────────────────────

class MelCNNInference:
    """Load a trained v20a1 model and run genre inference on audio files.

    Parameters
    ----------
    run_dir : str or Path
        Path to a training run directory as produced by
        ``baseline_logmel_cnn_v20a1.py`` (contains ``best_model_macro_f1.keras``
        and ``norm_stats.npz``).
    prefer_macro_f1 : bool
        When ``True`` (default) load ``best_model_macro_f1.keras`` which is
        optimized for balanced per-class performance.  Set to ``False`` to
        force loading the EarlyStopping-restored (best val-loss) weights.
    """

    # Ordered preference for model filenames
    _MODEL_CANDIDATES = [
        "best_model_macro_f1.keras",
        "best_model_val_loss.keras",
    ]

    def __init__(
        self,
        run_dir: str | Path,
        prefer_macro_f1: bool = True,
    ) -> None:
        import tensorflow as tf

        # Register custom LR schedule before any load_model call
        _register_cosine_schedule()

        self._tf = tf
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # ── Locate and load model ─────────────────────────────────────────
        candidates = self._MODEL_CANDIDATES if prefer_macro_f1 else list(reversed(self._MODEL_CANDIDATES))
        # Also accept any .keras file as last resort
        keras_files = sorted(run_dir.glob("*.keras"))
        fallbacks   = [p.name for p in keras_files if p.name not in candidates]

        model_path: Path | None = None
        for name in candidates + fallbacks:
            p = run_dir / name
            if p.exists():
                model_path = p
                break

        if model_path is None:
            raise FileNotFoundError(
                f"No .keras model found in run directory: {run_dir}\n"
                "Re-run the training script to generate model checkpoints."
            )

        print(f"[MelCNNInference] Loading model: {model_path.name}")
        self.model = tf.keras.models.load_model(str(model_path))
        self.model_path = model_path

        # ── Load normalization stats ──────────────────────────────────────
        stats_path = run_dir / "norm_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {stats_path}\n"
                "Re-run the training script to generate norm_stats.npz."
            )
        data = np.load(str(stats_path), allow_pickle=True)
        self.mu: np.ndarray            = data["mu"]             # (1, 128, 1, 1)
        self.std: np.ndarray           = data["std"]            # (1, 128, 1, 1)
        self.genre_classes: list[str]  = data["genre_classes"].tolist()
        self.n_classes: int            = len(self.genre_classes)

        # ── Load run report (optional, for metadata) ──────────────────────
        report_files = list(run_dir.glob("run_report_*.json"))
        self.run_report: dict | None = None
        if report_files:
            self.run_report = json.loads(report_files[0].read_text())

        self.run_dir = run_dir

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _normalize_logmel(self, logmel: np.ndarray) -> np.ndarray:
        """Z-normalize and reshape to model input ``(1, 128, 861, 1)``."""
        x = logmel[np.newaxis, ..., np.newaxis]           # (1, 128, 861, 1)
        x = ((x - self.mu) / self.std).astype(np.float32)
        return x

    def _predict_logmel(self, logmel: np.ndarray) -> np.ndarray:
        """Run model on a single normalized log-mel; return probability vector."""
        x = self._normalize_logmel(logmel)
        return self.model.predict(x, verbose=0)[0]

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        audio_path: str | Path,
        mode: str = "three_crop",
    ) -> PredictionResult:
        """Predict the genre of an audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to an audio file (MP3, WAV, FLAC, OGG, etc.).
        mode : str
            ``"three_crop"`` (default) — 3 deterministic crops averaged.
            ``"single_crop"`` — center-crop only (faster, slightly lower accuracy).

        Returns
        -------
        PredictionResult
        """
        audio_path = Path(audio_path)
        y = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)

        if mode == "single_crop":
            y_clip = normalize_to_fixed_duration(y, SAMPLE_RATE, CLIP_DURATION)
            logmel  = extract_logmel(y_clip)
            probs   = self._predict_logmel(logmel)
            pred_idx = int(np.argmax(probs))
            return PredictionResult(
                file=str(audio_path),
                genre=self.genre_classes[pred_idx],
                confidence=float(probs[pred_idx]),
                probs=[float(p) for p in probs],
                genre_classes=self.genre_classes,
                mode="single_crop",
            )

        # three_crop (default) ─────────────────────────────────────────────
        crops      = extract_three_crops(y, SAMPLE_RATE, CLIP_DURATION)
        crop_probs = [self._predict_logmel(extract_logmel(c)) for c in crops]
        avg_probs  = np.mean(crop_probs, axis=0)
        pred_idx   = int(np.argmax(avg_probs))

        crop_details = []
        for cp in crop_probs:
            ci = int(np.argmax(cp))
            crop_details.append(CropDetail(
                genre=self.genre_classes[ci],
                confidence=float(cp[ci]),
                probs=[float(p) for p in cp],
            ))

        return PredictionResult(
            file=str(audio_path),
            genre=self.genre_classes[pred_idx],
            confidence=float(avg_probs[pred_idx]),
            probs=[float(p) for p in avg_probs],
            genre_classes=self.genre_classes,
            mode="three_crop",
            crops=crop_details,
        )

    def predict_batch(
        self,
        audio_paths: list[str | Path],
        mode: str = "three_crop",
    ) -> list[PredictionResult]:
        """Predict genres for multiple audio files.

        Files that fail to load or decode are skipped with a warning printed to
        stderr; all other results are returned in input order.
        """
        import sys

        results = []
        for path in audio_paths:
            try:
                results.append(self.predict(path, mode=mode))
            except Exception as exc:
                print(f"[WARN] Skipped {path}: {exc}", file=sys.stderr)
        return results
