"""
Log-Mel CNN v20 Inference Module
==================================
Reusable inference engine for Log-Mel CNN v20 genre classification.

Loads a trained model, normalization stats, and genre labels from a run
directory produced by ``baseline_logmel_cnn_v20.ipynb``.  Supports both
single-crop (center 10 s) and 3-crop (early/middle/late) inference.

Run directory layout (produced by the training notebook)::

    logmel-cnn-v20-YYYYMMDD-HHMMSS/
        baseline_logmel_cnn_v20.keras   # trained Keras model
        norm_stats.npz                  # mu, std, genre_classes
        run_report_<subset>.json        # training report (optional)

Usage
-----
    from MelCNN_MGR.inference_logmel_v20 import LogMelV20Inference

    engine = LogMelV20Inference("MelCNN-MGR/models/logmel-cnn-v20-20260308-120000")
    result = engine.predict("path/to/song.mp3")
    print(result.genre, result.confidence)

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
LOGMEL_SHAPE  = (N_MELS, N_FRAMES)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class CropDetail:
    """Per-crop prediction detail."""
    genre: str
    confidence: float
    probs: list[float]


@dataclass
class PredictionResult:
    """Result returned by ``LogMelV20Inference.predict()``."""
    file: str
    genre: str
    confidence: float
    probs: list[float]
    genre_classes: list[str]
    mode: str                                  # "single_crop" | "three_crop"
    crops: list[CropDetail] = field(default_factory=list)

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k (genre, probability) pairs, sorted descending."""
        indexed = list(enumerate(self.probs))
        indexed.sort(key=lambda t: t[1], reverse=True)
        return [(self.genre_classes[i], p) for i, p in indexed[:k]]


# ── Audio helpers ──────────────────────────────────────────────────────────────

def _load_audio_ffmpeg(
    filepath: str, sr: int, mono: bool = True, duration: float | None = None,
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
    """Center-crop or center-pad *y* to exactly *target_sec* seconds."""
    target_len = int(round(target_sec * sr))
    n = len(y)
    if n == target_len:
        return y
    if n > target_len:
        mid = n // 2
        half = target_len // 2
        start = max(0, mid - half)
        end = start + target_len
        if end > n:
            end = n
            start = n - target_len
        return y[start:end]
    # pad
    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32)


def extract_logmel(y: np.ndarray) -> np.ndarray:
    """Compute a fixed-shape ``(128, 861)`` log-mel spectrogram."""
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    logmel = np.log1p(S)
    out = np.zeros(LOGMEL_SHAPE, dtype=np.float32)
    n = min(logmel.shape[1], N_FRAMES)
    out[:, :n] = logmel[:, :n].astype(np.float32, copy=False)
    return out


def extract_three_crops(
    y: np.ndarray, sr: int = SAMPLE_RATE, target_sec: float = CLIP_DURATION,
) -> list[np.ndarray]:
    """Extract 3 deterministic 10-second waveform crops (early / middle / late).

    - clips > 3 × target: crops centered at 25 %, 50 %, 75 % of duration
    - clips between 1× and 3× target: start / center / end
    - clips <= target: 3 copies of center-padded clip
    """
    target_len = int(round(target_sec * sr))
    n = len(y)

    if n <= target_len:
        padded = normalize_to_fixed_duration(y, sr, target_sec)
        return [padded, padded, padded]

    def _crop_at(center: int) -> np.ndarray:
        half = target_len // 2
        start = center - half
        end = start + target_len
        if start < 0:
            start, end = 0, target_len
        if end > n:
            end = n
            start = n - target_len
        return y[start:end]

    if n >= 3 * target_len:
        return [_crop_at(n // 4), _crop_at(n // 2), _crop_at(3 * n // 4)]
    return [
        y[:target_len],
        _crop_at(n // 2),
        y[n - target_len:],
    ]


# ── Inference engine ───────────────────────────────────────────────────────────

class LogMelV20Inference:
    """Load a trained v20 model and run genre inference on audio files.

    Parameters
    ----------
    run_dir : str or Path
        Path to a training run directory containing the ``.keras`` model and
        ``norm_stats.npz``.
    """

    def __init__(self, run_dir: str | Path) -> None:
        import tensorflow as tf

        self._tf = tf
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # ── Load model ────────────────────────────────────────────────────
        model_path = run_dir / "baseline_logmel_cnn_v20.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))

        # ── Load normalization stats ──────────────────────────────────────
        stats_path = run_dir / "norm_stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {stats_path}\n"
                "Re-run the training notebook to generate norm_stats.npz."
            )
        data = np.load(str(stats_path), allow_pickle=True)
        self.mu: np.ndarray = data["mu"]            # (1, 128, 1, 1)
        self.std: np.ndarray = data["std"]           # (1, 128, 1, 1)
        self.genre_classes: list[str] = data["genre_classes"].tolist()
        self.n_classes: int = len(self.genre_classes)

        # ── Load run report (optional, for metadata) ──────────────────────
        report_files = list(run_dir.glob("run_report_*.json"))
        self.run_report: dict | None = None
        if report_files:
            self.run_report = json.loads(report_files[0].read_text())

        self.run_dir = run_dir

    def _normalize_logmel(self, logmel: np.ndarray) -> np.ndarray:
        """Apply z-normalization and reshape for model input."""
        x = logmel[np.newaxis, ..., np.newaxis]          # (1, 128, 861, 1)
        x = ((x - self.mu) / self.std).astype(np.float32)
        return x

    def _predict_logmel(self, logmel: np.ndarray) -> np.ndarray:
        """Run model on a single normalized log-mel, return probability vector."""
        x = self._normalize_logmel(logmel)
        return self.model.predict(x, verbose=0)[0]

    def predict(
        self,
        audio_path: str | Path,
        mode: str = "single_crop",
    ) -> PredictionResult:
        """Predict the genre of an audio file.

        Parameters
        ----------
        audio_path : str or Path
            Path to an audio file (MP3, WAV, FLAC, etc.).
        mode : str
            ``"single_crop"`` (default) — center 10 s clip only.
            ``"three_crop"`` — 3 deterministic crops, averaged.

        Returns
        -------
        PredictionResult
        """
        audio_path = Path(audio_path)
        y = load_audio(audio_path, sr=SAMPLE_RATE, mono=True)

        if mode == "single_crop":
            y_clip = normalize_to_fixed_duration(y, SAMPLE_RATE, CLIP_DURATION)
            logmel = extract_logmel(y_clip)
            probs = self._predict_logmel(logmel)
            pred_idx = int(np.argmax(probs))
            return PredictionResult(
                file=str(audio_path),
                genre=self.genre_classes[pred_idx],
                confidence=float(probs[pred_idx]),
                probs=[float(p) for p in probs],
                genre_classes=self.genre_classes,
                mode="single_crop",
            )

        # three_crop
        crops = extract_three_crops(y, SAMPLE_RATE, CLIP_DURATION)
        crop_logmels = [extract_logmel(c) for c in crops]

        crop_probs = [self._predict_logmel(lm) for lm in crop_logmels]
        avg_probs = np.mean(crop_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))

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
        mode: str = "single_crop",
    ) -> list[PredictionResult]:
        """Predict genres for multiple audio files.

        Files that fail to load/decode are skipped with a warning printed to
        stderr.
        """
        import sys

        results = []
        for path in audio_paths:
            try:
                results.append(self.predict(path, mode=mode))
            except Exception as exc:
                print(f"[WARN] Skipped {path}: {exc}", file=sys.stderr)
        return results
