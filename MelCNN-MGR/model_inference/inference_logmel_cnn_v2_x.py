"""Config-driven inference module for the Log-Mel CNN v2.x family.

Supports run directories produced by:
  - MelCNN-MGR/model_training/logmel_cnn_v2.py
  - MelCNN-MGR/model_training/logmel_cnn_v2_1.py
  - MelCNN-MGR/model_training/logmel_cnn_v2_1_exp.py
    - MelCNN-MGR/model_training/logmel_cnn_v2_2.py

Unlike the older v1.1 inference module, this loader reads feature settings from
the run artifacts so it can match the training-time log-mel shape.

Key behavior:
  - Loads best_model_macro_f1.keras by default.
  - Falls back to the final saved model, then any other .keras file.
    - Loads normalization stats from norm_stats.npz, preferring explicit
        mean_per_bin/std_per_bin when available and falling back to legacy mu/std.
  - Reads feature_config from run_report_*.json when available.
  - Validates the resolved log-mel input shape against model.input_shape.
  - Supports single-crop and three-crop inference.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import sys
import subprocess
import tempfile
import zipfile
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


def _install_keras_module_aliases() -> None:
    """Bridge internal Keras module path changes across runtime versions.

    Some `.keras` archives saved from newer Keras builds serialize the
    Functional model class as `keras.src.models.functional.Functional`, while
    older TF/Keras runtimes still expose that implementation from
    `keras.src.engine.functional`.
    """

    module_aliases = {
        "keras.src.models.functional": "keras.src.engine.functional",
    }

    for missing_name, target_name in module_aliases.items():
        try:
            importlib.import_module(missing_name)
            continue
        except ModuleNotFoundError:
            pass

        try:
            target_module = importlib.import_module(target_name)
        except ModuleNotFoundError:
            continue

        sys.modules[missing_name] = target_module

        parent_name, child_name = missing_name.rsplit(".", 1)
        try:
            parent_module = importlib.import_module(parent_name)
            setattr(parent_module, child_name, target_module)
        except Exception:
            pass


def _rewrite_keras_config_for_legacy_runtime(config: dict[str, object]) -> bool:
    """Patch newer Keras serialization fields for older TF/Keras runtimes."""

    changed = False

    def _normalize_dtype_policy(node: object) -> None:
        nonlocal changed

        if isinstance(node, dict):
            dtype_value = node.get("dtype")
            if (
                isinstance(dtype_value, dict)
                and dtype_value.get("class_name") == "DTypePolicy"
                and isinstance(dtype_value.get("config"), dict)
                and isinstance(dtype_value["config"].get("name"), str)
            ):
                node["dtype"] = dtype_value["config"]["name"]
                changed = True

            for value in node.values():
                _normalize_dtype_policy(value)
        elif isinstance(node, list):
            for item in node:
                _normalize_dtype_policy(item)

    _normalize_dtype_policy(config)

    if config.get("module") == "keras.src.models.functional":
        config["module"] = "keras.src.engine.functional"
        changed = True

    model_config = config.get("config")
    if not isinstance(model_config, dict):
        return changed

    layers = model_config.get("layers")
    if not isinstance(layers, list):
        return changed

    for layer in layers:
        if not isinstance(layer, dict):
            continue

        layer_config = layer.get("config")
        if not isinstance(layer_config, dict):
            continue

        class_name = layer.get("class_name")
        if class_name == "InputLayer":
            batch_shape = layer_config.pop("batch_shape", None)
            if batch_shape is not None and "batch_input_shape" not in layer_config:
                layer_config["batch_input_shape"] = batch_shape
                changed = True
            if layer_config.pop("optional", None) is not None:
                changed = True
        elif class_name == "Dense":
            if "quantization_config" in layer_config:
                layer_config.pop("quantization_config", None)
                changed = True

    return changed


def _build_legacy_compatible_archive(model_path: Path) -> Path | None:
    """Create a temporary `.keras` archive with config rewrites if needed."""

    with zipfile.ZipFile(model_path) as src:
        config = json.loads(src.read("config.json"))
        if not _rewrite_keras_config_for_legacy_runtime(config):
            return None

        tmp_dir = Path(tempfile.mkdtemp(prefix="melcnn-keras-compat-"))
        compat_path = tmp_dir / model_path.name
        with zipfile.ZipFile(compat_path, "w") as dst:
            for info in src.infolist():
                if info.filename == "config.json":
                    dst.writestr(info, json.dumps(config))
                else:
                    dst.writestr(info, src.read(info.filename))
        return compat_path


def _load_v2x_model_from_archive(tf_module, model_path: Path):
    """Rebuild the simple v2.x CNN from archive JSON and load its weights."""

    with zipfile.ZipFile(model_path) as src:
        config = json.loads(src.read("config.json"))
        _rewrite_keras_config_for_legacy_runtime(config)

        model_config = config.get("config")
        if not isinstance(model_config, dict):
            raise ValueError("Unsupported Keras archive: missing top-level model config.")

        layers_config = model_config.get("layers")
        if not isinstance(layers_config, list) or not layers_config:
            raise ValueError("Unsupported Keras archive: missing layer config list.")

        input_entry = layers_config[0]
        if input_entry.get("class_name") != "InputLayer":
            raise ValueError("Unsupported v2.x archive: first layer is not an InputLayer.")

        input_config = input_entry.get("config", {})
        batch_input_shape = input_config.get("batch_input_shape") or input_config.get("batch_shape")
        if not isinstance(batch_input_shape, list) or len(batch_input_shape) != 4:
            raise ValueError(
                "Unsupported v2.x archive: expected InputLayer batch_input_shape with rank 4."
            )

        keras = tf_module.keras
        layers = keras.layers
        inputs = keras.Input(
            shape=tuple(batch_input_shape[1:]),
            name=input_config.get("name", "logmel"),
            dtype=input_config.get("dtype", "float32"),
            sparse=bool(input_config.get("sparse", False)),
            ragged=bool(input_config.get("ragged", False)),
        )

        x = inputs
        supported_layer_types = {
            "Conv2D": layers.Conv2D,
            "BatchNormalization": layers.BatchNormalization,
            "ReLU": layers.ReLU,
            "MaxPooling2D": layers.MaxPooling2D,
            "SpatialDropout2D": layers.SpatialDropout2D,
            "GlobalAveragePooling2D": layers.GlobalAveragePooling2D,
            "Dense": layers.Dense,
            "Dropout": layers.Dropout,
        }

        for layer_entry in layers_config[1:]:
            class_name = layer_entry.get("class_name")
            if class_name not in supported_layer_types:
                raise ValueError(f"Unsupported v2.x layer type in archive: {class_name}")

            layer_kwargs = dict(layer_entry.get("config", {}))
            layer_kwargs.pop("quantization_config", None)
            layer_cls = supported_layer_types[class_name]
            x = layer_cls(**layer_kwargs)(x)

        model = keras.Model(inputs, x, name=model_config.get("name", "logmel_cnn_v2_x"))

        weights_member = "model.weights.h5"
        if weights_member not in src.namelist():
            raise FileNotFoundError(f"Unsupported Keras archive: missing {weights_member}.")

        with tempfile.TemporaryDirectory(prefix="melcnn-keras-weights-") as tmp_dir:
            weights_path = Path(tmp_dir) / weights_member
            weights_path.write_bytes(src.read(weights_member))
            model.load_weights(str(weights_path))

    return model


def _register_cosine_schedule() -> None:
    """Register the custom LR schedule used during training."""
    import tensorflow as tf

    register_keras_serializable = getattr(
        getattr(tf.keras, "saving", None),
        "register_keras_serializable",
        tf.keras.utils.register_keras_serializable,
    )

    @register_keras_serializable(package="MelCNN")
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


def _load_keras_model(tf_module, model_path: Path):
    """Load a `.keras` model for inference with cross-version compatibility."""

    _install_keras_module_aliases()
    try:
        return tf_module.keras.models.load_model(str(model_path), compile=False)
    except (AttributeError, TypeError, ValueError, ModuleNotFoundError) as exc:
        compat_exc = None
        compat_path = _build_legacy_compatible_archive(model_path)
        if compat_path is not None:
            try:
                return tf_module.keras.models.load_model(str(compat_path), compile=False)
            except (AttributeError, TypeError, ValueError, ModuleNotFoundError) as retry_exc:
                compat_exc = retry_exc

        try:
            return _load_v2x_model_from_archive(tf_module, model_path)
        except Exception as rebuild_exc:
            detail = compat_exc or exc
            raise RuntimeError(
                "Failed to load Keras model for inference. "
                "The archive appears to have been saved by a newer Keras runtime, and both direct "
                "deserialization and manual v2.x reconstruction failed. "
                f"Direct load error: {detail}. Manual rebuild error: {rebuild_exc}"
            ) from rebuild_exc

        raise RuntimeError(
            "Failed to load Keras model for inference. "
            f"This usually means the model was saved with an incompatible Keras runtime: {exc}"
        ) from exc


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


class LogMelCNNV2XInference:
    """Load a trained v2.x-family model and run genre inference on audio files."""

    _MODEL_CANDIDATES = [
        "best_model_macro_f1.keras",
        "logmel_cnn_v2_2.keras",
        "logmel_cnn_v2_1.keras",
        "logmel_cnn_v2_1_exp.keras",
        "logmel_cnn_v2.keras",
    ]

    def __init__(self, model_dir: str | Path, prefer_macro_f1: bool = True) -> None:
        import tensorflow as tf

        _register_cosine_schedule()

        self._tf = tf
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        model_candidates = self._MODEL_CANDIDATES if prefer_macro_f1 else list(reversed(self._MODEL_CANDIDATES))
        keras_files = sorted(self.model_dir.glob("*.keras"))
        fallbacks = [p.name for p in keras_files if p.name not in model_candidates]

        self.model_path: Path | None = None
        for name in model_candidates + fallbacks:
            path = self.model_dir / name
            if path.exists():
                self.model_path = path
                break

        if self.model_path is None:
            raise FileNotFoundError(
                f"No .keras model found in model directory: {self.model_dir}\n"
                "Re-run the training script to generate model checkpoints."
            )

        print(f"[LogMelCNNV2XInference] Loading model: {self.model_path.name}")
        self.model = _load_keras_model(tf, self.model_path)

        self.run_report_path, self.run_report = self._load_run_report()
        feature_config = self._resolve_feature_config()
        self.sample_rate = int(feature_config["sample_rate"])
        self.n_mels = int(feature_config["n_mels"])
        self.n_fft = int(feature_config["n_fft"])
        self.hop_length = int(feature_config["hop_length"])
        self.clip_duration = float(feature_config["clip_duration_sec"])
        self.logmel_shape = tuple(int(v) for v in feature_config["logmel_shape"])

        self.stats_path = self.model_dir / "norm_stats.npz"
        self.normalization_metadata_path = self.model_dir / "normalization_stats.json"
        self.normalization = self._load_normalization_stats()
        self.mu: np.ndarray = self.normalization["mu"]
        self.std: np.ndarray = self.normalization["std"]
        self.mean_per_bin: np.ndarray | None = self.normalization.get("mean_per_bin")
        self.std_per_bin: np.ndarray | None = self.normalization.get("std_per_bin")
        self.genre_classes: list[str] = self.normalization["genre_classes"]
        self.n_classes: int = len(self.genre_classes)

        self._validate_feature_shape()

    def _load_run_report(self) -> tuple[Path | None, dict[str, object] | None]:
        report_candidates = sorted(self.model_dir.glob("run_report_*.json"))
        if not report_candidates:
            return None, None

        report_path = report_candidates[0]
        try:
            return report_path, json.loads(report_path.read_text())
        except Exception as exc:
            print(f"[WARN] Failed to read run report from {report_path}: {exc}", file=sys.stderr)
            return report_path, None

    def _load_normalization_stats(self) -> dict[str, object]:
        if not self.stats_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found: {self.stats_path}\n"
                "Re-run the training script to generate norm_stats.npz."
            )

        data = np.load(str(self.stats_path), allow_pickle=True)
        genre_classes = [str(item) for item in data["genre_classes"].tolist()]

        normalization_info: dict[str, object] = {
            "type": "legacy_mu_std",
            "source": "norm_stats.npz",
            "stats_path": str(self.stats_path),
            "metadata_path": None,
            "computed_from": None,
            "epsilon": None,
        }

        if self.normalization_metadata_path.exists():
            try:
                metadata = json.loads(self.normalization_metadata_path.read_text())
            except Exception as exc:
                print(
                    f"[WARN] Failed to read normalization metadata from {self.normalization_metadata_path}: {exc}",
                    file=sys.stderr,
                )
            else:
                if isinstance(metadata, dict):
                    normalization_info.update(metadata)
                    normalization_info["metadata_path"] = str(self.normalization_metadata_path)

        if "mean_per_bin" in data and "std_per_bin" in data:
            mean_per_bin = data["mean_per_bin"].astype(np.float32)
            std_per_bin = data["std_per_bin"].astype(np.float32)
            mu = mean_per_bin.reshape((1, mean_per_bin.shape[0], 1, 1)).astype(np.float32)
            std = std_per_bin.reshape((1, std_per_bin.shape[0], 1, 1)).astype(np.float32)
            normalization_info.setdefault("type", "train_only_per_mel_bin_standardization")
            normalization_info["resolved_type"] = "per_mel_bin"
            normalization_info["mean_per_bin_shape"] = list(mean_per_bin.shape)
            normalization_info["std_per_bin_shape"] = list(std_per_bin.shape)
            normalization_info["broadcast_mu_shape"] = list(mu.shape)
            normalization_info["broadcast_std_shape"] = list(std.shape)
            return {
                "mu": mu,
                "std": std,
                "mean_per_bin": mean_per_bin,
                "std_per_bin": std_per_bin,
                "genre_classes": genre_classes,
                "info": normalization_info,
            }

        mu = data["mu"].astype(np.float32)
        std = data["std"].astype(np.float32)
        normalization_info["resolved_type"] = "legacy_mu_std"
        normalization_info["broadcast_mu_shape"] = list(mu.shape)
        normalization_info["broadcast_std_shape"] = list(std.shape)
        return {
            "mu": mu,
            "std": std,
            "mean_per_bin": None,
            "std_per_bin": None,
            "genre_classes": genre_classes,
            "info": normalization_info,
        }

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
        if self.mean_per_bin is not None and self.mean_per_bin.shape != (self.n_mels,):
            raise ValueError(
                f"mean_per_bin has unexpected shape {self.mean_per_bin.shape}; expected ({self.n_mels},)."
            )
        if self.std_per_bin is not None and self.std_per_bin.shape != (self.n_mels,):
            raise ValueError(
                f"std_per_bin has unexpected shape {self.std_per_bin.shape}; expected ({self.n_mels},)."
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

    @property
    def normalization_info(self) -> dict[str, object]:
        return dict(self.normalization["info"])

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


LogMelCNNV21Inference = LogMelCNNV2XInference


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

# For testing and direct CLI usage; 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run direct inference with a trained Log-Mel CNN v2.x family model.",
    )
    parser.add_argument(
        "--model-dir",
        "--run-dir",
        dest="model_dir",
        required=True,
        type=Path,
        help="Model directory containing .keras files and norm_stats.npz. The --run-dir alias is deprecated.",
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

    engine = LogMelCNNV2XInference(args.model_dir, prefer_macro_f1=not args.final_model)
    print(f"Model dir     : {engine.model_dir}")
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

    # Example usage:
    #   python inference_logmel_cnn_v2_x.py --model-dir /path/to/model_dir /path/to/audio1.wav /path/to/audio2.wav

    main()