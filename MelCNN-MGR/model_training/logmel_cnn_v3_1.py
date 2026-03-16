# %% [markdown]
# # Log-Mel CNN v3.1
#
# True ResNet-style backbone upgrade of `logmel_cnn_v2_2.py`.
#
# Architecture implemented in this file:
# - **Input**: a single-channel log-mel spectrogram tensor with shape
#   `(*LOGMEL_SHAPE, 1)`.
# - **Stem**: `Conv2D(32, 7x7, stride=2, no bias) -> BatchNorm -> ReLU ->
#   MaxPool2D(3x3, stride=2, same)`.
# - **Residual backbone**: 4 stages with 2 **post-activation basic residual
#   blocks** per stage.
# - **Stage widths**: `32 -> 64 -> 128 -> 256` filters.
# - **Stage strides**:
#   - stage 1 keeps resolution (`stride=1`, `stride=1`)
#   - stages 2, 3, and 4 downsample in the first block (`stride=2`) and keep
#     resolution in the second block (`stride=1`)
# - **Residual block structure**: `3x3 conv -> BN -> ReLU -> 3x3 conv -> BN ->
#   shortcut add -> ReLU`.
# - **Shortcut rule**: identity shortcut when shape matches, otherwise a
#   projection shortcut `1x1 conv + BN` when stride or channel count changes.
# - **Spatial dropout placement**: `SpatialDropout2D` is applied only after the
#   output activation of the **first block of each stage**, using the staged
#   rates from the v2.x recipe; the second block of each stage has no spatial
#   dropout.
# - **Classifier head**: `GlobalAveragePooling2D -> Dense(256, ReLU) ->
#   Dropout(FINAL_DROPOUT_RATE) -> Dense(n_classes, softmax)`.
#
# Compared with v2.2, this is a real backbone replacement rather than a light
# patch: the old plain conv/pool stack is removed, downsampling follows a
# ResNet-style stem-plus-stage-transition pattern, and residual shortcuts become
# the main feature-transport mechanism while the classifier head and training
# pipeline remain broadly comparable for cleaner experiments.
#
# Keeps the v2.x regularization recipe:
# - **Reduced SpecAugment** (num_masks=1) — avoid overmasking when combined with Mixup
# - **Graduated spatial dropout** (0.05→0.20) — concentrate regularization in deeper layers
# - **Wider bottleneck** (256 units) — preserve full backbone feature space for the classifier
# - **Reduced weight decay** (1e-4) — lighter L2 when Mixup+dropout already active
# - **Gradient clipping** (clipnorm=1.0) — stabilize training with mixed class-weighted loss
# - **Pre-Mixup class weighting** — anchor-based sample weights preserve class-balance signal
# - **Longer early stopping patience** (20 epochs) — allow the cosine LR tail to refine
# - **Lower final dropout** (0.20) — avoid over-constraining the wider bottleneck
#
# Inherits all v2 features:
# - Macro-F1-driven checkpointing + EarlyStopping
# - Mixup augmentation (α=0.3) + SpecAugment
# - Adaptive class weights + label smoothing disabled with Mixup
# - Optional warm-start initialization + staged backbone freezing
# - Optimizer continuity across staged freeze/unfreeze training
#
# Train directly from prebuilt log-mel `.npy` features referenced by:
# - `logmel_manifest_train.parquet`
# - `logmel_manifest_val.parquet`
# - `logmel_manifest_test.parquet`
#
# Feature extraction should happen upstream via:
# `MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py`
#
# Preferred end-to-end run profile:
# python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py --mode stage1 --stage1a-sources fma --stage1b-sources fma
# python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py --mode stage1 --stage1a-sources additional --stage1b-sources additional
# python MelCNN-MGR/preprocessing/1_build_all_datasets_and_samples_v1_1.py --mode stage2
# python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py
# python MelCNN-MGR/model_training/logmel_cnn_v3_1.py
#
# Warm-start usage example:
# python MelCNN-MGR/model_training/logmel_cnn_v3_1.py --pretrained-model MelCNN-MGR/models/logmel-cnn-demo/best_model_macro_f1.keras
# python MelCNN-MGR/model_training/logmel_cnn_v3_1.py --pretrained-model MelCNN-MGR/models/logmel-cnn-demo/best_model_macro_f1.keras --pretrained-mode backbone_only --freeze-backbone-epochs 3

# %% [markdown]
# ## 1. Imports

# %%
import argparse
import atexit
import json
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace


def _resolve_early_device_backend(argv: list[str]) -> str:
    cli_backend = None
    for index, token in enumerate(argv):
        if token == "--device-backend" and index + 1 < len(argv):
            cli_backend = argv[index + 1].strip().lower()
            break
        if token.startswith("--device-backend="):
            cli_backend = token.split("=", 1)[1].strip().lower()
            break

    if cli_backend:
        return cli_backend
    return os.environ.get("LOGMEL_CNN_DEVICE_BACKEND", "auto").strip().lower()


EARLY_DEVICE_BACKEND = _resolve_early_device_backend(sys.argv[1:])

os.environ["ONEDNN_VERBOSE"] = "none"
os.environ["DNNL_VERBOSE"] = "0"
if EARLY_DEVICE_BACKEND in {"xpu", "cpu"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


warnings.filterwarnings("ignore", category=UserWarning)


_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
_CONSOLE_CAPTURE_FILE = None
_CONSOLE_CAPTURE_PATH = None


class _StreamTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


def _stop_console_capture() -> None:
    global _CONSOLE_CAPTURE_FILE, _CONSOLE_CAPTURE_PATH
    if _CONSOLE_CAPTURE_FILE is None:
        return
    sys.stdout = _ORIGINAL_STDOUT
    sys.stderr = _ORIGINAL_STDERR
    try:
        _CONSOLE_CAPTURE_FILE.flush()
    finally:
        _CONSOLE_CAPTURE_FILE.close()
    _CONSOLE_CAPTURE_FILE = None
    _CONSOLE_CAPTURE_PATH = None


def _start_console_capture(log_path: Path) -> None:
    global _CONSOLE_CAPTURE_FILE, _CONSOLE_CAPTURE_PATH
    resolved = str(log_path.resolve())
    if _CONSOLE_CAPTURE_PATH == resolved:
        return
    _stop_console_capture()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _StreamTee(_ORIGINAL_STDOUT, log_file)
    sys.stderr = _StreamTee(_ORIGINAL_STDERR, log_file)
    _CONSOLE_CAPTURE_FILE = log_file
    _CONSOLE_CAPTURE_PATH = resolved


atexit.register(_stop_console_capture)

print(f"Python     : {platform.python_version()}")
print(f"TensorFlow : {tf.__version__}")


CLI_PARSER = argparse.ArgumentParser(
    description="Train Log-Mel CNN v3.1 with a true ResNet-style backbone, gap-aware early stopping, and optional warm-start from an existing .keras checkpoint.",
)
CLI_PARSER.add_argument(
    "--pretrained-model",
    type=str,
    default=None,
    help="Optional path to an existing .keras checkpoint used to initialize the new v2 run.",
)
CLI_PARSER.add_argument(
    "--pretrained-mode",
    choices=["full_model", "backbone_only"],
    default=None,
    help="Warm-start strategy: load the full compatible model or transfer only backbone weights into a fresh classifier head.",
)
CLI_PARSER.add_argument(
    "--freeze-backbone-epochs",
    type=int,
    default=None,
    help="Freeze transferred backbone layers for the first N epochs after backbone_only initialization.",
)
CLI_PARSER.add_argument(
    "--device-backend",
    choices=["auto", "xpu", "cuda", "cpu"],
    default=None,
    help="Preferred runtime backend. Defaults to LOGMEL_CNN_DEVICE_BACKEND or auto-detection when unset.",
)
CLI_ARGS, _CLI_UNKNOWN = CLI_PARSER.parse_known_args()


# %% [markdown]
# ## 2. Configuration

# %%
NOTEBOOK_DIR = Path(__file__).resolve().parent
MELCNN_DIR = NOTEBOOK_DIR.parent
WORKSPACE = MELCNN_DIR.parent

PROCESSED_DIR = MELCNN_DIR / "data" / "processed"
CACHE_DIR = MELCNN_DIR / "cache"
MODELS_BASE_DIR = MELCNN_DIR / "models"
SETTINGS_PATH = MELCNN_DIR / "settings.json"
DEFAULT_SAMPLE_LENGTH_SEC = 15.0


def load_default_sample_length_from_settings(settings_path: Path) -> float:
    try:
        payload = json.loads(settings_path.read_text())
    except Exception:
        return DEFAULT_SAMPLE_LENGTH_SEC

    config = payload.get("data_sampling_settings")
    if not isinstance(config, dict):
        return DEFAULT_SAMPLE_LENGTH_SEC

    sample_length_sec = config.get("sample_length_sec")
    if not isinstance(sample_length_sec, (int, float)) or sample_length_sec <= 0:
        return DEFAULT_SAMPLE_LENGTH_SEC

    return float(sample_length_sec)


DEFAULT_LOGMEL_SAMPLE_LENGTH_SEC = load_default_sample_length_from_settings(SETTINGS_PATH)

# Default cache root follows settings.json sample length when available.
LOGMEL_DATASET_DIR = Path(
    os.environ.get(
        "LOGMEL_DATASET_DIR",
        str(CACHE_DIR / f"logmel_dataset_{DEFAULT_LOGMEL_SAMPLE_LENGTH_SEC:g}s"),
    )
)
TRAIN_MANIFEST_PATH = LOGMEL_DATASET_DIR / "logmel_manifest_train.parquet"
VAL_MANIFEST_PATH = LOGMEL_DATASET_DIR / "logmel_manifest_val.parquet"
TEST_MANIFEST_PATH = LOGMEL_DATASET_DIR / "logmel_manifest_test.parquet"
LOGMEL_CONFIG_PATH = LOGMEL_DATASET_DIR / "logmel_config.json"

RUN_DIR = None
RUN_FAMILY = "logmel-cnn-v3_1"
_run_ts = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = MODELS_BASE_DIR / f"{RUN_FAMILY}-{_run_ts}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CONSOLE_LOG_PATH = RUN_DIR / "console_output.txt"
_start_console_capture(CONSOLE_LOG_PATH)
print(f"Run directory : {RUN_DIR}")
print(f"Console log   : {CONSOLE_LOG_PATH}")

EPOCHS = 136
BATCH_SIZE = 48
LABEL_SMOOTHING = 0.0            # Mixup already softens labels; keep CE targets sharp otherwise
WEIGHT_DECAY = 1e-4                # L2 Regularization - v2.1: 1e-4 (was v2: 5e-4) — lighter with Mixup+dropout active

SPEC_AUG_FREQ_MASK = 24            # inherited mask size; v2.1 reduces total masking via num_masks=1
SPEC_AUG_TIME_MASK = 40            # inherited mask size; v2.1 reduces total masking via num_masks=1
SPEC_AUG_NUM_MASKS = 1             # v2.1: 1 (was v2: 2) — avoid overmasking with Mixup

MAX_CLASS_WEIGHT = 1.5             # cap class weights to prevent over-prediction
MIXUP_ALPHA = 0.3                  # keep the v2 Mixup blending parameter unchanged
CLASS_WEIGHT_IMBALANCE_THRESHOLD = 1.05
SPATIAL_DROPOUT_RATE_BLOCK2 = 0.05  # v2.1: graduated — light in early layers
SPATIAL_DROPOUT_RATE_BLOCK3 = 0.10
SPATIAL_DROPOUT_RATE_BLOCK4 = 0.15  # v2.1: graduated — stronger in deep layers
SPATIAL_DROPOUT_RATE_BLOCK5 = 0.20  # v2.1: graduated — strongest in deepest layer
FINAL_DROPOUT_RATE = 0.20           # v2.1: 0.20 (was v2: 0.25) — wider bottleneck absorbs capacity

STANDARD_EARLY_STOP_PATIENCE = 9
STANDARD_EARLY_STOP_MIN_DELTA = 0.002
STANDARD_EARLY_STOP_START_EPOCH = 66
STANDARD_EARLY_STOP_RESTORE_BEST_WEIGHTS = True

GAP_EARLY_STOP_THRESHOLD = 0.1
GAP_EARLY_STOP_PATIENCE = 9
GAP_EARLY_STOP_MIN_DELTA = 0.001
GAP_EARLY_STOP_REQUIRE_VAL_NOT_IMPROVING = True
GAP_EARLY_STOP_VAL_MACRO_F1_MIN_THRESHOLD = 0.80
GAP_EARLY_STOP_START_EPOCH = 60
GAP_EARLY_STOP_RESTORE_BEST_WEIGHTS = True

PRETRAINED_MODEL_ENV = os.environ.get("LOGMEL_CNN_PRETRAINED_MODEL", "").strip()
PRETRAINED_MODEL_RAW = (CLI_ARGS.pretrained_model or PRETRAINED_MODEL_ENV).strip()
PRETRAINED_MODEL_PATH = Path(PRETRAINED_MODEL_RAW).expanduser().resolve() if PRETRAINED_MODEL_RAW else None
PRETRAINED_MODE = (CLI_ARGS.pretrained_mode or os.environ.get("LOGMEL_CNN_PRETRAINED_MODE", "full_model")).strip().lower()
FREEZE_BACKBONE_EPOCHS = max(
    0,
    int(
        CLI_ARGS.freeze_backbone_epochs
        if CLI_ARGS.freeze_backbone_epochs is not None
        else os.environ.get("LOGMEL_CNN_FREEZE_BACKBONE_EPOCHS", "0")
    ),
)
DEVICE_BACKEND = (
    CLI_ARGS.device_backend
    or os.environ.get("LOGMEL_CNN_DEVICE_BACKEND", "auto")
).strip().lower()
if PRETRAINED_MODEL_PATH is not None:
    print(f"[WarmStart] LOGMEL_CNN_PRETRAINED_MODEL={PRETRAINED_MODEL_PATH}")
    print(f"[WarmStart] PRETRAINED_MODE={PRETRAINED_MODE}")
if FREEZE_BACKBONE_EPOCHS > 0:
    print(f"[WarmStart] FREEZE_BACKBONE_EPOCHS={FREEZE_BACKBONE_EPOCHS}")
print(f"[Runtime] DEVICE_BACKEND={DEVICE_BACKEND}")

SEED = 36
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"[Seed] SEED={SEED} applied to random / numpy / tensorflow")

_T0 = time.perf_counter()
_section_times = {}


# %% [markdown]
# ## 3. Device Runtime Selection

# %%
_t0 = time.perf_counter()


def _best_effort_set_memory_growth(tf_module, device_type: str):
    devices = tf_module.config.list_physical_devices(device_type)
    for device in devices:
        try:
            tf_module.config.experimental.set_memory_growth(device, True)
        except Exception:
            pass
    return devices


def _hide_device_type(tf_module, device_type: str) -> None:
    try:
        visible_devices = tf_module.config.get_visible_devices()
        keep_devices = [device for device in visible_devices if device.device_type != device_type]
        tf_module.config.set_visible_devices(keep_devices)
        print(f"Hid visible {device_type} devices to honor backend preference")
    except Exception as exc:
        print(f"[WARN] Failed to hide {device_type} devices: {exc}")


def _smoke_test_matmul(tf_module, device: str, n: int = 1024) -> tuple[bool, str]:
    try:
        with tf.device(device):
            a = tf_module.random.normal([n, n])
            b = tf_module.random.normal([n, n])
            c = tf_module.matmul(a, b)
            _ = c[0, 0].numpy()
        return True, "ok"
    except Exception as exc:
        return False, repr(exc)


def _try_cuda_runtime(tf_module):
    try:
        gpus = _best_effort_set_memory_growth(tf_module, "GPU")
    except Exception:
        gpus = []
    if not gpus:
        return None

    ok, info = _smoke_test_matmul(tf_module, "/GPU:0")
    if ok:
        return "/GPU:0", "cuda", [d.name for d in gpus], info
    print("CUDA present but failed smoke test ->", info)
    return None


def _try_xpu_runtime(tf_module):
    try:
        import intel_extension_for_tensorflow as itex  # noqa: F401

        xpus = _best_effort_set_memory_growth(tf_module, "XPU")
    except Exception as exc:
        print("ITEX/XPU not available:", repr(exc))
        return None

    if not xpus:
        return None

    ok, info = _smoke_test_matmul(tf_module, "/XPU:0")
    if ok:
        return "/XPU:0", "xpu", [d.name for d in xpus], info
    print("XPU present but failed smoke test ->", info)
    return None


def configure_runtime_device(tf_module, preferred_backend: str = "auto"):
    print(f"Platform   : {platform.platform()}")
    print(f"TensorFlow : {tf_module.__version__}")

    if preferred_backend == "xpu":
        _hide_device_type(tf_module, "GPU")
        runtime = _try_xpu_runtime(tf_module)
        if runtime is not None:
            return runtime
        print("Preferred backend XPU unavailable -> CPU fallback")
        return "/CPU:0", "cpu", [], "ok"

    if preferred_backend == "cuda":
        runtime = _try_cuda_runtime(tf_module)
        if runtime is not None:
            return runtime
        print("Preferred backend CUDA unavailable -> CPU fallback")
        return "/CPU:0", "cpu", [], "ok"

    if preferred_backend == "cpu":
        _hide_device_type(tf_module, "GPU")
        _hide_device_type(tf_module, "XPU")
        return "/CPU:0", "cpu", [], "ok"

    runtime = _try_cuda_runtime(tf_module)
    if runtime is not None:
        return runtime

    runtime = _try_xpu_runtime(tf_module)
    if runtime is not None:
        return runtime

    return "/CPU:0", "cpu", [], "ok"


RUNTIME_DEVICE, BACKEND, ACCEL_NAMES, SMOKE_INFO = configure_runtime_device(tf, DEVICE_BACKEND)
print(f"Backend    : {BACKEND.upper()} ({RUNTIME_DEVICE})")
print(f"Devices    : {ACCEL_NAMES if ACCEL_NAMES else 'none detected -> CPU fallback'}")
print(f"Smoke test : {SMOKE_INFO}")

_section_times["3. Device setup"] = time.perf_counter() - _t0
print(f"\nDevice setup : {_section_times['3. Device setup']:.2f}s")


# %% [markdown]
# ## 4. Load Prebuilt Log-Mel Dataset Manifests

# %%
_t0 = time.perf_counter()


def _load_logmel_manifest(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        msg_lines = [
            f"Log-mel manifest not found: {path}",
            "Run the upstream builder first:",
            "  python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py",
        ]
        raise FileNotFoundError(chr(10).join(msg_lines))

    df = pd.read_parquet(path)
    if "logmel_usable" in df.columns:
        df = df[df["logmel_usable"] == True].copy()
    if "logmel_path" not in df.columns:
        raise ValueError(f"Missing 'logmel_path' column in {path}")
    if df.empty:
        print(f"[WARN] {split_name} manifest is empty: {path}")
    return df.reset_index(drop=True)


def load_logmel_splits(train_path: Path, val_path: Path, test_path: Path):
    train_df = _load_logmel_manifest(train_path, "train")
    val_df = _load_logmel_manifest(val_path, "val")
    test_df = _load_logmel_manifest(test_path, "test")
    return train_df, val_df, test_df


def resolve_logmel_config(train_df: pd.DataFrame) -> dict[str, object]:
    if LOGMEL_CONFIG_PATH.exists():
        config = json.loads(LOGMEL_CONFIG_PATH.read_text())
    else:
        config = {}

    sample_rate = int(config.get("sample_rate") or train_df["sample_rate"].dropna().iloc[0])
    n_mels = int(config.get("n_mels") or train_df["n_mels"].dropna().iloc[0])
    n_fft = int(config.get("n_fft") or train_df["n_fft"].dropna().iloc[0])
    hop_length = int(config.get("hop_length") or train_df["hop_length"].dropna().iloc[0])

    sample_lengths = sorted({round(float(v), 6) for v in train_df["sample_length_sec"].dropna().tolist()})
    if len(sample_lengths) != 1:
        raise ValueError(f"Expected a single sample_length_sec value, found {sample_lengths}")
    clip_duration = float(sample_lengths[0])
    n_frames = int(clip_duration * sample_rate / hop_length)
    return {
        "sample_rate": sample_rate,
        "n_mels": n_mels,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "clip_duration": clip_duration,
        "n_frames": n_frames,
        "logmel_shape": (n_mels, n_frames),
    }


print("Loading prebuilt log-mel split manifests ...")
train_df, val_df, test_df = load_logmel_splits(
    TRAIN_MANIFEST_PATH,
    VAL_MANIFEST_PATH,
    TEST_MANIFEST_PATH,
)

feature_config = resolve_logmel_config(train_df)
SAMPLE_RATE = int(feature_config["sample_rate"])
N_MELS = int(feature_config["n_mels"])
N_FFT = int(feature_config["n_fft"])
HOP_LENGTH = int(feature_config["hop_length"])
CLIP_DURATION = float(feature_config["clip_duration"])
N_FRAMES = int(feature_config["n_frames"])
LOGMEL_SHAPE = tuple(feature_config["logmel_shape"])

print(f"  train : {len(train_df):>6,} rows")
print(f"  val   : {len(val_df):>6,} rows")
print(f"  test  : {len(test_df):>6,} rows")
print(f"  root  : {LOGMEL_DATASET_DIR}")
print(f"  shape : {LOGMEL_SHAPE}  |  clip={CLIP_DURATION}s  |  sr={SAMPLE_RATE}")

all_genres = sorted(pd.concat([train_df, val_df, test_df], axis=0)["genre_top"].unique().tolist())
N_CLASSES = len(all_genres)
GENRE_CLASSES = all_genres
if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
    raise ValueError(
        "Train/val/test manifests must all be non-empty. "
        "Rebuild the log-mel dataset or point LOGMEL_DATASET_DIR to a complete dataset."
    )
if N_CLASSES < 2:
    raise ValueError(
        f"Expected at least 2 genres for multiclass training, found {N_CLASSES}: {GENRE_CLASSES}."
    )
label_enc = LabelEncoder().fit(GENRE_CLASSES)

_section_times["4. Load logmel manifests"] = time.perf_counter() - _t0
print(f"\nLoad manifests : {_section_times['4. Load logmel manifests']:.2f}s")


# %%
_t0 = time.perf_counter()

splits = ["train", "val", "test"]
dfs = [train_df, val_df, test_df]
split_colors = ["#2196F3", "#4CAF50", "#FF9800"]

genres_sorted = sorted(GENRE_CLASSES)
counts_matrix = np.array(
    [[df["genre_top"].value_counts().get(g, 0) for df in dfs] for g in genres_sorted],
    dtype=int,
)

row_max = counts_matrix.max(axis=1).astype(float)
row_min = np.where(counts_matrix.min(axis=1) == 0, 1, counts_matrix.min(axis=1)).astype(float)
imbalance = row_max / row_min

fig, (ax_bar, ax_imb) = plt.subplots(
    2,
    1,
    figsize=(max(12, len(genres_sorted) * 0.9), 9),
    gridspec_kw={"height_ratios": [3, 1.2]},
)

x = np.arange(len(genres_sorted))
width = 0.25
offsets = np.linspace(-(len(splits) - 1) / 2, (len(splits) - 1) / 2, len(splits)) * width

for split_name, color, offset, counts in zip(splits, split_colors, offsets, counts_matrix.T):
    bars = ax_bar.bar(
        x + offset,
        counts,
        width=width,
        label=f"{split_name} (n={counts.sum():,})",
        color=color,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.4,
    )
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(cnt), ha="center", va="bottom", fontsize=6.5)

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(genres_sorted, rotation=35, ha="right", fontsize=9)
ax_bar.set_ylabel("Sample count")
ax_bar.set_title("Genre × split counts — prebuilt log-mel dataset")
ax_bar.legend(fontsize=9, loc="upper right")
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_bar.set_axisbelow(True)
for spine in ["top", "right"]:
    ax_bar.spines[spine].set_visible(False)

bar_colors = ["#d16d1b" if ratio > 1.5 else "#159dd3" for ratio in imbalance]
ax_imb.bar(x, imbalance, color=bar_colors, width=0.55, alpha=0.85)
ax_imb.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
ax_imb.set_xticks(x)
ax_imb.set_xticklabels(genres_sorted, rotation=35, ha="right", fontsize=9)
ax_imb.set_ylabel("Imbalance ratio")
ax_imb.set_title("Split balance per genre")
ax_imb.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_imb.set_axisbelow(True)
for spine in ["top", "right"]:
    ax_imb.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()

_section_times["4b. Genre plot"] = time.perf_counter() - _t0
print(f"Genre distribution plot : {_section_times['4b. Genre plot']:.2f}s")


# %% [markdown]
# ## 5. Sanity Check a Cached Log-Mel Sample

# %%
_t0 = time.perf_counter()

sample_row = train_df.iloc[0] if len(train_df) else val_df.iloc[0] if len(val_df) else test_df.iloc[0]
sample_logmel = np.load(sample_row["logmel_path"])

fig, ax = plt.subplots(figsize=(14, 5))
img = ax.imshow(
    sample_logmel,
    aspect="auto",
    origin="lower",
    extent=[0, N_FRAMES * HOP_LENGTH / SAMPLE_RATE, 0, N_MELS],
    cmap="magma",
)
ax.set_title(f"Sample log-mel spectrogram — genre: {sample_row['genre_top']}")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mel band")
plt.colorbar(img, ax=ax, label="log(1 + power)")
plt.tight_layout()
plt.show()

_section_times["5. Sample logmel plot"] = time.perf_counter() - _t0
print(f"Sample logmel plot : {_section_times['5. Sample logmel plot']:.2f}s")


# %% [markdown]
# ## 6. Preprocessing

# %%
_t0 = time.perf_counter()

train_df["label_int"] = label_enc.transform(train_df["genre_top"].to_numpy())
val_df["label_int"] = label_enc.transform(val_df["genre_top"].to_numpy())
test_df["label_int"] = label_enc.transform(test_df["genre_top"].to_numpy())

train_labels = train_df["label_int"].to_numpy()
train_class_counts = train_df["label_int"].value_counts().reindex(range(N_CLASSES), fill_value=0).astype(int)
present_train_classes = np.unique(train_labels)
missing_train_class_ids = sorted(set(range(N_CLASSES)) - set(present_train_classes.tolist()))
present_train_counts = train_class_counts[train_class_counts > 0]
train_class_balance_ratio = (
    float(present_train_counts.max() / present_train_counts.min())
    if not present_train_counts.empty
    else float("inf")
)
USE_CLASS_WEIGHTS = train_class_balance_ratio > CLASS_WEIGHT_IMBALANCE_THRESHOLD

if USE_CLASS_WEIGHTS:
    raw_class_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_train_classes,
        y=train_labels,
    )
    capped_class_weights = np.minimum(raw_class_weights, MAX_CLASS_WEIGHT)
    class_weight_dict = {
        int(class_id): float(weight)
        for class_id, weight in zip(present_train_classes, capped_class_weights)
    }
else:
    raw_class_weights = np.ones(len(present_train_classes), dtype=np.float32)
    capped_class_weights = raw_class_weights.copy()
    class_weight_dict = {int(class_id): 1.0 for class_id in present_train_classes}

print("Train split counts by genre:")
for i, genre in enumerate(GENRE_CLASSES):
    print(f"  {genre:<20s}  {int(train_class_counts.iloc[i]):>5d}")
print(
    f"Train class balance ratio (max/min among present classes): {train_class_balance_ratio:.4f} "
    f"| class weights enabled: {USE_CLASS_WEIGHTS}"
)

if USE_CLASS_WEIGHTS:
    print(f"Class weights (capped at {MAX_CLASS_WEIGHT}):")
    for i, genre in enumerate(GENRE_CLASSES):
        if i in class_weight_dict:
            raw_w = float(raw_class_weights[list(present_train_classes).index(i)]) if i in present_train_classes else 0.0
            capped = " (capped)" if raw_w > MAX_CLASS_WEIGHT else ""
            print(f"  {genre:<20s}  {class_weight_dict[i]:.4f}{capped}  (raw: {raw_w:.4f})")
        else:
            print(f"  {genre:<20s}  absent in train -> no class weight applied")
else:
    print("Class weights disabled because the train split is already effectively balanced.")

if missing_train_class_ids:
    missing_train_genres = [GENRE_CLASSES[i] for i in missing_train_class_ids]
    print(f"[WARN] Train split is missing classes: {missing_train_genres}")

CLASS_WEIGHT_VECTOR = tf.constant(
    [class_weight_dict.get(i, 1.0) for i in range(N_CLASSES)],
    dtype=tf.float32,
)


def compute_train_stats(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sum_c = np.zeros((N_MELS,), dtype=np.float64)
    sumsq_c = np.zeros((N_MELS,), dtype=np.float64)
    count = 0

    for i, path in enumerate(index_df["logmel_path"]):
        x = np.load(path)
        sum_c += x.sum(axis=1)
        sumsq_c += (x * x).sum(axis=1)
        count += x.shape[1]
        if (i + 1) % 2000 == 0:
            print(f"  stats pass: {i + 1}/{len(index_df)}")

    mean = sum_c / max(1, count)
    var = (sumsq_c / max(1, count)) - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    mu = mean.reshape((1, N_MELS, 1, 1)).astype(np.float32)
    sigma = std.reshape((1, N_MELS, 1, 1)).astype(np.float32)
    return mu, sigma


print("\nComputing training mean/std (streaming) ...")
mu, std = compute_train_stats(train_df)
print(f"mu shape={mu.shape}, std shape={std.shape}")

AUTOTUNE = tf.data.AUTOTUNE
mu_tf = tf.constant(mu, dtype=tf.float32)
std_tf = tf.constant(std, dtype=tf.float32)


def _np_load_logmel(path_bytes):
    path = path_bytes.decode("utf-8")
    x = np.load(path).astype(np.float32, copy=False)
    x = x[..., np.newaxis]
    return x


def spec_augment(x, freq_mask=SPEC_AUG_FREQ_MASK, time_mask=SPEC_AUG_TIME_MASK, num_masks=SPEC_AUG_NUM_MASKS):
    shape = tf.shape(x)
    freq_dim = shape[0]
    time_dim = shape[1]

    for _ in range(num_masks):
        f_max = tf.minimum(freq_mask, freq_dim)
        f = tf.random.uniform([], 0, f_max + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, tf.maximum(freq_dim - f, 1), dtype=tf.int32)
        freq_mask_tensor = tf.concat(
            [
                tf.ones([f0, time_dim, 1]),
                tf.zeros([f, time_dim, 1]),
                tf.ones([freq_dim - f0 - f, time_dim, 1]),
            ],
            axis=0,
        )
        x = x * freq_mask_tensor

        t_max = tf.minimum(time_mask, time_dim)
        t = tf.random.uniform([], 0, t_max + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, tf.maximum(time_dim - t, 1), dtype=tf.int32)
        time_mask_tensor = tf.concat(
            [
                tf.ones([freq_dim, t0, 1]),
                tf.zeros([freq_dim, t, 1]),
                tf.ones([freq_dim, time_dim - t0 - t, 1]),
            ],
            axis=1,
        )
        x = x * time_mask_tensor
    return x


def spec_augment_batch(x_batch):
    augmented = tf.map_fn(
        lambda x: spec_augment(x),
        x_batch,
        fn_output_signature=tf.float32,
    )
    augmented.set_shape((None, *LOGMEL_SHAPE, 1))
    return augmented


def mixup_batch(x_batch, y_batch, alpha=MIXUP_ALPHA):
    """Apply Mixup augmentation at the batch level using one λ per sample pair."""
    batch_size = tf.shape(x_batch)[0]
    lam_a = tf.random.gamma(shape=[batch_size], alpha=alpha, dtype=tf.float32)
    lam_b = tf.random.gamma(shape=[batch_size], alpha=alpha, dtype=tf.float32)
    lam = lam_a / (lam_a + lam_b)
    lam = tf.maximum(lam, 1.0 - lam)  # keep λ >= 0.5 so the anchor sample dominates

    indices = tf.random.shuffle(tf.range(batch_size))
    x_shuffled = tf.gather(x_batch, indices)
    y_shuffled = tf.gather(y_batch, indices)

    lam_x = tf.reshape(lam, [-1, 1, 1, 1])
    lam_y = tf.reshape(lam, [-1, 1])

    x_mixed = lam_x * x_batch + (1.0 - lam_x) * x_shuffled
    y_mixed = lam_y * y_batch + (1.0 - lam_y) * y_shuffled
    x_mixed.set_shape((None, *LOGMEL_SHAPE, 1))
    y_mixed.set_shape((None, N_CLASSES))
    return x_mixed, y_mixed


def mixup_batch_weighted(x_batch, y_batch, alpha=MIXUP_ALPHA):
    """Apply Mixup with pre-Mixup (anchor-based) class weights.

    Computes each sample's class weight from its original one-hot label
    *before* blending, then returns the anchor sample's weight as the
    per-sample loss weight.  This avoids the diluted gradient signal that
    occurs when class weights are computed from post-Mixup soft labels.
    """
    anchor_weights = tf.reduce_sum(y_batch * CLASS_WEIGHT_VECTOR[tf.newaxis, :], axis=1)
    x_mixed, y_mixed = mixup_batch(x_batch, y_batch, alpha)
    anchor_weights.set_shape((None,))
    return x_mixed, y_mixed, anchor_weights


def make_dataset(index_df: pd.DataFrame, batch_size: int, shuffle: bool, augment: bool = False, weighted_mixup: bool = False) -> tf.data.Dataset:
    paths = index_df["logmel_path"].to_numpy(dtype=str)
    labels = index_df["label_int"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=SEED, reshuffle_each_iteration=True)

    def _load_and_norm(path, y):
        x = tf.numpy_function(_np_load_logmel, [path], Tout=tf.float32)
        x.set_shape((*LOGMEL_SHAPE, 1))
        x = (x - mu_tf[0]) / std_tf[0]
        y_oh = tf.one_hot(y, N_CLASSES)
        return x, y_oh

    ds = ds.map(_load_and_norm, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)

    if augment:
        # Order: normalize -> batch -> Mixup -> SpecAugment
        if weighted_mixup:
            # Pre-Mixup class weighting: anchor sample's class weight is preserved
            ds = ds.map(lambda x, y: mixup_batch_weighted(x, y, MIXUP_ALPHA), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x, y, w: (spec_augment_batch(x), y, w), num_parallel_calls=AUTOTUNE)
        else:
            ds = ds.map(lambda x, y: mixup_batch(x, y, MIXUP_ALPHA), num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda x, y: (spec_augment_batch(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, BATCH_SIZE, shuffle=True, augment=True, weighted_mixup=USE_CLASS_WEIGHTS)
train_fit_ds = train_ds
train_eval_ds = make_dataset(train_df, BATCH_SIZE, shuffle=False, augment=False)
val_ds = make_dataset(val_df, BATCH_SIZE, shuffle=False, augment=False)
test_ds = make_dataset(test_df, BATCH_SIZE, shuffle=False, augment=False)

print("\nDatasets ready:")
print(f"  train batches: {tf.data.experimental.cardinality(train_ds).numpy()}  (SpecAugment=ON, Mixup=ON)")
print(f"  train fit     : class_weights={'ON (pre-Mixup anchor)' if USE_CLASS_WEIGHTS else 'OFF'}")
print(f"  train eval    : {tf.data.experimental.cardinality(train_eval_ds).numpy()}  (SpecAugment=OFF, Mixup=OFF)")
print(f"  val   batches: {tf.data.experimental.cardinality(val_ds).numpy()}  (SpecAugment=OFF, Mixup=OFF)")
print(f"  test  batches: {tf.data.experimental.cardinality(test_ds).numpy()}  (SpecAugment=OFF, Mixup=OFF)")

_section_times["6. Preprocessing"] = time.perf_counter() - _t0
print(f"\nPreprocessing : {_section_times['6. Preprocessing']:.2f}s")


# %% [markdown]
# ## 7. Build the CNN Model

# %%
_t0 = time.perf_counter()



def _resnet_basic_block(
    x: tf.Tensor,
    filters: int,
    stage: int,
    block: int,
    stride: int = 1,
    dropout_rate: float = 0.0,
) -> tf.Tensor:
    """Standard post-activation ResNet basic block.

    This is a true ResNet-style block:
    - conv -> bn -> relu
    - conv -> bn
    - identity/projection shortcut
    - add -> relu

    A projection shortcut is used whenever spatial resolution or channel count
    changes (for example, the first block of a new stage).
    """
    shortcut = x
    in_channels = int(x.shape[-1]) if x.shape[-1] is not None else None
    needs_projection = stride != 1 or in_channels != filters

    base = f"stage{stage}_block{block}"

    x = layers.Conv2D(
        filters,
        (3, 3),
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{base}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{base}_bn1")(x)
    x = layers.ReLU(name=f"{base}_relu1")(x)

    x = layers.Conv2D(
        filters,
        (3, 3),
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{base}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{base}_bn2")(x)

    if needs_projection:
        shortcut = layers.Conv2D(
            filters,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"{base}_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{base}_proj_bn")(shortcut)

    x = layers.Add(name=f"{base}_add")([x, shortcut])
    x = layers.ReLU(name=f"{base}_out")(x)

    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate, name=f"{base}_sdrop")(x)
    return x


def build_model(n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*LOGMEL_SHAPE, 1), name="logmel")

    # ResNet-style stem: wider initial receptive field + early downsampling.
    x = layers.Conv2D(
        32,
        (7, 7),
        strides=2,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPool2D((3, 3), strides=2, padding="same", name="stem_pool")(x)

    # 4 residual stages, 2 basic blocks each — compact ResNet18-style pattern.
    x = _resnet_basic_block(x, filters=32, stage=1, block=1, stride=1, dropout_rate=SPATIAL_DROPOUT_RATE_BLOCK2)
    x = _resnet_basic_block(x, filters=32, stage=1, block=2, stride=1, dropout_rate=0.0)

    x = _resnet_basic_block(x, filters=64, stage=2, block=1, stride=2, dropout_rate=SPATIAL_DROPOUT_RATE_BLOCK3)
    x = _resnet_basic_block(x, filters=64, stage=2, block=2, stride=1, dropout_rate=0.0)

    x = _resnet_basic_block(x, filters=128, stage=3, block=1, stride=2, dropout_rate=SPATIAL_DROPOUT_RATE_BLOCK4)
    x = _resnet_basic_block(x, filters=128, stage=3, block=2, stride=1, dropout_rate=0.0)

    x = _resnet_basic_block(x, filters=256, stage=4, block=1, stride=2, dropout_rate=SPATIAL_DROPOUT_RATE_BLOCK5)
    x = _resnet_basic_block(x, filters=256, stage=4, block=2, stride=1, dropout_rate=0.0)

    # Keep the compact classifier head pattern from the v2.x family.
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc_bottleneck")(x)
    x = layers.Dropout(FINAL_DROPOUT_RATE, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="fc_out")(x)
    return keras.Model(inputs, outputs, name="logmel_cnn_v3_1")

def _load_pretrained_genre_classes(model_path: Path) -> list[str] | None:
    stats_path = model_path.parent / "norm_stats.npz"
    if not stats_path.exists():
        return None
    data = np.load(str(stats_path), allow_pickle=True)
    if "genre_classes" not in data:
        return None
    return data["genre_classes"].tolist()


def _load_source_run_report(model_path: Path) -> tuple[dict[str, object] | None, Path | None]:
    report_candidates = sorted(model_path.parent.glob("run_report_*.json"))
    if not report_candidates:
        return None, None

    report_path = report_candidates[0]
    try:
        return json.loads(report_path.read_text()), report_path
    except Exception as exc:
        print(f"[WARN] Failed to read source run report from {report_path}: {exc}")
        return None, report_path


def _validate_pretrained_backbone_model(model: keras.Model, model_path: Path) -> None:
    expected_input_shape = (None, *LOGMEL_SHAPE, 1)
    if tuple(model.input_shape) != expected_input_shape:
        raise ValueError(
            "Pretrained backbone input shape mismatch. "
            f"Expected {expected_input_shape}, got {model.input_shape} from {model_path}."
        )


def _transfer_backbone_weights(source_model: keras.Model, target_model: keras.Model) -> dict[str, object]:
    transferred_layers: list[str] = []
    skipped_layers: list[str] = []
    source_layers = {layer.name: layer for layer in source_model.layers}

    for target_layer in target_model.layers:
        if target_layer.name == "fc_out":
            skipped_layers.append("fc_out (classifier head replaced)")
            continue

        source_layer = source_layers.get(target_layer.name)
        if source_layer is None:
            skipped_layers.append(f"{target_layer.name} (missing in source model)")
            continue

        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if not source_weights and not target_weights:
            continue

        if len(source_weights) != len(target_weights) or any(sw.shape != tw.shape for sw, tw in zip(source_weights, target_weights)):
            skipped_layers.append(f"{target_layer.name} (shape mismatch)")
            continue

        target_layer.set_weights(source_weights)
        transferred_layers.append(target_layer.name)

    return {
        "transferred_layers": transferred_layers,
        "transferred_layer_count": len(transferred_layers),
        "skipped_layers": skipped_layers,
        "skipped_layer_count": len(skipped_layers),
    }


def _validate_pretrained_model(model: keras.Model, model_path: Path) -> None:
    expected_input_shape = (None, *LOGMEL_SHAPE, 1)
    if tuple(model.input_shape) != expected_input_shape:
        raise ValueError(
            "Pretrained model input shape mismatch. "
            f"Expected {expected_input_shape}, got {model.input_shape} from {model_path}."
        )

    output_units = model.output_shape[-1]
    if int(output_units) != N_CLASSES:
        raise ValueError(
            "Pretrained model output dimension mismatch. "
            f"Expected {N_CLASSES} classes, got {output_units} from {model_path}."
        )

    pretrained_genres = _load_pretrained_genre_classes(model_path)
    if pretrained_genres is not None and list(pretrained_genres) != list(GENRE_CLASSES):
        raise ValueError(
            "Pretrained model genre_classes do not match the current dataset order. "
            f"Checkpoint genres={pretrained_genres}; current genres={GENRE_CLASSES}."
        )


def _warn_if_full_model_architecture_diff(model: keras.Model) -> None:
    if model.name != "logmel_cnn_v3_1":
        print(
            "[WarmStart][WARN] full_model mode keeps the source architecture as-is. "
            f"Loaded model.name={model.name!r}, expected 'logmel_cnn_v3_1'. "
            "For cross-family warm starts, prefer --pretrained-mode backbone_only."
        )




SOURCE_RUN_REPORT = None
SOURCE_RUN_REPORT_PATH = None
WARM_START_TRANSFER_SUMMARY = None
SOURCE_PRETRAINED_GENRES = None
WARM_START_CLASS_SET_MATCH = False
WARM_START_HEAD_RESET_INTENTIONAL = False
APPLIED_FREEZE_BACKBONE_EPOCHS = 0
TRAINING_STAGES = []

if PRETRAINED_MODEL_PATH is not None:
    if not PRETRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pretrained model not found: {PRETRAINED_MODEL_PATH}")
    SOURCE_RUN_REPORT, SOURCE_RUN_REPORT_PATH = _load_source_run_report(PRETRAINED_MODEL_PATH)

    if PRETRAINED_MODE == "full_model":
        print(f"Warm start   : loading full pretrained model from {PRETRAINED_MODEL_PATH}")
        with tf.device(RUNTIME_DEVICE):
            model = tf.keras.models.load_model(str(PRETRAINED_MODEL_PATH), compile=False)
        _validate_pretrained_model(model, PRETRAINED_MODEL_PATH)
        _warn_if_full_model_architecture_diff(model)
        INIT_MODE = "warm_start_full_model"
    elif PRETRAINED_MODE == "backbone_only":
        print(f"Warm start   : loading backbone weights from {PRETRAINED_MODEL_PATH}")
        with tf.device(RUNTIME_DEVICE):
            source_model = tf.keras.models.load_model(str(PRETRAINED_MODEL_PATH), compile=False)
            model = build_model(N_CLASSES)
        _validate_pretrained_backbone_model(source_model, PRETRAINED_MODEL_PATH)
        SOURCE_PRETRAINED_GENRES = _load_pretrained_genre_classes(PRETRAINED_MODEL_PATH)
        source_output_units = int(source_model.output_shape[-1])
        WARM_START_CLASS_SET_MATCH = (
            SOURCE_PRETRAINED_GENRES is not None
            and list(SOURCE_PRETRAINED_GENRES) == list(GENRE_CLASSES)
            and source_output_units == N_CLASSES
        )
        if WARM_START_CLASS_SET_MATCH:
            WARM_START_HEAD_RESET_INTENTIONAL = True
            print(
                "[WarmStart][WARN] backbone_only was selected even though the source genre_classes "
                "still match the current dataset. The classifier head `fc_out` is being reset intentionally. "
                "Use `full_model` instead if that was not intended."
            )
        WARM_START_TRANSFER_SUMMARY = _transfer_backbone_weights(source_model, model)
        print(
            "Backbone transfer : "
            f"transferred={WARM_START_TRANSFER_SUMMARY['transferred_layer_count']} "
            f"skipped={WARM_START_TRANSFER_SUMMARY['skipped_layer_count']}"
        )
        INIT_MODE = "warm_start_backbone_only"
    else:
        raise ValueError(
            f"Unsupported PRETRAINED_MODE={PRETRAINED_MODE!r}. Use 'full_model' or 'backbone_only'."
        )
else:
    with tf.device(RUNTIME_DEVICE):
        model = build_model(N_CLASSES)
    INIT_MODE = "scratch"

model.summary()

_section_times["7. Build model"] = time.perf_counter() - _t0
print(f"\nBuild model : {_section_times['7. Build model']:.2f}s")


# %% [markdown]
# ## 8. Compile & Train

# %%
_t0 = time.perf_counter()

print(f"Init mode     : {INIT_MODE}")
if PRETRAINED_MODEL_PATH is not None:
    print(f"Warm-start mapping : source checkpoint={PRETRAINED_MODEL_PATH} -> destination run_dir={RUN_DIR}")

WARMUP_EPOCHS = 5                   # v1.1: 5 (was 3)
LR_MAX = 5e-4                       # v1.1: 5e-4 (was 1e-3)
LR_MIN = 1e-6

steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
if steps_per_epoch <= 0:
    steps_per_epoch = max(1, len(train_df) // BATCH_SIZE)

total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch


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
        warmup_lr = self.lr_min + (self.lr_max - self.lr_min) * (step / tf.maximum(self.warmup_steps, 1.0))
        progress = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
        }


lr_schedule = CosineAnnealingWithWarmup(warmup_steps, total_steps, LR_MAX, LR_MIN)


def _build_optimizer():
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY, clipnorm=1.0)


with tf.device(RUNTIME_DEVICE):
    SHARED_OPTIMIZER = _build_optimizer()
    # Register the full variable set once so staged unfreezing can continue
    # with the same optimizer state and LR-schedule progress.
    SHARED_OPTIMIZER.build(model.trainable_variables)


def compile_training_model(current_model: keras.Model) -> None:
    with tf.device(RUNTIME_DEVICE):
        current_model.compile(
            optimizer=SHARED_OPTIMIZER,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
            metrics=["accuracy"],
        )


def set_backbone_trainable(current_model: keras.Model, backbone_trainable: bool) -> None:
    head_layers = {"fc_bottleneck", "fc_out"}
    for layer in current_model.layers:
        layer.trainable = backbone_trainable or layer.name in head_layers


class _ValMacroF1(tf.keras.callbacks.Callback):
    def __init__(self, val_ds: tf.data.Dataset):
        super().__init__()
        self.val_ds = val_ds
        self.f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for xb, yb in self.val_ds:
            preds = self.model(xb, training=False).numpy()
            y_pred.append(np.argmax(preds, axis=1))
            y_true.append(np.argmax(yb.numpy(), axis=1))

        macro_f1 = float(
            f1_score(
                np.concatenate(y_true),
                np.concatenate(y_pred),
                average="macro",
                zero_division=0,
            )
        )
        self.f1_history.append(macro_f1)

        if logs is not None:
            logs["val_macro_f1"] = macro_f1


class _TrainEvalMetrics(tf.keras.callbacks.Callback):
    def __init__(self, train_eval_ds: tf.data.Dataset):
        super().__init__()
        self.train_eval_ds = train_eval_ds
        self.accuracy_history = []
        self.f1_history = []

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for xb, yb in self.train_eval_ds:
            preds = self.model(xb, training=False).numpy()
            y_pred.append(np.argmax(preds, axis=1))
            y_true.append(np.argmax(yb.numpy(), axis=1))

        train_eval_accuracy = float(accuracy_score(np.concatenate(y_true), np.concatenate(y_pred)))
        train_eval_macro_f1 = float(
            f1_score(
                np.concatenate(y_true),
                np.concatenate(y_pred),
                average="macro",
                zero_division=0,
            )
        )
        self.accuracy_history.append(train_eval_accuracy)
        self.f1_history.append(train_eval_macro_f1)

        if logs is not None:
            logs["train_eval_accuracy"] = train_eval_accuracy
            logs["train_eval_macro_f1"] = train_eval_macro_f1


class _GapAwareEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(
        self,
        gap_threshold: float,
        patience: int,
        min_delta: float,
        require_val_not_improving: bool,
        val_macro_f1_min_threshold: float | None,
        start_epoch: int,
        restore_best_weights: bool,
        verbose: int = 1,
    ):
        super().__init__()
        self.gap_threshold = float(gap_threshold)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.require_val_not_improving = bool(require_val_not_improving)
        self.val_macro_f1_min_threshold = (
            None if val_macro_f1_min_threshold is None else float(val_macro_f1_min_threshold)
        )
        self.start_epoch = int(start_epoch)
        self.restore_best_weights = bool(restore_best_weights)
        self.verbose = int(verbose)
        self.best_val_macro_f1 = float("-inf")
        self.best_epoch = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = None
        self.stop_reason = None
        self.last_gap = None
        self.gap_history = []
        self.active_history = []
        self.wait_history = []

    def _fmt(self, value: float | None) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_macro_f1 = logs.get("train_eval_macro_f1")
        val_macro_f1 = logs.get("val_macro_f1")

        if train_macro_f1 is None or val_macro_f1 is None:
            self.last_gap = None
            self.wait = 0
            self.gap_history.append(None)
            self.active_history.append(False)
            self.wait_history.append(self.wait)
            if self.verbose:
                print(
                    f"[GapEarlyStop] epoch={epoch + 1:03d} missing metrics "
                    f"train_eval_macro_f1={train_macro_f1} val_macro_f1={val_macro_f1}; counter reset"
                )
            return

        train_macro_f1 = float(train_macro_f1)
        val_macro_f1 = float(val_macro_f1)
        val_improved = val_macro_f1 > (self.best_val_macro_f1 + self.min_delta)
        if val_improved:
            self.best_val_macro_f1 = val_macro_f1
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        gap = train_macro_f1 - val_macro_f1
        self.last_gap = gap
        gap_exceeded = gap > self.gap_threshold
        val_not_improving = (not val_improved) if self.require_val_not_improving else True
        val_threshold_met = (
            True
            if self.val_macro_f1_min_threshold is None
            else val_macro_f1 >= self.val_macro_f1_min_threshold
        )
        active = (
            (epoch + 1) >= self.start_epoch
            and gap_exceeded
            and val_not_improving
            and val_threshold_met
        )
        self.wait = self.wait + 1 if active else 0
        self.gap_history.append(float(gap))
        self.active_history.append(bool(active))
        self.wait_history.append(int(self.wait))

        logs["macro_f1_gap"] = float(gap)
        logs["gap_early_stop_wait"] = int(self.wait)

        if self.verbose:
            threshold_text = (
                "disabled"
                if self.val_macro_f1_min_threshold is None
                else f">={self.val_macro_f1_min_threshold:.3f}"
            )
            print(
                "[GapEarlyStop] "
                f"epoch={epoch + 1:03d} gap={self._fmt(gap)} threshold>{self.gap_threshold:.3f}="
                f"{gap_exceeded} val_improved={val_improved} val_threshold={threshold_text} "
                f"met={val_threshold_met} active={active} wait={self.wait}/{self.patience} "
                f"best_val_macro_f1={self._fmt(self.best_val_macro_f1 if self.best_epoch is not None else None)}"
            )

        if self.patience > 0 and self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.stop_reason = (
                f"persistent_macro_f1_gap>{self.gap_threshold:.3f} for {self.patience} epoch(s)"
            )
            if self.restore_best_weights and self.best_weights is not None:
                self.model.set_weights(self.best_weights)
                if self.verbose:
                    best_epoch_text = "n/a" if self.best_epoch is None else str(self.best_epoch + 1)
                    print(
                        "[GapEarlyStop] Restored best weights from epoch "
                        f"{best_epoch_text} (best_val_macro_f1={self._fmt(self.best_val_macro_f1)})"
                    )
            self.model.stop_training = True
            if self.verbose:
                print(
                    "[GapEarlyStop] Triggered stop at epoch "
                    f"{epoch + 1} due to gap={self._fmt(gap)} and wait={self.wait}/{self.patience}"
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_threshold": self.gap_threshold,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "require_val_not_improving": self.require_val_not_improving,
            "val_macro_f1_min_threshold": self.val_macro_f1_min_threshold,
            "start_epoch": self.start_epoch,
            "restore_best_weights": self.restore_best_weights,
            "best_epoch": None if self.best_epoch is None else self.best_epoch + 1,
            "best_val_macro_f1": None if self.best_epoch is None else round(float(self.best_val_macro_f1), 6),
            "stopped_epoch": None if self.stopped_epoch is None else self.stopped_epoch + 1,
            "stop_reason": self.stop_reason,
            "final_wait": self.wait,
            "last_gap": None if self.last_gap is None else round(float(self.last_gap), 6),
            "active_history": self.active_history,
            "wait_history": self.wait_history,
        }


class _EpochMetricsSummary(tf.keras.callbacks.Callback):
    def __init__(self, max_epochs: int):
        super().__init__()
        self.max_epochs = max_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get("loss")
        val_loss = logs.get("val_loss")
        train_acc = logs.get("train_eval_accuracy")
        val_acc = logs.get("val_accuracy")
        train_macro_f1 = logs.get("train_eval_macro_f1")
        val_macro_f1 = logs.get("val_macro_f1")

        acc_gap = None if train_acc is None or val_acc is None else float(train_acc) - float(val_acc)
        f1_gap = None if train_macro_f1 is None or val_macro_f1 is None else float(train_macro_f1) - float(val_macro_f1)

        def _fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"

        print(
            "\n"
            f"  [EpochSummary] {epoch + 1:03d}/{self.max_epochs:03d} "
            f"train_loss={_fmt(train_loss)} val_loss={_fmt(val_loss)} "
            f"train_acc={_fmt(train_acc)} val_acc={_fmt(val_acc)} acc_gap={_fmt(acc_gap)} "
            f"train_macro_f1={_fmt(train_macro_f1)} val_macro_f1={_fmt(val_macro_f1)} f1_gap={_fmt(f1_gap)}"
        )


class _LRLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.lrs = []

    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        lr_attr = opt.learning_rate
        lr = float(lr_attr(opt.iterations)) if callable(lr_attr) else float(lr_attr)
        self.lrs.append(lr)


class _EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.times = []
        self._epoch_t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_t0 = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(round(time.perf_counter() - self._epoch_t0, 3))


lr_logger = _LRLogger()
epoch_timer = _EpochTimer()
val_macro_f1_cb = _ValMacroF1(val_ds=val_ds)
train_eval_metrics_cb = _TrainEvalMetrics(train_eval_ds=train_eval_ds)
epoch_metrics_summary_cb = _EpochMetricsSummary(max_epochs=EPOCHS)


def make_training_callbacks(enable_checkpointing: bool, enable_early_stopping: bool):
    stage_callbacks = [
        val_macro_f1_cb,
        train_eval_metrics_cb,
    ]
    if enable_early_stopping:
        stage_callbacks.append(
            _GapAwareEarlyStopping(
                gap_threshold=GAP_EARLY_STOP_THRESHOLD,
                patience=GAP_EARLY_STOP_PATIENCE,
                min_delta=GAP_EARLY_STOP_MIN_DELTA,
                require_val_not_improving=GAP_EARLY_STOP_REQUIRE_VAL_NOT_IMPROVING,
                val_macro_f1_min_threshold=GAP_EARLY_STOP_VAL_MACRO_F1_MIN_THRESHOLD,
                start_epoch=GAP_EARLY_STOP_START_EPOCH,
                restore_best_weights=GAP_EARLY_STOP_RESTORE_BEST_WEIGHTS,
                verbose=1,
            )
        )
    stage_callbacks.append(epoch_metrics_summary_cb)
    if enable_checkpointing:
        stage_callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(RUN_DIR / "best_model_macro_f1.keras"),
                monitor="val_macro_f1",
                mode="max",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )
    if enable_early_stopping:
        stage_callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                patience=STANDARD_EARLY_STOP_PATIENCE,
                min_delta=STANDARD_EARLY_STOP_MIN_DELTA,
                start_from_epoch=STANDARD_EARLY_STOP_START_EPOCH,
                restore_best_weights=STANDARD_EARLY_STOP_RESTORE_BEST_WEIGHTS,
                verbose=1,
            )
        )
    stage_callbacks.extend([lr_logger, epoch_timer])
    return stage_callbacks


def _merge_stage_histories(stage_histories: list[keras.callbacks.History]) -> SimpleNamespace:
    merged_history = {}
    for stage_history in stage_histories:
        for key, values in stage_history.history.items():
            merged_history.setdefault(key, []).extend(values)
    return SimpleNamespace(history=merged_history)


def _run_training_stage(
    stage_name: str,
    epochs_to_run: int,
    initial_epoch: int,
    backbone_trainable: bool,
    enable_checkpointing: bool,
    enable_early_stopping: bool,
):
    if epochs_to_run <= 0:
        return None

    set_backbone_trainable(model, backbone_trainable)
    compile_training_model(model)
    print(
        f"Training stage : {stage_name} | epochs {initial_epoch + 1}..{initial_epoch + epochs_to_run} | "
        f"backbone_trainable={backbone_trainable}"
    )
    stage_callbacks = make_training_callbacks(enable_checkpointing, enable_early_stopping)
    with tf.device(RUNTIME_DEVICE):
        stage_history = model.fit(
            train_fit_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + epochs_to_run,
            callbacks=stage_callbacks,
        )

    completed_epochs = len(stage_history.history.get("loss", []))
    gap_early_stop_cb = next(
        (callback for callback in stage_callbacks if isinstance(callback, _GapAwareEarlyStopping)),
        None,
    )
    TRAINING_STAGES.append(
        {
            "name": stage_name,
            "backbone_trainable": backbone_trainable,
            "epochs_requested": epochs_to_run,
            "epochs_completed": completed_epochs,
            "initial_epoch": initial_epoch,
            "final_epoch": initial_epoch + completed_epochs,
            "gap_aware_early_stopping": gap_early_stop_cb.as_dict() if gap_early_stop_cb is not None else None,
        }
    )
    return stage_history

print(f"Training on {RUNTIME_DEVICE} | epochs={EPOCHS}, batch_size={BATCH_SIZE}")
print(f"Input source : {LOGMEL_DATASET_DIR}")
print(
    f"Optimizer    : AdamW(cosine_annealing, warmup={WARMUP_EPOCHS}ep, lr_max={LR_MAX}, weight_decay={WEIGHT_DECAY})"
)
print("Optimizer continuity : preserved across staged freeze/unfreeze training")
if PRETRAINED_MODEL_PATH is not None:
    print(f"Warm-start   : {PRETRAINED_MODEL_PATH}")
    print(f"Warm-start mode : {PRETRAINED_MODE}")
if FREEZE_BACKBONE_EPOCHS > 0 and INIT_MODE != "warm_start_backbone_only":
    print(
        "[WarmStart][WARN] --freeze-backbone-epochs was provided, but it only applies to "
        "backbone_only warm starts. The value will be ignored for this run."
    )

requested_frozen_epochs = FREEZE_BACKBONE_EPOCHS if INIT_MODE == "warm_start_backbone_only" else 0
APPLIED_FREEZE_BACKBONE_EPOCHS = min(requested_frozen_epochs, EPOCHS)

stage_histories = []
if APPLIED_FREEZE_BACKBONE_EPOCHS > 0:
    frozen_stage_history = _run_training_stage(
        stage_name="head_only_frozen_backbone",
        epochs_to_run=APPLIED_FREEZE_BACKBONE_EPOCHS,
        initial_epoch=0,
        backbone_trainable=False,
        enable_checkpointing=APPLIED_FREEZE_BACKBONE_EPOCHS == EPOCHS,
        enable_early_stopping=APPLIED_FREEZE_BACKBONE_EPOCHS == EPOCHS,
    )
    if frozen_stage_history is not None:
        stage_histories.append(frozen_stage_history)

remaining_epochs = EPOCHS - APPLIED_FREEZE_BACKBONE_EPOCHS
if remaining_epochs > 0:
    full_stage_history = _run_training_stage(
        stage_name="full_finetune",
        epochs_to_run=remaining_epochs,
        initial_epoch=APPLIED_FREEZE_BACKBONE_EPOCHS,
        backbone_trainable=True,
        enable_checkpointing=True,
        enable_early_stopping=True,
    )
    if full_stage_history is not None:
        stage_histories.append(full_stage_history)

if not stage_histories:
    raise ValueError("No training stages were executed. Check EPOCHS and freeze-backbone settings.")

history = _merge_stage_histories(stage_histories)

num_epochs = len(history.history["loss"])
best_loss_epoch = min(range(num_epochs), key=lambda i: history.history["val_loss"][i])
best_f1_epoch = int(np.argmax(val_macro_f1_cb.f1_history)) if val_macro_f1_cb.f1_history else best_loss_epoch
best_epoch = best_f1_epoch

print(f"\nBest epoch by val_loss : {best_loss_epoch + 1}/{num_epochs}  (val_loss={history.history['val_loss'][best_loss_epoch]:.4f})")
print(f"Best epoch by macro_F1 : {best_f1_epoch + 1}/{num_epochs}  (val_macro_f1={val_macro_f1_cb.f1_history[best_f1_epoch]:.4f})")

_section_times["8. Compile & train"] = time.perf_counter() - _t0
print(f"\nCompile & train : {_section_times['8. Compile & train']:.1f}s")

model_path = RUN_DIR / "logmel_cnn_v3_1.keras"
model.save(str(model_path))
np.savez(str(RUN_DIR / "norm_stats.npz"), mu=mu, std=std, genre_classes=np.array(GENRE_CLASSES))


# %% [markdown]
# ## 9. Training History

# %%
_t0 = time.perf_counter()

hist = history.history
epochs_range = range(1, len(hist["accuracy"]) + 1)

fig, (ax_acc, ax_loss, ax_f1) = plt.subplots(1, 3, figsize=(18, 4))

if train_eval_metrics_cb.accuracy_history:
    ax_acc.plot(
        epochs_range,
        train_eval_metrics_cb.accuracy_history,
        label="Train (clean eval)",
        linewidth=2,
    )
ax_acc.plot(epochs_range, hist["val_accuracy"], label="Validation", linewidth=2, linestyle="--")
ax_acc.set_title("Accuracy")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.legend(fontsize=8)
ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax_loss.plot(epochs_range, hist["loss"], label="Train", linewidth=2)
ax_loss.plot(epochs_range, hist["val_loss"], label="Validation", linewidth=2, linestyle="--")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Categorical cross-entropy")
ax_loss.legend(fontsize=8)
ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

if val_macro_f1_cb.f1_history:
    if train_eval_metrics_cb.f1_history:
        ax_f1.plot(
            epochs_range,
            train_eval_metrics_cb.f1_history,
            label="Train Macro-F1 (clean eval)",
            linewidth=2,
            color="seagreen",
            linestyle="--",
        )
    ax_f1.plot(epochs_range, val_macro_f1_cb.f1_history, label="Val Macro-F1", linewidth=2.5, color="darkorange")
    ax_f1.axvline(best_f1_epoch + 1, color="darkorange", linestyle=":", linewidth=1.5, label=f"Best F1 epoch ({best_f1_epoch + 1})")
    if best_loss_epoch != best_f1_epoch:
        ax_f1.axvline(best_loss_epoch + 1, color="steelblue", linestyle=":", linewidth=1.2, label=f"Best val_loss epoch ({best_loss_epoch + 1})")
    ax_f1.set_title("Val Macro-F1")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro-F1")
    ax_f1.set_ylim(0, 1)
    ax_f1.legend(fontsize=8)
    ax_f1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

for axis in (ax_acc, ax_loss, ax_f1):
    for spine in ["top", "right"]:
        axis.spines[spine].set_visible(False)

fig.suptitle("Training history — Log-Mel CNN v2.3")
plt.tight_layout()
plt.show()

_section_times["9. Training history plot"] = time.perf_counter() - _t0
print(f"Training history plot : {_section_times['9. Training history plot']:.2f}s")


# %% [markdown]
# ## 10. Evaluate the Model

# %%
_t0 = time.perf_counter()


def eval_dataset(eval_model, ds: tf.data.Dataset, genre_classes, split_label: str):
    eval_results = eval_model.evaluate(ds, verbose=0, return_dict=True)
    cost = float(eval_results.get("loss", float("nan")))
    labels = np.arange(len(genre_classes), dtype=np.int64)

    y_true, y_pred = [], []
    for xb, yb in ds:
        pred = eval_model(xb, training=False).numpy()
        y_pred.append(np.argmax(pred, axis=1))
        y_true.append(np.argmax(yb.numpy(), axis=1))
    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cr_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=genre_classes,
        zero_division=0,
    )
    cr_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=genre_classes,
        zero_division=0,
        output_dict=True,
    )
    print(
        f"\n{'=' * 60}\n{split_label}\n{'=' * 60}\n"
        f"  Cost     : {cost:.4f}\n"
        f"  Accuracy : {acc:.4f} ({acc:.2%})\n"
        f"  Macro-F1 : {macro_f1:.4f}\n\n"
        f"Per-genre classification report:\n{cr_text}"
    )
    per_genre = {genre: {k: round(float(v), 4) for k, v in cr_dict[genre].items()} for genre in genre_classes if genre in cr_dict}
    metrics = {
        "cost": round(cost, 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_genre": per_genre,
    }
    return y_true, y_pred, metrics


best_f1_path = RUN_DIR / "best_model_macro_f1.keras"
if best_f1_path.exists():
    print(f"Loading best macro-F1 model for evaluation: {best_f1_path}")
    eval_model = tf.keras.models.load_model(str(best_f1_path))
else:
    print("[WARN] best_model_macro_f1.keras not found; evaluating in-memory model.")
    eval_model = model

with tf.device(RUNTIME_DEVICE):
    y_train_true, y_train_pred, train_metrics = eval_dataset(eval_model, train_eval_ds, GENRE_CLASSES, "TRAIN SET")
    y_val_true, y_val_pred, val_metrics = eval_dataset(eval_model, val_ds, GENRE_CLASSES, "VALIDATION SET")
    y_test_true, y_test_pred, test_metrics = eval_dataset(eval_model, test_ds, GENRE_CLASSES, "TEST SET")

print("\nPrimary metric — Macro-F1:")
for label, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
    print(f"  {label:<5} macro-f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")

cm = confusion_matrix(y_test_true, y_test_pred, labels=np.arange(N_CLASSES))
fig, ax = plt.subplots(figsize=(13, 11))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRE_CLASSES)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues", values_format="d")
ax.set_title("Confusion matrix — test set (logmel_cnn_v3_1)")
plt.tight_layout()
plt.show()

_section_times["10. Evaluation"] = time.perf_counter() - _t0
print(f"Evaluation : {_section_times['10. Evaluation']:.2f}s")


def _build_warm_start_comparison() -> dict[str, object]:
    comparison = {
        "enabled": PRETRAINED_MODEL_PATH is not None,
        "mode": PRETRAINED_MODE if PRETRAINED_MODEL_PATH is not None else None,
        "source_checkpoint": str(PRETRAINED_MODEL_PATH) if PRETRAINED_MODEL_PATH is not None else None,
        "source_run_report": str(SOURCE_RUN_REPORT_PATH) if SOURCE_RUN_REPORT_PATH is not None else None,
        "source_run_id": SOURCE_RUN_REPORT.get("run_id") if SOURCE_RUN_REPORT else None,
        "source_genre_classes": SOURCE_PRETRAINED_GENRES,
        "class_set_matches_current": WARM_START_CLASS_SET_MATCH,
        "head_reset_intentional": WARM_START_HEAD_RESET_INTENTIONAL,
        "transfer_summary": WARM_START_TRANSFER_SUMMARY,
    }
    if PRETRAINED_MODEL_PATH is None:
        return comparison

    source_evaluation = SOURCE_RUN_REPORT.get("evaluation") if SOURCE_RUN_REPORT else None
    comparison["source_evaluation"] = source_evaluation
    comparison["new_evaluation"] = {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }

    if not isinstance(source_evaluation, dict):
        comparison["delta_vs_source"] = None
        return comparison

    deltas = {}
    for split_name, new_metrics in {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }.items():
        source_metrics = source_evaluation.get(split_name)
        if not isinstance(source_metrics, dict):
            continue
        source_macro_f1 = source_metrics.get("macro_f1")
        source_accuracy = source_metrics.get("accuracy")
        if source_macro_f1 is None or source_accuracy is None:
            continue
        deltas[split_name] = {
            "source_macro_f1": round(float(source_macro_f1), 4),
            "new_macro_f1": round(float(new_metrics["macro_f1"]), 4),
            "delta_macro_f1": round(float(new_metrics["macro_f1"]) - float(source_macro_f1), 4),
            "source_accuracy": round(float(source_accuracy), 4),
            "new_accuracy": round(float(new_metrics["accuracy"]), 4),
            "delta_accuracy": round(float(new_metrics["accuracy"]) - float(source_accuracy), 4),
        }
    comparison["delta_vs_source"] = deltas or None
    return comparison


# %% [markdown]
# ## 11. Save Run Report

# %%
_t0 = time.perf_counter()

report = {
    "run_id": f"{RUN_FAMILY}-{_run_ts}",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "initialization": {
        "mode": INIT_MODE,
        "pretrained_model": str(PRETRAINED_MODEL_PATH) if PRETRAINED_MODEL_PATH is not None else None,
        "pretrained_mode": PRETRAINED_MODE if PRETRAINED_MODEL_PATH is not None else None,
    },
    "warm_start_comparison": _build_warm_start_comparison(),
    "artifacts": {
        "console_log": str(CONSOLE_LOG_PATH),
    },
    "logmel_dataset_dir": str(LOGMEL_DATASET_DIR),
    "manifests": {
        "train": str(TRAIN_MANIFEST_PATH),
        "val": str(VAL_MANIFEST_PATH),
        "test": str(TEST_MANIFEST_PATH),
    },
    "feature_config": {
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "clip_duration_sec": CLIP_DURATION,
        "logmel_shape": list(LOGMEL_SHAPE),
    },
    "dataset": {
        "n_classes": N_CLASSES,
        "genres": GENRE_CLASSES,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
    },
    "training": {
        "architecture": {
            "family": "true_resnet_style",
            "stem": {"filters": 32, "kernel": [7, 7], "stride": 2, "pool": [3, 3], "pool_stride": 2},
            "stages": [
                {"stage": 1, "filters": 32, "blocks": 2, "downsample_first_block": False},
                {"stage": 2, "filters": 64, "blocks": 2, "downsample_first_block": True},
                {"stage": 3, "filters": 128, "blocks": 2, "downsample_first_block": True},
                {"stage": 4, "filters": 256, "blocks": 2, "downsample_first_block": True},
            ],
            "block_type": "basic_residual_block",
            "classifier_head": "gap -> dense(256) -> dropout -> softmax",
        },
        "epochs_max": EPOCHS,
        "epochs_actual": num_epochs,
        "batch_size": BATCH_SIZE,
        "optimizer_state_preserved_between_stages": True,
        "standard_early_stopping": {
            "monitor": "val_macro_f1",
            "patience": STANDARD_EARLY_STOP_PATIENCE,
            "min_delta": STANDARD_EARLY_STOP_MIN_DELTA,
            "start_from_epoch": STANDARD_EARLY_STOP_START_EPOCH,
            "restore_best_weights": STANDARD_EARLY_STOP_RESTORE_BEST_WEIGHTS,
        },
        "gap_aware_early_stopping": {
            "gap_threshold": GAP_EARLY_STOP_THRESHOLD,
            "patience": GAP_EARLY_STOP_PATIENCE,
            "min_delta": GAP_EARLY_STOP_MIN_DELTA,
            "require_val_not_improving": GAP_EARLY_STOP_REQUIRE_VAL_NOT_IMPROVING,
            "val_macro_f1_min_threshold": GAP_EARLY_STOP_VAL_MACRO_F1_MIN_THRESHOLD,
            "start_epoch": GAP_EARLY_STOP_START_EPOCH,
            "restore_best_weights": GAP_EARLY_STOP_RESTORE_BEST_WEIGHTS,
        },
        "freeze_backbone_epochs_requested": FREEZE_BACKBONE_EPOCHS,
        "freeze_backbone_epochs_applied": APPLIED_FREEZE_BACKBONE_EPOCHS,
        "training_stages": TRAINING_STAGES,
        "label_smoothing": LABEL_SMOOTHING,
        "weight_decay": WEIGHT_DECAY,
        "mixup_alpha": MIXUP_ALPHA,
        "spec_aug_freq_mask": SPEC_AUG_FREQ_MASK,
        "spec_aug_time_mask": SPEC_AUG_TIME_MASK,
        "class_weights_enabled": USE_CLASS_WEIGHTS,
        "class_weight_balance_ratio": round(float(train_class_balance_ratio), 4),
        "max_class_weight": MAX_CLASS_WEIGHT,
        "spatial_dropout_block2": SPATIAL_DROPOUT_RATE_BLOCK2,
        "spatial_dropout_block3": SPATIAL_DROPOUT_RATE_BLOCK3,
        "spatial_dropout_block4": SPATIAL_DROPOUT_RATE_BLOCK4,
        "spatial_dropout_block5": SPATIAL_DROPOUT_RATE_BLOCK5,
        "final_dropout": FINAL_DROPOUT_RATE,
        "warmup_epochs": WARMUP_EPOCHS,
        "lr_max": LR_MAX,
        "best_epoch_val_loss": best_loss_epoch + 1,
        "best_epoch_macro_f1": best_f1_epoch + 1,
        "train_eval_accuracy_per_epoch": train_eval_metrics_cb.accuracy_history,
        "train_eval_macro_f1_per_epoch": train_eval_metrics_cb.f1_history,
        "lr_per_epoch": lr_logger.lrs,
        "seconds_per_epoch": epoch_timer.times,
    },
    "evaluation": {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    },
    "changes_from_v2_2": {
        "backbone": "Replaced the plain VGG-like conv stack with a true ResNet-style stem plus 4 residual stages of 2 basic blocks each",
        "downsampling": "Spatial reduction now follows a ResNet pattern: stem stride-2 conv + stem max-pool + stride-2 transitions at the first block of stages 2/3/4",
        "shortcuts": "Identity shortcuts are used inside same-shape blocks; 1×1 projection shortcuts are used whenever stage transitions change resolution or channel width",
        "classifier_head": "Kept compact and familiar: GAP -> Dense(256) -> Dropout -> Softmax",
        "regularization_recipe": "Most of the v2.2 training recipe is preserved so validation differences are easier to attribute to the backbone change",
        "warm_start_note": "full_model mode now warns when the loaded checkpoint belongs to a different architecture family; backbone_only remains the safer cross-family option, though cross-family transfer may be partial when layer names differ",
    },
}

report_path = RUN_DIR / "run_report_logmel_cnn_v3_1.json"
report_path.write_text(json.dumps(report, indent=2))
print(f"Run report -> {report_path}")

_section_times["11. Save report"] = time.perf_counter() - _t0
print(f"Save report : {_section_times['11. Save report']:.2f}s")


# %% [markdown]
# ## Runtime Summary

# %%
total_runtime = time.perf_counter() - _T0

sep = "=" * 52
print(sep)
print("  Runtime summary")
print(sep)
for section, elapsed in _section_times.items():
    bar_len = max(1, int(elapsed / total_runtime * 30))
    bar = "█" * bar_len
    pct = elapsed / total_runtime * 100
    mins, secs = divmod(elapsed, 60)
    time_str = f"{int(mins)}m {secs:04.1f}s" if mins else f"{elapsed:6.1f}s"
    print(f"  {section:<28}  {time_str:>9}  {pct:5.1f}%  {bar}")

mins_total, secs_total = divmod(total_runtime, 60)
total_str = f"{int(mins_total)}m {secs_total:04.1f}s" if mins_total else f"{total_runtime:.1f}s"
print(sep)
print(f"  {'TOTAL':<28}  {total_str:>9}")
print(sep)
