# %% [markdown]
# # FMA Baseline - Log-Mel → 2D CNN (Version 20a1)
#
# **Accuracy-focused upgrade** — builds on v20a with deeper architecture,
# reduced mel bands, cosine LR schedule, and improved regularization.
#
# Key improvements over v20a:
# - **Deeper architecture** — 5 conv blocks (32→64→128→256→256), +SpatialDropout2D
# - **Proper BN ordering** — Conv (linear) → BN → ReLU instead of Conv(ReLU) → BN
# - **N_MELS 256→128** — smoother frequency representation, faster training
# - **SpecAugment** — freq_mask=15, time_mask=25, num_masks=2 (unchanged from v20a)
# - **Cosine annealing LR** with 3-epoch warmup (replaces ReduceLROnPlateau)
# - **Batch size 16→32** — more stable gradients
# - **EarlyStopping re-enabled** (patience=9, restore_best_weights=True)
# - **_MacroF1Checkpoint** — persist best val_macro_f1 weights to disk
#
# References:
# - `docs/Implementation Guide - baseline_logmel_cnn_v20a.md`
# - `docs/Proposed Quality Improvement Plan for baseline_logmel_cnn_v20.ipynb.md`
# - `MelCNN-MGR/notebooks/baseline_logmel_cnn_v20a.py`
# - `docs/Final-Project-Proposal.md`

# %% [markdown]
# ## Changelog — v20a → v20a1
#
# | # | Section | What changed | Why |
# |---|---------|-------------|-----|
# | 1 | Config | `N_MELS` 256 → 128 | Smoother frequency representation; 256 was near-1:1 with FFT bins |
# | 2 | Config | `BATCH_SIZE` 16 → 32 | More stable gradients |
# | 3 | Config | SpecAugment unchanged (15/25/2) | Kept v20a params; no coverage change in this version |
# | 4 | Config | Cache dir `logmel_v20a1_10s` | Separate namespace (different N_MELS) |
# | 5 | Model (Section 6) | 5 conv blocks (32→64→128→256→256) | Deeper capacity for 16-class task |
# | 6 | Model (Section 6) | BN ordering: Conv(linear)→BN→ReLU | Normalize pre-activation values |
# | 7 | Model (Section 6) | SpatialDropout2D(0.10) after blocks 2 & 3 | Regularize intermediate feature maps |
# | 8 | Compile (Section 7) | Cosine annealing LR with 3-epoch warmup | Smoother convergence vs ReduceLROnPlateau |
# | 9 | Compile (Section 7) | EarlyStopping re-enabled (patience=9) | Stop overfitting, restore best weights |
# | 10 | Compile (Section 7) | _MacroF1Checkpoint added | Persist best val_macro_f1 model to disk |
#
# ### Training behavior: v20a vs v20a1
#
# | Aspect | v20a | v20a1 |
# |--------|------|-------|
# | N_MELS | 256 | 128 |
# | Batch size | 16 | 32 |
# | Architecture | 4 blocks (32→64→128→128) | 5 blocks (32→64→128→256→256) |
# | BN ordering | Conv(ReLU)→BN | Conv(linear)→BN→ReLU |
# | SpatialDropout | None | 0.10 after blocks 2 & 3 |
# | SpecAugment | freq=15, time=25, masks=2 | freq=15, time=25, masks=2 (unchanged) |
# | LR schedule | ReduceLROnPlateau(patience=3) | Cosine annealing + 3-epoch warmup |
# | EarlyStopping | Disabled | Enabled (patience=9) |
# | ModelCheckpoint | None | _MacroF1Checkpoint (saves best val_macro_f1) |

# %% [markdown]
# ## 1. Imports

# %%
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os

# Turn off oneDNN verbose logging for this Python process
os.environ["ONEDNN_VERBOSE"] = "none"
os.environ["DNNL_VERBOSE"] = "0"   # legacy/compatibility knob often still honored

warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

print(f"Python     : {sys.version.split()[0]}")
print(f"TensorFlow : {tf.__version__}")

# %% [markdown]
# ## Device runtime selection (CUDA -> Intel XPU -> CPU)
#
# This notebook prefers acceleration in this order:
# 1. CUDA GPU (`/GPU:0`)
# 2. Intel XPU (`/XPU:0`) via ITEX
# 3. CPU (`/CPU:0`)

# %% [markdown]
# ## 2. Configuration & Hyperparameters
#
# Set `SUBSET` to `"tiny"`, `"small"`, or `"medium"`.
# Set `CLEAR_CACHE = True` to force re-extraction.

# %%
# -- Paths ---------------------------------------------------------------------
NOTEBOOK_DIR    = Path(__file__).resolve().parent   # always the script's own dir
MELCNN_DIR      = NOTEBOOK_DIR.parent
WORKSPACE       = MELCNN_DIR.parent

PROCESSED_DIR   = MELCNN_DIR / "data" / "processed"
CACHE_DIR       = MELCNN_DIR / "cache"

# -- Per-run model directory ---------------------------------------------------
MODELS_BASE_DIR = MELCNN_DIR / "models"
RUN_DIR         = None

# -- Subset --------------------------------------------------------------------
SUBSET        = "medium"   # "tiny" | "small" | "medium" | "large"
CLEAR_CACHE   = False

# -- Audio backend -------------------------------------------------------------
import shutil as _shutil
FFMPEG_AVAILABLE = _shutil.which("ffmpeg") is not None
AUDIO_BACKEND = "ffmpeg" if FFMPEG_AVAILABLE else "librosa"
print(f"[Audio] backend = {AUDIO_BACKEND} (ffmpeg_available={FFMPEG_AVAILABLE})")

# -- Performance knobs ---------------------------------------------------------
NUM_WORKERS   = min(3, (os.cpu_count() or 6))

# -- Audio sanity checks -------------------------------------------------------
MIN_SECONDS   = 1.0
SKIP_SILENT   = False
SILENCE_PEAK  = 1e-4
SILENCE_STD   = 1e-5

# -- Clip duration (Strategy 1: deterministic center extraction) ---------------
CLIP_DURATION = 10.0   # seconds — center-crop / center-pad target

# -- Cache layout (per-track, separate from v20 cache) -------------------------
LOGMEL_CACHE_SHARED = True
LOGMEL_CACHE_DIR = CACHE_DIR / "logmel_v20a_10s" / ("shared" if LOGMEL_CACHE_SHARED else SUBSET)

# -- Training hyperparameters --------------------------------------------------
EPOCHS        = 60
BATCH_SIZE    = 32

# -- v20a improvements ---------------------------------------------------------
LABEL_SMOOTHING = 0.02          # label smoothing for cross-entropy loss
WEIGHT_DECAY    = 1e-4          # AdamW decoupled weight decay

# -- SpecAugment parameters ----------------------------------------------------
SPEC_AUG_FREQ_MASK  = 15       # max frequency bands to mask  (unchanged from v20a)
SPEC_AUG_TIME_MASK  = 25       # max time frames to mask      (unchanged from v20a)
SPEC_AUG_NUM_MASKS  = 2        # number of masks per axis     (unchanged from v20a)

# -- Log-mel extraction params -------------------------------------------------
SAMPLE_RATE   = 22050
N_MELS        = 128
N_FFT         = 512
HOP_LENGTH    = 256
N_FRAMES      = int(CLIP_DURATION * SAMPLE_RATE / HOP_LENGTH)   # 861 for 10s
LOGMEL_SHAPE  = (N_MELS, N_FRAMES)

print(f"[Clip] CLIP_DURATION={CLIP_DURATION}s  →  N_FRAMES={N_FRAMES}  →  LOGMEL_SHAPE={LOGMEL_SHAPE}")

# -- Reproducibility seed ------------------------------------------------------
SEED = 36

import random
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
print(f"[Seed] SEED={SEED} applied to random / numpy / tensorflow")

# -- Global timer --------------------------------------------------------------
import time as _time_module
_T0             = _time_module.perf_counter()
_section_times  = {}

# %%
_t0 = _time_module.perf_counter()

import platform
import traceback

def _best_effort_set_memory_growth(tf, device_type: str):
    devs = tf.config.list_physical_devices(device_type)
    for d in devs:
        try:
            tf.config.experimental.set_memory_growth(d, True)
        except Exception:
            pass
    return devs

def _smoke_test_matmul(tf, device: str, n: int = 1024) -> tuple[bool, str]:
    try:
        with tf.device(device):
            a = tf.random.normal([n, n])
            b = tf.random.normal([n, n])
            c = tf.matmul(a, b)
            _ = c[0, 0].numpy()
        return True, "ok"
    except Exception as e:
        return False, repr(e)

def configure_runtime_device(tf):
    print(f"Platform   : {platform.platform()}")
    print(f"TensorFlow : {tf.__version__}")

    try:
        gpus = _best_effort_set_memory_growth(tf, "GPU")
    except Exception:
        gpus = []
    if gpus:
        ok, info = _smoke_test_matmul(tf, "/GPU:0", n=1024)
        if ok:
            return "/GPU:0", "cuda", [d.name for d in gpus], info
        print("CUDA present but failed smoke test ->", info)

    try:
        import intel_extension_for_tensorflow as itex  # noqa: F401
        xpus = _best_effort_set_memory_growth(tf, "XPU")
    except Exception as e:
        xpus = []
        print("ITEX/XPU not available:", repr(e))

    if xpus:
        ok, info = _smoke_test_matmul(tf, "/XPU:0", n=1024)
        if ok:
            return "/XPU:0", "xpu", [d.name for d in xpus], info
        print("XPU present but failed smoke test ->", info)

    return "/CPU:0", "cpu", [], "ok"

RUNTIME_DEVICE, BACKEND, ACCEL_NAMES, SMOKE_INFO = configure_runtime_device(tf)

print(f"Backend    : {BACKEND.upper()} ({RUNTIME_DEVICE})")
if ACCEL_NAMES:
    print(f"Devices    : {ACCEL_NAMES}")
else:
    print("Devices    : none detected -> CPU fallback")
print(f"Smoke test : {SMOKE_INFO}")

_section_times["2. Device setup"] = _time_module.perf_counter() - _t0
print()
print(f"Device setup : {_section_times['2. Device setup']:.2f}s")

# %% [markdown]
# ## 3. Load Manifest Splits

# %%
_t0 = _time_module.perf_counter()

def load_manifest_splits(processed_dir: Path, subset: str):
    def _load(name: str) -> pd.DataFrame:
        path = processed_dir / f"{name}_{subset}.parquet"
        if not path.exists():
            msg_lines = [
                f"Manifest parquet not found: {path}",
                "Run build_manifest.py first:",
                "  python MelCNN-MGR/preprocessing/build_manifest.py",
            ]
            raise FileNotFoundError(chr(10).join(msg_lines))
        return pd.read_parquet(path)

    train_df = _load("train")
    val_df   = _load("val")
    test_df  = _load("test")
    return train_df, val_df, test_df

print("Loading manifest parquets ...")
train_df, val_df, test_df = load_manifest_splits(PROCESSED_DIR, SUBSET)

print(f"  train : {len(train_df):>5,} rows")
print(f"  val   : {len(val_df):>5,} rows")
print(f"  test  : {len(test_df):>5,} rows")

all_genres    = sorted(pd.concat([train_df, val_df, test_df])["genre_top"].unique().tolist())
N_CLASSES     = len(all_genres)
GENRE_CLASSES = all_genres
print()
print(f"  Genres ({N_CLASSES}): {GENRE_CLASSES}")

label_enc = LabelEncoder().fit(GENRE_CLASSES)

_section_times["3. Load manifest"] = _time_module.perf_counter() - _t0
print()
print(f"Load manifest : {_section_times['3. Load manifest']:.2f}s")

# %%
_t0 = _time_module.perf_counter()

splits      = ["train", "val", "test"]
dfs         = [train_df, val_df, test_df]
split_colors = ["#2196F3", "#4CAF50", "#FF9800"]

genres_sorted = sorted(all_genres)
n_genres      = len(genres_sorted)

counts_matrix = np.array(
    [[df["genre_top"].value_counts().get(g, 0) for df in dfs] for g in genres_sorted],
    dtype=int,
)

row_max   = counts_matrix.max(axis=1).astype(float)
row_min   = np.where(counts_matrix.min(axis=1) == 0, 1, counts_matrix.min(axis=1)).astype(float)
imbalance = row_max / row_min

fig, (ax_bar, ax_imb) = plt.subplots(
    2, 1,
    figsize=(max(12, n_genres * 0.9), 9),
    gridspec_kw={"height_ratios": [3, 1.2]},
)

x       = np.arange(n_genres)
n_grps  = len(splits)
width   = 0.25
offsets = np.linspace(-(n_grps - 1) / 2, (n_grps - 1) / 2, n_grps) * width

for j, (split, color, offset) in enumerate(zip(splits, split_colors, offsets)):
    counts = counts_matrix[:, j]
    bars   = ax_bar.bar(x + offset, counts, width=width,
                        label=f"{split}  (n={counts.sum():,})",
                        color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(cnt),
                ha="center", va="bottom", fontsize=6.5, color="#333333",
            )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(genres_sorted, rotation=35, ha="right", fontsize=9)
ax_bar.set_ylabel("Sample count", fontsize=9)
ax_bar.set_title(f"Genre x split counts -- fma_{SUBSET}", fontsize=11, pad=10)
ax_bar.legend(fontsize=9, loc="upper right")
for spine in ["top", "right"]:
    ax_bar.spines[spine].set_visible(False)
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_bar.set_axisbelow(True)

bar_colors = ["#d16d1b" if r > 1.5 else "#159dd3" for r in imbalance]
ax_imb.bar(x, imbalance, color=bar_colors, width=0.55, edgecolor="none", alpha=0.85)
ax_imb.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
ax_imb.set_xticks(x)
ax_imb.set_xticklabels(genres_sorted, rotation=35, ha="right", fontsize=9)
ax_imb.set_ylabel("Imbalance ratio (max/min)", fontsize=8)
ax_imb.set_title("Split balance per genre", fontsize=10, pad=6)
for i, (xi, r) in enumerate(zip(x, imbalance)):
    ax_imb.text(xi, r + 0.03, f"{r:.1f}x", ha="center", va="bottom",
                fontsize=7, color="#d16d1b" if r > 1.5 else "#159dd3")
for spine in ["top", "right"]:
    ax_imb.spines[spine].set_visible(False)
ax_imb.yaxis.grid(True, linestyle="--", alpha=0.4)
ax_imb.set_axisbelow(True)

plt.tight_layout()
plt.show()

worst_genre = genres_sorted[int(np.argmax(imbalance))]
worst_ratio = float(imbalance.max())
print(f"Most imbalanced genre across splits : {worst_genre}  (ratio = {worst_ratio:.2f}x)")
if worst_ratio <= 1.05:
    print("-> All genres are near-perfectly balanced across splits.")
elif worst_ratio <= 2.0:
    print("-> Minor imbalance; using class-weighted loss is optional.")
else:
    print("-> Significant imbalance detected -- consider class-weighted loss or reporting Macro-F1 as the primary metric.")

_section_times["3b. Genre plot"] = _time_module.perf_counter() - _t0
print()
print(f"Genre distribution plot : {_section_times['3b. Genre plot']:.2f}s")

# %% [markdown]
# ## 4. Log-Mel Feature Extraction
#
# Each original audio clip is first **normalized to exactly 10 seconds** using
# deterministic center-crop (or center-pad for clips shorter than 10 s).
# The 10-second waveform is then transformed into a 128-band log-mel spectrogram
# with `n_fft=512`, `hop_length=256`, yielding a `(128, 861)` matrix.
#
# ### Waveform normalization pipeline (per track)
#
# 1. **Load** full audio clip (no duration cap)
# 2. **Sanity check** (non-empty, finite, minimum length)
# 3. **`normalize_to_fixed_duration()`** — center-crop if > 10 s, center-pad if < 10 s
# 4. **Extract** log-mel spectrogram → fixed `(128, 861)` shape
# 5. **Cache** as `.npy` in `logmel_10s/` directory
#
# ### Corrupt or unreadable tracks
#
# Some tracks in each split may be silently dropped:
#
# * source MP3 cannot be decoded (corrupt file or truncated download)
# * decoded audio is too short, empty, or has non-finite samples
# * a previously cached `.npy` has the wrong shape or is unreadable
#
# If corrupt `.npy` files are found after loading a cached index, they are deleted and
# **re-extracted in the same run** (not deferred to the next execution).
# The final "Usable rows" printout at the end of this section shows how many tracks
# from each split actually reach the model — compare against the manifest totals to
# detect any data-quality losses before interpreting results.

# %%
import librosa
import subprocess
import concurrent.futures as _fut


def _track_id_from_path(filepath: Path) -> int:
    try:
        return int(Path(filepath).stem)
    except Exception:
        return abs(hash(str(filepath))) % (10**12)


def _load_audio_ffmpeg(filepath: Path, sr: int, mono: bool = True, duration: float = None) -> np.ndarray:
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


def _load_audio_simple(filepath: Path, sr: int, mono: bool = True, duration: float = None) -> np.ndarray:
    if AUDIO_BACKEND == "ffmpeg":
        return _load_audio_ffmpeg(filepath, sr=sr, mono=mono, duration=duration)
    y, _sr = librosa.load(str(filepath), sr=sr, mono=mono, duration=duration)
    return y.astype(np.float32, copy=False)


def normalize_to_fixed_duration(y: np.ndarray, sr: int, target_sec: float) -> np.ndarray:
    """Normalize waveform to exactly target_sec seconds via center-crop or center-pad.

    Strategy 1 from the development guideline:
    - If clip > target_sec: center-crop around midpoint
    - If clip < target_sec: center-pad with silence (zeros)
    - If clip == target_sec: return unchanged
    """
    target_len = int(round(target_sec * sr))
    n = len(y)

    if n == target_len:
        return y

    if n > target_len:
        mid = n // 2
        half = target_len // 2
        start = mid - half
        end = start + target_len
        # Defensive clamping
        if start < 0:
            start = 0
            end = target_len
        if end > n:
            end = n
            start = n - target_len
        return y[start:end]

    # n < target_len → center-pad with silence
    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant").astype(np.float32)


def _sanity_check_audio(y: np.ndarray, sr: int) -> tuple[bool, str]:
    if y is None or len(y) == 0:
        return False, "empty_audio"
    if not np.isfinite(y).all():
        return False, "non_finite_samples"
    if len(y) < int(MIN_SECONDS * sr):
        return False, f"too_short(<{MIN_SECONDS}s)"
    if SKIP_SILENT:
        peak = float(np.max(np.abs(y)))
        st   = float(np.std(y))
        if peak < SILENCE_PEAK or st < SILENCE_STD:
            return False, "near_silent"
    return True, ""


def _logmel_fixed_shape(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    logmel = np.log1p(S)
    out = np.zeros(LOGMEL_SHAPE, dtype=np.float32)
    n = min(logmel.shape[1], N_FRAMES)
    out[:, :n] = logmel[:, :n].astype(np.float32, copy=False)
    return out


def _is_valid_npy(path: Path, expected_shape: tuple) -> bool:
    """Return True only if the .npy file loads cleanly and has the expected shape."""
    try:
        arr = np.load(str(path), mmap_mode='r')
        return arr.shape == expected_shape
    except Exception:
        return False


def _process_one_track(task: tuple) -> dict:
    filepath_str, split_name, genre_top, track_id, logmel_cache_dir = task
    filepath = Path(filepath_str)
    logmel_path = Path(logmel_cache_dir) / f"{track_id}.npy"

    _corrupt_deleted = False

    if logmel_path.exists():
        if _is_valid_npy(logmel_path, LOGMEL_SHAPE):
            return {
                "track_id": track_id,
                "filepath": filepath_str,
                "split": split_name,
                "genre_top": genre_top,
                "logmel_path": str(logmel_path),
                "status": "cached",
                "reason": "",
                "corrupt_deleted": False,
            }
        # Corrupt or wrong-shape file — log, delete, then fall through to re-extract
        print(f"[WARN] Corrupt/invalid .npy deleted, will re-extract: {logmel_path}", flush=True)
        _corrupt_deleted = True
        try:
            logmel_path.unlink()
        except Exception:
            pass

    # Load full audio (no duration cap) so we can compute the midpoint
    try:
        y = _load_audio_simple(filepath, sr=SAMPLE_RATE, mono=True, duration=None)
    except Exception as exc:
        return {
            "track_id": track_id,
            "filepath": filepath_str,
            "split": split_name,
            "genre_top": genre_top,
            "logmel_path": "",
            "status": "skipped",
            "reason": f"decode_fail:{type(exc).__name__}",
            "corrupt_deleted": _corrupt_deleted,
        }

    ok, reason = _sanity_check_audio(y, SAMPLE_RATE)
    if not ok:
        return {
            "track_id": track_id,
            "filepath": filepath_str,
            "split": split_name,
            "genre_top": genre_top,
            "logmel_path": "",
            "status": "skipped",
            "reason": reason,
            "corrupt_deleted": _corrupt_deleted,
        }

    # Normalize to exactly CLIP_DURATION seconds (center-crop / center-pad)
    y = normalize_to_fixed_duration(y, SAMPLE_RATE, CLIP_DURATION)

    try:
        logmel = _logmel_fixed_shape(y)
        logmel_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(logmel_path, logmel)
        return {
            "track_id": track_id,
            "filepath": filepath_str,
            "split": split_name,
            "genre_top": genre_top,
            "logmel_path": str(logmel_path),
            "status": "ok",
            "reason": "",
            "corrupt_deleted": _corrupt_deleted,
        }
    except Exception as exc:
        return {
            "track_id": track_id,
            "filepath": filepath_str,
            "split": split_name,
            "genre_top": genre_top,
            "logmel_path": "",
            "status": "skipped",
            "reason": f"logmel_fail:{type(exc).__name__}",
            "corrupt_deleted": _corrupt_deleted,
        }


def build_logmel_index(
    split_df: pd.DataFrame,
    split_name: str,
    cache_dir: Path,
    num_workers: int,
    clear_cache: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = CACHE_DIR / f"logmel_v20a1_10s_index_{split_name}_{SUBSET}.parquet"

    if clear_cache:
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        if index_path.exists():
            index_path.unlink()

    if index_path.exists():
        print(f"  [{split_name}] Loading log-mel index from cache ...")
        return pd.read_parquet(index_path)

    print(f"  [{split_name}] Building log-mel cache with {num_workers} workers ...")
    tasks = []
    for row in split_df.itertuples(index=False):
        fp = getattr(row, "filepath") if hasattr(row, "filepath") else row[split_df.columns.get_loc("filepath")]
        gt = getattr(row, "genre_top") if hasattr(row, "genre_top") else row[split_df.columns.get_loc("genre_top")]
        tid = _track_id_from_path(fp)
        tasks.append((str(fp), split_name, str(gt), int(tid), str(cache_dir)))

    results = []
    skipped = 0
    t0 = time.time()

    with _fut.ProcessPoolExecutor(max_workers=num_workers) as ex:
        for i, res in enumerate(ex.map(_process_one_track, tasks, chunksize=32), start=1):
            if res.get("corrupt_deleted"):
                print(f"  [WARN] Corrupt cache re-extracted: {Path(res['filepath']).name}  (track_id={res['track_id']})")
            results.append(res)
            if res["status"] == "skipped":
                skipped += 1
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"    {i}/{len(tasks)}  skipped={skipped}  -- {elapsed:.0f}s elapsed")

    index_df = pd.DataFrame(results)
    index_df.to_parquet(index_path, index=False)
    print(f"    Saved index -> {index_path}")
    return index_df

# %%

_t0 = _time_module.perf_counter()

print("Building/loading log-mel per-track cache + index parquets ...")
print(f"  clip duration = {CLIP_DURATION}s  |  cache dir = {LOGMEL_CACHE_DIR}")
print()

train_index = build_logmel_index(train_df, "training",   LOGMEL_CACHE_DIR, NUM_WORKERS, clear_cache=CLEAR_CACHE)
val_index   = build_logmel_index(val_df,   "validation", LOGMEL_CACHE_DIR, NUM_WORKERS, clear_cache=CLEAR_CACHE)
test_index  = build_logmel_index(test_df,  "test",       LOGMEL_CACHE_DIR, NUM_WORKERS, clear_cache=CLEAR_CACHE)

def _usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].isin(["ok", "cached"])].reset_index(drop=True)

def _purge_corrupt(index_df: pd.DataFrame, expected_shape: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check each cached .npy; delete bad files.
    Returns (clean_df, corrupt_df) — corrupt_df rows need re-extraction."""
    ok_mask = np.array([_is_valid_npy(Path(p), expected_shape) for p in index_df["logmel_path"]])
    n_corrupt = int((~ok_mask).sum())
    for p in index_df.loc[~ok_mask, "logmel_path"]:
        print(f"  [WARN] Corrupt/missing — will re-extract: {p}")
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass
    if n_corrupt:
        print(f"  Found {n_corrupt} corrupt/missing entries — re-extracting now ...")
    return index_df[ok_mask].reset_index(drop=True), index_df[~ok_mask].reset_index(drop=True)


def _reextract_corrupt(
    corrupt_df: pd.DataFrame,
    full_index_df: pd.DataFrame,
    index_path: Path,
) -> pd.DataFrame:
    """Re-extract tracks that were purged, update the index parquet, return repaired usable rows."""
    if len(corrupt_df) == 0:
        return corrupt_df
    tasks = [
        (row["filepath"], row["split"], row["genre_top"], int(row["track_id"]), str(LOGMEL_CACHE_DIR))
        for _, row in corrupt_df.iterrows()
    ]
    results, n_ok, n_skip = [], 0, 0
    with _fut.ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for i, res in enumerate(ex.map(_process_one_track, tasks, chunksize=16), start=1):
            results.append(res)
            if res["status"] in ("ok", "cached"):
                n_ok += 1
            else:
                n_skip += 1
            if i % 50 == 0 or i == len(tasks):
                print(f"    re-extract {i}/{len(tasks)}  ok={n_ok}  skipped={n_skip}")
    print(f"  Re-extraction complete: {n_ok} ok, {n_skip} permanently skipped")
    repaired_df = pd.DataFrame(results)
    # Persist updated index — replace old corrupt rows with fresh results
    updated_full = (
        pd.concat([full_index_df, repaired_df])
        .drop_duplicates(subset=["track_id"], keep="last")
        .reset_index(drop=True)
    )
    updated_full.to_parquet(index_path, index=False)
    return repaired_df[repaired_df["status"].isin(["ok", "cached"])].reset_index(drop=True)


def _purge_and_repair(full_index: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Purge corrupt cache entries and immediately re-extract them in the same run."""
    index_path = CACHE_DIR / f"logmel_v20a1_10s_index_{split_name}_{SUBSET}.parquet"
    clean_df, corrupt_df = _purge_corrupt(_usable(full_index), LOGMEL_SHAPE)
    repaired_df = _reextract_corrupt(corrupt_df, full_index, index_path)
    return pd.concat([clean_df, repaired_df]).reset_index(drop=True)


train_index_u = _purge_and_repair(train_index, "training")
val_index_u   = _purge_and_repair(val_index,   "validation")
test_index_u  = _purge_and_repair(test_index,  "test")

print("Usable rows:")
print(f"  train: {len(train_index_u):,} / {len(train_index):,}")
print(f"  val  : {len(val_index_u):,} / {len(val_index):,}")
print(f"  test : {len(test_index_u):,} / {len(test_index):,}")

_section_times["4. Log-mel extraction"] = _time_module.perf_counter() - _t0
print()
print(f"Log-mel cache+index : {_section_times['4. Log-mel extraction']:.2f}s")

# %%
_t0 = _time_module.perf_counter()

idx_row = 0
row = train_index_u.iloc[idx_row]
logmel = np.load(row["logmel_path"])
genre = row["genre_top"]
tid = row["track_id"]

fig, ax = plt.subplots(figsize=(14, 5))
img = ax.imshow(logmel, aspect="auto", origin="lower",
                extent=[0, N_FRAMES * HOP_LENGTH / SAMPLE_RATE, 0, N_MELS],
                cmap="magma")
ax.set_title(f"Sample log-mel spectrogram - genre: {genre}  (track_id={tid})")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mel band")
plt.colorbar(img, ax=ax, label="log(1 + power)")
plt.tight_layout()
plt.show()

_section_times["4b. Log-mel plot"] = _time_module.perf_counter() - _t0
print(f"Log-mel sample plot : {_section_times['4b. Log-mel plot']:.2f}s")

# %% [markdown]
# ## 5. Preprocessing

# %%
_t0 = _time_module.perf_counter()

# -- Label encoding ------------------------------------------------------------
train_index_u["label_int"] = label_enc.transform(train_index_u["genre_top"].to_numpy())
val_index_u["label_int"]   = label_enc.transform(val_index_u["genre_top"].to_numpy())
test_index_u["label_int"]  = label_enc.transform(test_index_u["genre_top"].to_numpy())

# -- Class weights (handle genre imbalance) ------------------------------------
from sklearn.utils.class_weight import compute_class_weight

_train_labels_int = train_index_u["label_int"].to_numpy()
_class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(N_CLASSES),
    y=_train_labels_int,
)
class_weight_dict = {i: float(w) for i, w in enumerate(_class_weights)}
print("Class weights:")
for i, g in enumerate(GENRE_CLASSES):
    print(f"  {g:<20s}  {class_weight_dict[i]:.4f}")

# -- Compute per-band mean/std over TRAIN (streaming, no big array in RAM) -----
def compute_train_stats(index_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sum_c   = np.zeros((N_MELS,), dtype=np.float64)
    sumsq_c = np.zeros((N_MELS,), dtype=np.float64)
    count = 0

    for i, p in enumerate(index_df["logmel_path"]):
        x = np.load(p)
        sum_c   += x.sum(axis=1)
        sumsq_c += (x * x).sum(axis=1)
        count   += x.shape[1]
        if (i + 1) % 2000 == 0:
            print(f"  stats pass: {i+1}/{len(index_df)}")

    mean = sum_c / max(1, count)
    var  = (sumsq_c / max(1, count)) - mean**2
    std  = np.sqrt(np.maximum(var, 1e-12))

    mu  = mean.reshape((1, N_MELS, 1, 1)).astype(np.float32)
    sig = std.reshape((1, N_MELS, 1, 1)).astype(np.float32)
    return mu, sig

print("\nComputing training mean/std (streaming) ...")
mu, std = compute_train_stats(train_index_u)
print(f"mu shape={mu.shape}, std shape={std.shape}")

# -- tf.data input pipeline ----------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
mu_tf  = tf.constant(mu, dtype=tf.float32)
std_tf = tf.constant(std, dtype=tf.float32)

def _np_load_logmel(path_bytes):
    path = path_bytes.decode("utf-8")
    x = np.load(path).astype(np.float32, copy=False)
    x = x[..., np.newaxis]
    return x

# -- SpecAugment (training-only augmentation) ----------------------------------
def spec_augment(x, freq_mask=SPEC_AUG_FREQ_MASK, time_mask=SPEC_AUG_TIME_MASK,
                 num_masks=SPEC_AUG_NUM_MASKS):
    """SpecAugment: random frequency and time masking on a (N_MELS, N_FRAMES, 1) tensor."""
    shape = tf.shape(x)
    freq_dim = shape[0]   # N_MELS = 128
    time_dim = shape[1]   # N_FRAMES = 861

    for _ in range(num_masks):
        # Frequency mask — clamp mask size to actual dimension so config changes
        # (e.g. reducing N_MELS) never produce invalid upper bounds for uniform sampling.
        f_max = tf.minimum(freq_mask, freq_dim)
        f  = tf.random.uniform([], 0, f_max + 1, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, tf.maximum(freq_dim - f, 1), dtype=tf.int32)
        freq_mask_tensor = tf.concat([
            tf.ones([f0, time_dim, 1]),
            tf.zeros([f, time_dim, 1]),
            tf.ones([freq_dim - f0 - f, time_dim, 1]),
        ], axis=0)
        x = x * freq_mask_tensor

        # Time mask — same clamping for time axis.
        t_max = tf.minimum(time_mask, time_dim)
        t  = tf.random.uniform([], 0, t_max + 1, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, tf.maximum(time_dim - t, 1), dtype=tf.int32)
        time_mask_tensor = tf.concat([
            tf.ones([freq_dim, t0, 1]),
            tf.zeros([freq_dim, t, 1]),
            tf.ones([freq_dim, time_dim - t0 - t, 1]),
        ], axis=1)
        x = x * time_mask_tensor

    return x

def make_dataset(index_df: pd.DataFrame, batch_size: int, shuffle: bool, augment: bool = False) -> tf.data.Dataset:
    paths  = index_df["logmel_path"].to_numpy(dtype=str)
    labels = index_df["label_int"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 10_000), seed=SEED, reshuffle_each_iteration=True)

    def _load_and_norm(path, y):
        x = tf.numpy_function(_np_load_logmel, [path], Tout=tf.float32)
        x.set_shape((*LOGMEL_SHAPE, 1))
        x = (x - mu_tf[0]) / std_tf[0]
        y_oh = tf.one_hot(y, N_CLASSES)
        return x, y_oh

    ds = ds.map(_load_and_norm, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(lambda x, y: (spec_augment(x), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_index_u, BATCH_SIZE, shuffle=True, augment=True)
val_ds   = make_dataset(val_index_u,   BATCH_SIZE, shuffle=False, augment=False)
test_ds  = make_dataset(test_index_u,  BATCH_SIZE, shuffle=False, augment=False)

print("\nDatasets ready:")
print(f"  train batches: {tf.data.experimental.cardinality(train_ds).numpy()}  (SpecAugment=ON)")
print(f"  val   batches: {tf.data.experimental.cardinality(val_ds).numpy()}  (SpecAugment=OFF)")
print(f"  test  batches: {tf.data.experimental.cardinality(test_ds).numpy()}  (SpecAugment=OFF)")

_section_times["5. Preprocessing"] = _time_module.perf_counter() - _t0
print(f"\nPreprocessing : {_section_times['5. Preprocessing']:.2f}s")

# %% [markdown]
# ## 6. Build the CNN Model
#
# Deeper v20a1 architecture with 5 conv blocks, proper BN ordering
# (Conv→BN→ReLU), SpatialDropout2D, and increased filter capacity.
#
# | Block | Layer | Kernel | Pool | Output shape (128×861 input) |
# |-------|-------|--------|------|-------------------------------|
# | 1 | Conv2D(32) + BN + ReLU + MaxPool(2,2) | (5,5) | (2,2) | (64, 430, 32) |
# | 2 | Conv2D(64) + BN + ReLU + SpatialDrop(0.10) + MaxPool(2,2) | (3,3) | (2,2) | (32, 215, 64) |
# | 3 | Conv2D(128) + BN + ReLU + SpatialDrop(0.10) + MaxPool(2,2) | (3,3) | (2,2) | (16, 107, 128) |
# | 4 | Conv2D(256) + BN + ReLU + MaxPool(2,2) | (3,3) | (2,2) | (8, 53, 256) |
# | 5 | Conv2D(256) + BN + ReLU + MaxPool(2,2) | (3,3) | (2,2) | (4, 26, 256) |
# | Head | GAP + Dropout(0.2) + Dense | — | — | (N_CLASSES,) |

# %%
_t0 = _time_module.perf_counter()

def build_model(n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*LOGMEL_SHAPE, 1), name="logmel")

    # Block 1 — local spectro-temporal features
    x = layers.Conv2D(32, (5, 5), padding="same", use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPool2D((2, 2), name="pool1")(x)

    # Block 2 — mid-level motifs
    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.SpatialDropout2D(0.10, name="sdrop2")(x)
    x = layers.MaxPool2D((2, 2), name="pool2")(x)

    # Block 3 — higher-level patterns
    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.SpatialDropout2D(0.10, name="sdrop3")(x)
    x = layers.MaxPool2D((2, 2), name="pool3")(x)

    # Block 4 — global structure (256 filters)
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPool2D((2, 2), name="pool4")(x)

    # Block 5 — deeper abstraction (256 filters)
    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.ReLU(name="relu5")(x)
    x = layers.MaxPool2D((2, 2), name="pool5")(x)

    # Classifier head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="fc_out")(x)

    return keras.Model(inputs, outputs, name="logmel_2dcnn_v20a1")

with tf.device(RUNTIME_DEVICE):
    model = build_model(N_CLASSES)

model.summary()

_section_times["6. Build model"] = _time_module.perf_counter() - _t0
print(f"\nBuild model : {_section_times['6. Build model']:.2f}s")

# %% [markdown]
# ## 7. Compile & Train

# %%
import datetime as _dt
import json as _json

_t0 = _time_module.perf_counter()

_run_ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = MODELS_BASE_DIR / f"logmel-cnn-v20a1-{_run_ts}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run directory : {RUN_DIR}")

# ── Cosine annealing LR schedule with warmup ─────────────────────────────────
WARMUP_EPOCHS = 3
LR_MAX        = 1e-3
LR_MIN        = 1e-6

# Estimate steps_per_epoch from training dataset
_steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
if _steps_per_epoch <= 0:
    _steps_per_epoch = len(train_index_u) // BATCH_SIZE

_total_steps  = EPOCHS * _steps_per_epoch
_warmup_steps = WARMUP_EPOCHS * _steps_per_epoch

class CosineAnnealingWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup followed by cosine decay to min_lr."""
    def __init__(self, warmup_steps, total_steps, lr_max, lr_min):
        super().__init__()
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.lr_max       = lr_max
        self.lr_min       = lr_min

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        # Warmup phase
        warmup_lr = self.lr_min + (self.lr_max - self.lr_min) * (step / tf.maximum(self.warmup_steps, 1.0))
        # Cosine decay phase
        progress = (step - self.warmup_steps) / tf.maximum(self.total_steps - self.warmup_steps, 1.0)
        cosine_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + tf.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {"warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
                "lr_max": self.lr_max, "lr_min": self.lr_min}

_lr_schedule = CosineAnnealingWithWarmup(_warmup_steps, _total_steps, LR_MAX, LR_MIN)

with tf.device(RUNTIME_DEVICE):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=_lr_schedule, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )

# ── Custom MacroF1Checkpoint — saves model when val macro-F1 improves ────────
class _MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Compute val macro-F1 at end of every ``check_freq`` epochs; save model when it improves.

    EarlyStopping monitors val_loss (stable stopping signal).
    This callback monitors val_macro_f1 (genre-balanced performance goal).
    The two objectives can diverge in later epochs when label smoothing and
    class weights cause val_loss to move independently of classification accuracy.

    ``check_freq`` controls how often the extra validation pass runs:
      - check_freq=1  : every epoch (default; most accurate F1 curve, highest cost)
      - check_freq=2+ : every N epochs (halves/reduces extra validation overhead;
                        f1_history repeats the last measured value on skipped epochs
                        so the list stays epoch-aligned for plotting)
    """
    def __init__(self, val_ds: tf.data.Dataset, genre_classes: list, filepath: str,
                 check_freq: int = 1):
        super().__init__()
        self.val_ds        = val_ds
        self.genre_classes = genre_classes
        self.filepath      = filepath
        self.check_freq    = max(1, check_freq)
        self.best_f1       = -1.0
        self.f1_history    = []

    def on_epoch_end(self, epoch, logs=None):
        # On skipped epochs, repeat the last known F1 so history stays aligned.
        if (epoch + 1) % self.check_freq != 0:
            last = self.f1_history[-1] if self.f1_history else 0.0
            self.f1_history.append(last)
            return

        y_true, y_pred = [], []
        for xb, yb in self.val_ds:
            preds = self.model(xb, training=False).numpy()
            y_pred.append(np.argmax(preds, axis=1))
            y_true.append(np.argmax(yb.numpy(), axis=1))
        macro_f1 = float(f1_score(
            np.concatenate(y_true), np.concatenate(y_pred),
            average="macro", zero_division=0,
        ))
        self.f1_history.append(macro_f1)
        if logs is not None:
            logs["val_macro_f1"] = macro_f1
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.model.save(self.filepath)
            print(f"\n  [MacroF1Checkpoint] epoch {epoch+1}: val_macro_f1 improved to {macro_f1:.4f} → saved")

# ── Custom callbacks for plot data (LR schedule + per-epoch timing) ─────────
class _LRLogger(tf.keras.callbacks.Callback):
    """Record the optimizer learning rate at the end of every epoch."""
    def __init__(self):
        super().__init__()
        self.lrs = []
    def on_epoch_end(self, epoch, logs=None):
        opt = self.model.optimizer
        lr_attr = opt.learning_rate          # always present in TF2/Keras
        if callable(lr_attr):                # LRSchedule or similar callable
            lr = float(lr_attr(opt.iterations))
        else:                                # tf.Variable or plain Python float
            lr = float(lr_attr)
        self.lrs.append(lr)

class _EpochTimer(tf.keras.callbacks.Callback):
    """Record wall-clock seconds for every epoch."""
    def __init__(self):
        super().__init__()
        self.times = []
        self._t0 = None
    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = _time_module.perf_counter()
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(round(_time_module.perf_counter() - self._t0, 3))

_lr_logger    = _LRLogger()
_epoch_timer  = _EpochTimer()
_f1_ckpt      = _MacroF1Checkpoint(
    val_ds=val_ds,
    genre_classes=GENRE_CLASSES,
    filepath=str(RUN_DIR / "best_model_macro_f1.keras"),
    check_freq=2,
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=9, restore_best_weights=True, verbose=1,
    ),
    _f1_ckpt,
    _lr_logger,
    _epoch_timer,
]

print(f"Training on {RUNTIME_DEVICE}  |  epochs={EPOCHS}, batch_size={BATCH_SIZE}")
print(f"Optimizer: AdamW(cosine_annealing, warmup={WARMUP_EPOCHS}ep, lr_max={LR_MAX}, weight_decay={WEIGHT_DECAY})")
print(f"Loss: CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING})")
print(f"Callbacks: EarlyStopping(val_loss, patience=9), MacroF1Checkpoint(best_macro_f1), CosineAnnealing\n")

with tf.device(RUNTIME_DEVICE):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

_n_ep = len(history.history["loss"])
best_loss = min(range(_n_ep), key=lambda i: history.history["val_loss"][i])
best_f1   = int(np.argmax(_f1_ckpt.f1_history)) if _f1_ckpt.f1_history else best_loss
# Use best macro-F1 epoch as canonical "best" for reporting
best = best_f1
print(f"\nBest epoch by val_loss   : {best_loss + 1} / {_n_ep}  (val_loss={history.history['val_loss'][best_loss]:.4f})")
print(f"Best epoch by macro_F1   : {best_f1 + 1} / {_n_ep}  (val_macro_f1={_f1_ckpt.f1_history[best_f1]:.4f})  ← primary metric")
print(f"\nMetrics at best macro-F1 epoch ({best_f1 + 1}):")
print(f"  {'val_macro_f1':<16}: {_f1_ckpt.f1_history[best_f1]:.4f}  ← primary (genre-balanced)")
for k in ["val_accuracy", "val_loss", "accuracy", "loss"]:
    note = "  ← secondary (inspect alongside Macro-F1 and per-genre F1)" if k == "val_accuracy" else ""
    print(f"  {k:<16}: {history.history[k][best]:.4f}{note}")

_section_times["7. Compile & train"] = _time_module.perf_counter() - _t0
print(f"\nCompile & train : {_section_times['7. Compile & train']:.1f}s  ({_section_times['7. Compile & train']/_n_ep:.1f}s per epoch)")

model_path = RUN_DIR / "baseline_logmel_cnn_v20a1.keras"
model.save(str(model_path))
print(f"Model saved  -> {model_path}")

# ── Save normalization stats for standalone inference ──────────────────────────
norm_stats_path = RUN_DIR / "norm_stats.npz"
np.savez(str(norm_stats_path), mu=mu, std=std, genre_classes=np.array(GENRE_CLASSES))
print(f"Norm stats   -> {norm_stats_path}")

_hist    = history.history
_t_train = _section_times["7. Compile & train"]

_summary_lines = []
model.summary(print_fn=lambda l: _summary_lines.append(l))

_report = {
    "run_id":       f"logmel-cnn-v20a1-{_run_ts}",
    "subset":       SUBSET,
    "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "model_file":   model_path.name,
    "feature_type": "logmel",
    "version":      "v20a1",
    "config": {
        "device":        RUNTIME_DEVICE,
        "audio_backend": AUDIO_BACKEND,
        "num_workers":   NUM_WORKERS,
        "clip_duration_sec": CLIP_DURATION,
        "waveform_normalization": "center_crop_or_center_pad",
        "logmel_extraction": {
            "sample_rate":  SAMPLE_RATE,
            "n_mels":       N_MELS,
            "n_fft":        N_FFT,
            "hop_length":   HOP_LENGTH,
            "n_frames":     N_FRAMES,
            "logmel_shape": list(LOGMEL_SHAPE),
            "log_compress": "log1p",
            "cache_shared": LOGMEL_CACHE_SHARED,
        },
        "training": {
            "epochs_max":           EPOCHS,
            "epochs_actual":        _n_ep,
            "batch_size":           BATCH_SIZE,
            "optimizer":            "AdamW",
            "lr_schedule":          "cosine_annealing_with_warmup",
            "lr_max":               LR_MAX,
            "lr_min":               LR_MIN,
            "warmup_epochs":        WARMUP_EPOCHS,
            "weight_decay":         WEIGHT_DECAY,
            "loss":                 "CategoricalCrossentropy",
            "label_smoothing":      LABEL_SMOOTHING,
            "class_weights":        class_weight_dict,
            "early_stopping":       {"monitor": "val_loss", "patience": 9, "restore_best_weights": True},
            "macro_f1_checkpoint":   {"monitor": "val_macro_f1", "save_best_only": True, "filepath": "best_model_macro_f1.keras"},
            "spec_augment": {
                "freq_mask": SPEC_AUG_FREQ_MASK,
                "time_mask": SPEC_AUG_TIME_MASK,
                "num_masks": SPEC_AUG_NUM_MASKS,
            },
        },
        "architecture_changes": "v20a1_deeper_5block_proper_bn_spatial_dropout",
    },
    "dataset": {
        "n_classes":     N_CLASSES,
        "genres":        GENRE_CLASSES,
        "train_samples": len(train_index_u),
        "val_samples":   len(val_index_u),
        "test_samples":  len(test_index_u),
    },
    "model_architecture": {
        "name":         model.name,
        "total_params": model.count_params(),
        "summary":      "\n".join(_summary_lines),
    },
    "training_history": {
        "epochs": [
            {
                "epoch":          i + 1,
                "loss":           float(_hist["loss"][i]),
                "accuracy":       float(_hist["accuracy"][i]),
                "val_loss":       float(_hist["val_loss"][i]),
                "val_accuracy":   float(_hist["val_accuracy"][i]),
                "val_macro_f1":   round(float(_f1_ckpt.f1_history[i]), 4) if i < len(_f1_ckpt.f1_history) else None,
            }
            for i in range(_n_ep)
        ],
        "best_epoch_val_loss": {
            "epoch":          best_loss + 1,
            "loss":           float(_hist["loss"][best_loss]),
            "accuracy":       float(_hist["accuracy"][best_loss]),
            "val_loss":       float(_hist["val_loss"][best_loss]),
            "val_accuracy":   float(_hist["val_accuracy"][best_loss]),
            "val_macro_f1":   round(float(_f1_ckpt.f1_history[best_loss]), 4) if best_loss < len(_f1_ckpt.f1_history) else None,
        },
        "best_epoch_macro_f1": {
            "epoch":          best_f1 + 1,
            "loss":           float(_hist["loss"][best_f1]),
            "accuracy":       float(_hist["accuracy"][best_f1]),
            "val_loss":       float(_hist["val_loss"][best_f1]),
            "val_accuracy":   float(_hist["val_accuracy"][best_f1]),
            "val_macro_f1":   round(float(_f1_ckpt.f1_history[best_f1]), 4) if best_f1 < len(_f1_ckpt.f1_history) else None,
        },
        "timing_seconds":    round(_t_train, 2),
        "seconds_per_epoch": round(_t_train / _n_ep, 2),
    },
    "evaluation": None,
}

REPORT_PATH = RUN_DIR / f"run_report_{SUBSET}.json"
REPORT_PATH.write_text(_json.dumps(_report, indent=2))
print(f"Run report   -> {REPORT_PATH}")

# ── Plot data file — collects all data needed to recreate every plot ───────────
# Populated incrementally: training history here, confusion matrices in Section 9,
# inference probabilities in Section 10.  Written to disk after each section.
PLOT_DATA_PATH = RUN_DIR / f"plot_data_{SUBSET}.json"
_plot_data = {
    "genre_classes": GENRE_CLASSES,
    "n_classes":     N_CLASSES,
    "training_history": {
        "epochs":           list(range(1, _n_ep + 1)),
        "loss":             [float(v) for v in _hist["loss"]],
        "accuracy":         [float(v) for v in _hist["accuracy"]],
        "val_loss":         [float(v) for v in _hist["val_loss"]],
        "val_accuracy":     [float(v) for v in _hist["val_accuracy"]],
        "val_macro_f1":     [round(float(v), 4) for v in _f1_ckpt.f1_history],
        "best_epoch_loss":  best_loss + 1,
        "best_epoch_f1":    best_f1 + 1,
        "best_epoch":       best_f1 + 1,  # canonical best = macro-F1 epoch
        # Plot 1 — LR schedule overlay
        "lr_per_epoch":     _lr_logger.lrs,
        # Plot 2 — train-val loss gap (derivable, but pre-computed for convenience)
        "loss_gap":         [round(float(vl - tl), 6)
                             for tl, vl in zip(_hist["loss"], _hist["val_loss"])],
        # Plot 3 — per-epoch wall-clock timing
        "seconds_per_epoch": _epoch_timer.times,
    },
    # Plot 8 — class distribution per split (sample count per genre)
    "class_distribution": {
        split: {g: int(df["genre_top"].value_counts().get(g, 0))
                for g in GENRE_CLASSES}
        for split, df in [("train", train_df), ("validation", val_df), ("test", test_df)]
    },
    "confusion_matrices":   None,  # filled in by Section 9b
    "per_genre_metrics":    None,  # filled in by Section 9
    "confidence_histogram": None,  # filled in by Section 9
    "inference":            None,  # filled in by Section 10
}
PLOT_DATA_PATH.write_text(_json.dumps(_plot_data, indent=2))
print(f"Plot data    -> {PLOT_DATA_PATH}")

# %% [markdown]
# ## 8. Training History

# %%
_t0 = _time_module.perf_counter()

hist = history.history
epochs_range = range(1, len(hist["accuracy"]) + 1)

fig, (ax_acc, ax_loss, ax_f1) = plt.subplots(1, 3, figsize=(18, 4))

ax_acc.plot(epochs_range, hist["accuracy"],     label="Train",      linewidth=2)
ax_acc.plot(epochs_range, hist["val_accuracy"], label="Validation", linewidth=2, linestyle="--")
ax_acc.set_title("Accuracy (secondary — interpret alongside Macro-F1)")
ax_acc.set_xlabel("Epoch")
ax_acc.set_ylabel("Accuracy")
ax_acc.legend(fontsize=8)
ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
for spine in ["top", "right"]:
    ax_acc.spines[spine].set_visible(False)

ax_loss.plot(epochs_range, hist["loss"],     label="Train",      linewidth=2)
ax_loss.plot(epochs_range, hist["val_loss"], label="Validation", linewidth=2, linestyle="--")
ax_loss.set_title("Loss")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Categorical cross-entropy")
ax_loss.legend()
ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
for spine in ["top", "right"]:
    ax_loss.spines[spine].set_visible(False)

# Macro-F1 per epoch (from _MacroF1Checkpoint)
if _f1_ckpt.f1_history:
    ax_f1.plot(epochs_range, _f1_ckpt.f1_history, label="Val Macro-F1", linewidth=2.5, color="darkorange")
    ax_f1.axvline(best_f1 + 1, color="darkorange", linestyle=":", linewidth=1.5,
                  label=f"Best F1 epoch ({best_f1+1}): {_f1_ckpt.best_f1:.4f}")
    if best_loss != best_f1:
        ax_f1.axvline(best_loss + 1, color="steelblue", linestyle=":", linewidth=1.2,
                      label=f"Best val_loss epoch ({best_loss+1})")
    ax_f1.set_title("Val Macro-F1 (PRIMARY metric)", fontweight="bold")
    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("Macro-F1")
    ax_f1.set_ylim(0, 1)
    ax_f1.legend(fontsize=8)
    ax_f1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for spine in ["top", "right"]:
        ax_f1.spines[spine].set_visible(False)

fig.suptitle("Training history -- Log-Mel 2D CNN v20a1 (10s center-crop)", fontsize=13)
plt.tight_layout()
plt.show()

if _f1_ckpt.f1_history:
    print(f"Best val macro-F1    : {_f1_ckpt.best_f1:.4f}  (epoch {best_f1 + 1})  ← primary metric")
final_val_acc = hist["val_accuracy"][-1]
print(f"Final val   accuracy : {final_val_acc:.4f}  ({final_val_acc:.2%})  ← secondary (interpret alongside Macro-F1 and per-genre F1)")
if _f1_ckpt.f1_history and best_loss != best_f1:
    print(f"\n[NOTE] best val_loss epoch ({best_loss+1}) ≠ best macro-F1 epoch ({best_f1+1}).")
    print( "  This is expected: label smoothing + class weights decouple loss from classification quality.")
    print( "  Use best_model_macro_f1.keras for deployment.")

_section_times["8. Training history plot"] = _time_module.perf_counter() - _t0
print(f"\nTraining history plot : {_section_times['8. Training history plot']:.2f}s")

# %% [markdown]
# ## 9. Evaluate the Model

# %%
import json as _json

_t0 = _time_module.perf_counter()


def eval_dataset(model, ds: tf.data.Dataset, genre_classes, split_label: str):
    # ── Cost (aggregate loss over the full split) ──────────────────────────────
    eval_results = model.evaluate(ds, verbose=0, return_dict=True)
    cost = float(eval_results.get("loss", float("nan")))

    # ── Predictions + per-sample confidence for histogram (Plot 7) ───────────
    y_true, y_pred, max_probs = [], [], []
    for xb, yb in ds:
        pred = model(xb, training=False).numpy()
        y_pred.append(np.argmax(pred, axis=1))
        y_true.append(np.argmax(yb.numpy(), axis=1))
        max_probs.append(np.max(pred, axis=1))
    y_true    = np.concatenate(y_true)    if y_true    else np.array([], dtype=np.int64)
    y_pred    = np.concatenate(y_pred)    if y_pred    else np.array([], dtype=np.int64)
    max_probs = np.concatenate(max_probs) if max_probs else np.array([], dtype=np.float32)
    correct_mask = (y_true == y_pred)

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cr_text  = classification_report(y_true, y_pred, target_names=genre_classes, zero_division=0)
    cr_dict  = classification_report(y_true, y_pred, target_names=genre_classes, zero_division=0, output_dict=True)
    block = (
        f"\n{'='*60}\n {split_label}\n{'='*60}\n"
        f"  Cost     : {cost:.4f}  (CategoricalCrossentropy, label_smoothing={LABEL_SMOOTHING})\n"
        f"  Accuracy : {acc:.4f}  ({acc:.2%})\n"
        f"  Macro-F1 : {macro_f1:.4f}\n\n"
        f"Per-genre classification report:\n{cr_text}"
    )
    print(block)

    per_genre = {
        g: {k: round(float(v), 4) for k, v in cr_dict[g].items()}
        for g in genre_classes
        if g in cr_dict
    }
    metrics = {
        "cost":      round(cost, 4),
        "accuracy":  round(float(acc), 4),
        "macro_f1":  round(float(macro_f1), 4),
        "per_genre": per_genre,
    }
    # Plot 7 — confidence histogram data
    confidence_data = {
        "correct":   [round(float(p), 6) for p in max_probs[correct_mask]],
        "incorrect": [round(float(p), 6) for p in max_probs[~correct_mask]],
    }
    return block, y_true, y_pred, metrics, confidence_data


# ── Load the best macro-F1 model for evaluation (not the EarlyStopping model) ─
# After training, the in-memory `model` holds the weights restored by
# EarlyStopping (best val_loss epoch).  Since our primary metric is Macro-F1,
# evaluation should use the checkpoint saved by _MacroF1Checkpoint.
_best_f1_path = RUN_DIR / "best_model_macro_f1.keras"
if _best_f1_path.exists():
    print(f"Loading best macro-F1 model for evaluation: {_best_f1_path}")
    _eval_model = tf.keras.models.load_model(str(_best_f1_path))
else:
    print("[WARN] best_model_macro_f1.keras not found — evaluating EarlyStopping-restored model instead.")
    _eval_model = model

with tf.device(RUNTIME_DEVICE):
    train_block, y_train_true, y_train_pred, train_metrics, train_conf = eval_dataset(_eval_model, train_ds, GENRE_CLASSES, "TRAIN SET")
    val_block,   y_val_true,   y_val_pred,   val_metrics,   val_conf   = eval_dataset(_eval_model, val_ds,   GENRE_CLASSES, "VALIDATION SET")
    test_block,  y_test_true,  y_test_pred,  test_metrics,  test_conf  = eval_dataset(_eval_model, test_ds,  GENRE_CLASSES, "TEST SET")

_section_times["9. Evaluation"] = _time_module.perf_counter() - _t0
print(f"\nEvaluation : {_section_times['9. Evaluation']:.2f}s")

# ── Primary metric summary (Macro-F1 first) ───────────────────────────────────
print("\nPrimary metric — Macro-F1 (balanced across all genres):")
for label, m in [("train", train_metrics), ("val  ", val_metrics), ("test ", test_metrics)]:
    print(f"  {label}  macro-f1={m['macro_f1']:.4f}")
print("\nSecondary metrics (accuracy is useful but incomplete — inspect with per-genre F1):")
for label, m in [("train", train_metrics), ("val  ", val_metrics), ("test ", test_metrics)]:
    print(f"  {label}  acc={m['accuracy']:.4f}  cost={m['cost']:.4f}")

_report = _json.loads(REPORT_PATH.read_text())
_report["evaluation"] = {
    "timing_seconds": round(_section_times["9. Evaluation"], 2),
    "splits": {
        "train":      train_metrics,
        "validation": val_metrics,
        "test":       test_metrics,
    },
}
REPORT_PATH.write_text(_json.dumps(_report, indent=2))
print(f"Report updated -> {REPORT_PATH}")

# ── Per-genre F1 bar chart (test set) — makes Macro-F1 components visible ─────
_genre_f1_test = [test_metrics["per_genre"].get(g, {}).get("f1-score", 0.0) for g in GENRE_CLASSES]
_chance = 1.0 / N_CLASSES
_bar_colors_f1 = ["#d16d1b" if f < _chance else ("#4CAF50" if f >= 0.6 else "#2196F3")
                  for f in _genre_f1_test]
fig_f1, ax_gf1 = plt.subplots(figsize=(max(10, N_CLASSES * 0.75), 4))
ax_gf1.bar(GENRE_CLASSES, _genre_f1_test, color=_bar_colors_f1, edgecolor="white", linewidth=0.5)
ax_gf1.axhline(_chance,             color="red",        linestyle="--", linewidth=1.0,
               label=f"Chance ({_chance:.2f})")
ax_gf1.axhline(test_metrics["macro_f1"], color="darkorange", linestyle="-",  linewidth=1.5,
               label=f"Macro-F1 = {test_metrics['macro_f1']:.4f}")
for i, (g, f) in enumerate(zip(GENRE_CLASSES, _genre_f1_test)):
    ax_gf1.text(i, f + 0.01, f"{f:.2f}", ha="center", va="bottom", fontsize=7.5)
ax_gf1.set_xticks(range(N_CLASSES))
ax_gf1.set_xticklabels(GENRE_CLASSES, rotation=35, ha="right", fontsize=9)
ax_gf1.set_ylabel("F1-score")
ax_gf1.set_ylim(0, 1.05)
ax_gf1.set_title("Per-genre F1-score — test set  (orange = Macro-F1 average; red = chance)", fontsize=11)
ax_gf1.legend(fontsize=9)
for spine in ["top", "right"]:
    ax_gf1.spines[spine].set_visible(False)
plt.tight_layout()
plt.show()

# Interpretation printout
print("\nPer-genre F1 interpretation (test set):")
_below_chance = [g for g, f in zip(GENRE_CLASSES, _genre_f1_test) if f < _chance]
_strong       = [g for g, f in zip(GENRE_CLASSES, _genre_f1_test) if f >= 0.6]
_weak         = [g for g, f in zip(GENRE_CLASSES, _genre_f1_test) if _chance <= f < 0.4]
print(f"  Strong genres  (F1 ≥ 0.60): {_strong if _strong else 'none'}")
print(f"  Weak genres    (F1 < 0.40): {_weak   if _weak   else 'none'}")
print(f"  Below-chance   (F1 < {_chance:.2f}): {_below_chance if _below_chance else 'none'}")
print(f"\n  Macro-F1 = {test_metrics['macro_f1']:.4f}  |  Accuracy = {test_metrics['accuracy']:.4f}")
if test_metrics["macro_f1"] < test_metrics["accuracy"] - 0.05:
    print("  [NOTE] Macro-F1 is noticeably lower than accuracy — the model performs "
          "well on majority genres but struggles on minority ones. Review below-chance genres above.")
else:
    print("  [OK] Macro-F1 and accuracy are close — genre-level performance is fairly balanced.")

# ── Plot 4 & 9: per-genre F1 + support for all splits ─────────────────────────
_plot_data["per_genre_metrics"] = {
    "train":      train_metrics["per_genre"],
    "validation": val_metrics["per_genre"],
    "test":       test_metrics["per_genre"],
}
# ── Plot 7: confidence histogram data ──────────────────────────────────────────
_plot_data["confidence_histogram"] = {
    "train":      train_conf,
    "validation": val_conf,
    "test":       test_conf,
}
# ── Plot 8: per-class accuracy per split ───────────────────────────────────────
def _per_class_accuracy(y_true, y_pred, genre_classes):
    """Per-class accuracy as {genre: accuracy}."""
    result = {}
    for i, g in enumerate(genre_classes):
        mask = y_true == i
        if mask.sum() > 0:
            result[g] = round(float((y_pred[mask] == i).mean()), 4)
        else:
            result[g] = None
    return result

if "class_distribution" not in _plot_data:
    _plot_data["class_distribution"] = {}
_plot_data["class_distribution"]["per_class_accuracy"] = {
    "train":      _per_class_accuracy(y_train_true, y_train_pred, GENRE_CLASSES),
    "validation": _per_class_accuracy(y_val_true,   y_val_pred,   GENRE_CLASSES),
    "test":       _per_class_accuracy(y_test_true,  y_test_pred,  GENRE_CLASSES),
}
PLOT_DATA_PATH.write_text(_json.dumps(_plot_data, indent=2))
print(f"Plot data updated (per-genre metrics, confidence, per-class accuracy) -> {PLOT_DATA_PATH}")

# %%
_t0 = _time_module.perf_counter()

cm_train = confusion_matrix(y_train_true, y_train_pred)
cm_val   = confusion_matrix(y_val_true,   y_val_pred)
cm       = confusion_matrix(y_test_true,  y_test_pred)

fig, ax = plt.subplots(figsize=(13, 11))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRE_CLASSES)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues", values_format="d")
ax.set_title("Confusion matrix -- test set (log-mel v20a1, 10s)", fontsize=13, pad=14)
plt.tight_layout()
plt.show()

_section_times["9b. Confusion matrix"] = _time_module.perf_counter() - _t0
print(f"Confusion matrix : {_section_times['9b. Confusion matrix']:.2f}s")

# ── Save confusion matrix data for all 3 splits ───────────────────────────────
_plot_data["confusion_matrices"] = {
    "train":      {"matrix": cm_train.tolist(), "labels": GENRE_CLASSES},
    "validation": {"matrix": cm_val.tolist(),   "labels": GENRE_CLASSES},
    "test":       {"matrix": cm.tolist(),        "labels": GENRE_CLASSES},
}
PLOT_DATA_PATH.write_text(_json.dumps(_plot_data, indent=2))
print(f"Plot data updated (confusion matrices) -> {PLOT_DATA_PATH}")

# %% [markdown]
# ## 10. Predict on a New Audio Sample

# %%
_t0 = _time_module.perf_counter()

# INFER_PATHS: list of file paths (str or Path) to run inference on.
# Set to None to auto-select 3 random samples from the test split.
INFER_PATHS = None  # None → auto-select 3 random test samples

if INFER_PATHS is None:
    _rng_size    = min(3, len(test_df))
    _rng_indices = np.random.choice(len(test_df), size=_rng_size, replace=False)
    INFER_PATHS  = [Path(test_df.iloc[i]["filepath"]) for i in _rng_indices]
    _true_genres = [test_df.iloc[i]["genre_top"]      for i in _rng_indices]
    for _p, _g in zip(INFER_PATHS, _true_genres):
        print(f"Using test sample: {_p.name}  (true genre: {_g})")
else:
    INFER_PATHS  = [Path(p) for p in INFER_PATHS]
    _true_genres = ["?" for _ in INFER_PATHS]

# -- 3-crop extraction ---------------------------------------------------------
def extract_three_crops(y: np.ndarray, sr: int, target_sec: float) -> list[np.ndarray]:
    """Extract 3 deterministic 10-second crops from a waveform.

    For clips longer than 3 * target_sec:
      - early crop:  centered at 25% of clip duration
      - middle crop: centered at 50% of clip duration
      - late crop:   centered at 75% of clip duration

    For clips between target_sec and 3 * target_sec:
      - early crop:  starting from the beginning
      - middle crop: centered at midpoint
      - late crop:   ending at the end

    For clips <= target_sec:
      - return 3 copies of the same center-padded clip
    """
    target_len = int(round(target_sec * sr))
    n = len(y)

    if n <= target_len:
        padded = normalize_to_fixed_duration(y, sr, target_sec)
        return [padded, padded, padded]

    def _crop_at(center):
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
    else:
        return [
            y[:target_len],                                     # early
            _crop_at(n // 2),                                   # middle
            y[n - target_len:],                                 # late
        ]

# -- Multi-crop inference over each file ---------------------------------------
_infer_results = []

for _infer_path, _true_genre in zip(INFER_PATHS, _true_genres):
    print(f"\n── {_infer_path.name} ──")
    try:
        y_raw = _load_audio_simple(_infer_path, sr=SAMPLE_RATE, mono=True, duration=None)
        ok, reason = _sanity_check_audio(y_raw, SAMPLE_RATE)
        if not ok:
            raise ValueError(f"sanity_check_failed:{reason}")
        crops = extract_three_crops(y_raw, SAMPLE_RATE, CLIP_DURATION)
        crop_logmels = [_logmel_fixed_shape(c) for c in crops]
    except Exception as exc:
        print(f"  Could not load/extract log-mel: {exc}")
        continue

    crop_probs = []
    for logmel in crop_logmels:
        x_infer = ((logmel[np.newaxis, ..., np.newaxis] - mu) / std).astype(np.float32)
        p = _eval_model.predict(x_infer, verbose=0)[0]
        crop_probs.append(p)

    avg_probs  = np.mean(crop_probs, axis=0)
    pred_idx   = int(np.argmax(avg_probs))
    pred_genre = GENRE_CLASSES[pred_idx]
    confidence = float(avg_probs[pred_idx])

    print(f"  Predicted genre : {pred_genre}  (confidence: {confidence:.2%})")
    print(f"  True genre      : {_true_genre}")
    print(f"  Inference mode  : 3-crop average")
    for i, p in enumerate(crop_probs):
        ci = int(np.argmax(p))
        print(f"    Crop {i+1}: {GENRE_CLASSES[ci]} ({float(p[ci]):.2%})")

    _infer_results.append({
        "file":             str(_infer_path),
        "true_genre":       _true_genre,
        "pred_genre":       pred_genre,
        "confidence":       round(confidence, 4),
        "avg_probs":        [round(float(p), 6) for p in avg_probs],
        "crop_probs":       [[round(float(p), 6) for p in cp] for cp in crop_probs],
        "genre_classes":    GENRE_CLASSES,
        # Plot 11 — log-mel spectrogram of the middle crop
        "logmel_spectrogram": crop_logmels[1].tolist(),
        "logmel_shape":       list(crop_logmels[1].shape),
    })

# -- Combined probability plot (one subplot per file) -------------------------
if _infer_results:
    n_res  = len(_infer_results)
    fig, axes = plt.subplots(n_res, 1, figsize=(10, 3 * n_res), squeeze=False)
    for ax, res in zip(axes[:, 0], _infer_results):
        avg_p  = res["avg_probs"]
        colors = ["steelblue" if g != res["pred_genre"] else "tomato" for g in GENRE_CLASSES]
        ax.barh(GENRE_CLASSES, avg_p, color=colors)
        ax.set_xlabel("Probability (3-crop avg)")
        ax.set_title(
            f"{Path(res['file']).name}  →  pred: {res['pred_genre']} "
            f"({res['confidence']:.2%})  |  true: {res['true_genre']}"
        )
        ax.axvline(1 / N_CLASSES, color="grey", linestyle="--", linewidth=0.8,
                   label=f"Chance ({1/N_CLASSES:.2%})")
        ax.legend(fontsize=9)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()

_section_times["10. Inference"] = _time_module.perf_counter() - _t0
print(f"\nInference : {_section_times['10. Inference']:.2f}s")

# ── Save inference data (Plots 10 + 11) ───────────────────────────────────────
if _infer_results:
    _plot_data["inference_samples"] = _infer_results
    PLOT_DATA_PATH.write_text(_json.dumps(_plot_data, indent=2))
    print(f"Plot data updated (inference + spectrogram) -> {PLOT_DATA_PATH}")

# %% [markdown]
# ---
#
# ## Runtime Summary

# %%
_total = _time_module.perf_counter() - _T0

SEP = "=" * 52
print(SEP)
print("  Runtime summary")
print(SEP)
for section, elapsed in _section_times.items():
    bar_len  = max(1, int(elapsed / _total * 30))
    bar      = "█" * bar_len
    pct      = elapsed / _total * 100
    mins, secs = divmod(elapsed, 60)
    time_str = f"{int(mins)}m {secs:04.1f}s" if mins else f"{elapsed:6.1f}s"
    print(f"  {section:<28}  {time_str:>9}  {pct:5.1f}%  {bar}")

print(SEP)
mins_total, secs_total = divmod(_total, 60)
total_str = f"{int(mins_total)}m {secs_total:04.1f}s" if mins_total else f"{_total:.1f}s"
print(f"  {'TOTAL':<28}  {total_str:>9}")
print(SEP)

