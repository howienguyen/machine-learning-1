# %% [markdown]
# # Log-Mel CNN v1
#
# Train directly from prebuilt log-mel `.npy` features referenced by:
# - `logmel_manifest_train.parquet`
# - `logmel_manifest_val.parquet`
# - `logmel_manifest_test.parquet`
#
# This script treats the precomputed log-mel dataset as the authoritative model
# input. Feature extraction should happen upstream via:
# `MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py`

# %% [markdown]
# ## 1. Imports

# %%
import json
import os
import platform
import random
import time
import warnings
from pathlib import Path

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


os.environ["ONEDNN_VERBOSE"] = "none"
os.environ["DNNL_VERBOSE"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

print(f"Python     : {platform.python_version()}")
print(f"TensorFlow : {tf.__version__}")


# %% [markdown]
# ## 2. Configuration

# %%
NOTEBOOK_DIR = Path(__file__).resolve().parent
MELCNN_DIR = NOTEBOOK_DIR.parent
WORKSPACE = MELCNN_DIR.parent

PROCESSED_DIR = MELCNN_DIR / "data" / "processed"
CACHE_DIR = MELCNN_DIR / "cache"
MODELS_BASE_DIR = MELCNN_DIR / "models"

LOGMEL_DATASET_DIR = Path(
    os.environ.get(
        "LOGMEL_DATASET_DIR",
        str(CACHE_DIR / "logmel_dataset_10s"),
    )
)
TRAIN_MANIFEST_PATH = LOGMEL_DATASET_DIR / "logmel_manifest_train.parquet"
VAL_MANIFEST_PATH = LOGMEL_DATASET_DIR / "logmel_manifest_val.parquet"
TEST_MANIFEST_PATH = LOGMEL_DATASET_DIR / "logmel_manifest_test.parquet"
LOGMEL_CONFIG_PATH = LOGMEL_DATASET_DIR / "logmel_config.json"

RUN_DIR = None

EPOCHS = int(os.environ.get("LOGMEL_CNN_EPOCHS", "99"))
BATCH_SIZE = int(os.environ.get("LOGMEL_CNN_BATCH_SIZE", "32"))
LABEL_SMOOTHING = 0.02
WEIGHT_DECAY = 1e-4

SPEC_AUG_FREQ_MASK = 15
SPEC_AUG_TIME_MASK = 25
SPEC_AUG_NUM_MASKS = 2

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


def configure_runtime_device(tf_module):
    print(f"Platform   : {platform.platform()}")
    print(f"TensorFlow : {tf_module.__version__}")

    try:
        gpus = _best_effort_set_memory_growth(tf_module, "GPU")
    except Exception:
        gpus = []
    if gpus:
        ok, info = _smoke_test_matmul(tf_module, "/GPU:0")
        if ok:
            return "/GPU:0", "cuda", [d.name for d in gpus], info
        print("CUDA present but failed smoke test ->", info)

    try:
        import intel_extension_for_tensorflow as itex  # noqa: F401

        xpus = _best_effort_set_memory_growth(tf_module, "XPU")
    except Exception as exc:
        xpus = []
        print("ITEX/XPU not available:", repr(exc))

    if xpus:
        ok, info = _smoke_test_matmul(tf_module, "/XPU:0")
        if ok:
            return "/XPU:0", "xpu", [d.name for d in xpus], info
        print("XPU present but failed smoke test ->", info)

    return "/CPU:0", "cpu", [], "ok"


RUNTIME_DEVICE, BACKEND, ACCEL_NAMES, SMOKE_INFO = configure_runtime_device(tf)
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
present_train_classes = np.unique(train_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=present_train_classes,
    y=train_labels,
)
class_weight_dict = {
    int(class_id): float(weight)
    for class_id, weight in zip(present_train_classes, class_weights)
}
missing_train_class_ids = sorted(set(range(N_CLASSES)) - set(present_train_classes.tolist()))

print("Class weights:")
for i, genre in enumerate(GENRE_CLASSES):
    if i in class_weight_dict:
        print(f"  {genre:<20s}  {class_weight_dict[i]:.4f}")
    else:
        print(f"  {genre:<20s}  absent in train -> no class weight applied")

if missing_train_class_ids:
    missing_train_genres = [GENRE_CLASSES[i] for i in missing_train_class_ids]
    print(f"[WARN] Train split is missing classes: {missing_train_genres}")


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


def make_dataset(index_df: pd.DataFrame, batch_size: int, shuffle: bool, augment: bool = False) -> tf.data.Dataset:
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
    if augment:
        ds = ds.map(lambda x, y: (spec_augment(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train_df, BATCH_SIZE, shuffle=True, augment=True)
val_ds = make_dataset(val_df, BATCH_SIZE, shuffle=False, augment=False)
test_ds = make_dataset(test_df, BATCH_SIZE, shuffle=False, augment=False)

print("\nDatasets ready:")
print(f"  train batches: {tf.data.experimental.cardinality(train_ds).numpy()}  (SpecAugment=ON)")
print(f"  val   batches: {tf.data.experimental.cardinality(val_ds).numpy()}  (SpecAugment=OFF)")
print(f"  test  batches: {tf.data.experimental.cardinality(test_ds).numpy()}  (SpecAugment=OFF)")

_section_times["6. Preprocessing"] = time.perf_counter() - _t0
print(f"\nPreprocessing : {_section_times['6. Preprocessing']:.2f}s")


# %% [markdown]
# ## 7. Build the CNN Model

# %%
_t0 = time.perf_counter()


def build_model(n_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(*LOGMEL_SHAPE, 1), name="logmel")

    x = layers.Conv2D(32, (5, 5), padding="same", use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPool2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.SpatialDropout2D(0.10, name="sdrop2")(x)
    x = layers.MaxPool2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.SpatialDropout2D(0.10, name="sdrop3")(x)
    x = layers.MaxPool2D((2, 2), name="pool3")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPool2D((2, 2), name="pool4")(x)

    x = layers.Conv2D(256, (3, 3), padding="same", use_bias=False, name="conv5")(x)
    x = layers.BatchNormalization(name="bn5")(x)
    x = layers.ReLU(name="relu5")(x)
    x = layers.MaxPool2D((2, 2), name="pool5")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2, name="dropout")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="fc_out")(x)
    return keras.Model(inputs, outputs, name="logmel_cnn_v1")


with tf.device(RUNTIME_DEVICE):
    model = build_model(N_CLASSES)

model.summary()

_section_times["7. Build model"] = time.perf_counter() - _t0
print(f"\nBuild model : {_section_times['7. Build model']:.2f}s")


# %% [markdown]
# ## 8. Compile & Train

# %%
_t0 = time.perf_counter()

_run_ts = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = MODELS_BASE_DIR / f"logmel-cnn-v1-{_run_ts}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run directory : {RUN_DIR}")

WARMUP_EPOCHS = 3
LR_MAX = 1e-3
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

with tf.device(RUNTIME_DEVICE):
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )


class _MacroF1Checkpoint(tf.keras.callbacks.Callback):
    def __init__(self, val_ds: tf.data.Dataset, filepath: str, total_epochs: int, check_freq: int = 3, mid_freq: int = 2, mid_freq_from_epoch: int = 30, dense_from_epoch: int = 60):
        super().__init__()
        self.val_ds = val_ds
        self.filepath = filepath
        self.total_epochs = max(1, int(total_epochs))
        self.check_freq = max(1, check_freq)
        self.mid_freq = max(1, mid_freq)
        self.mid_freq_from_epoch = mid_freq_from_epoch
        self.dense_from_epoch = dense_from_epoch
        self.best_f1 = -1.0
        self.f1_history = []

    def _should_check(self, epoch: int) -> bool:
        if (epoch + 1) >= self.total_epochs or getattr(self.model, "stop_training", False):
            return True
        if epoch >= self.dense_from_epoch:
            return True
        if epoch >= self.mid_freq_from_epoch:
            return (epoch - self.mid_freq_from_epoch) % self.mid_freq == 0
        return (epoch + 1) % self.check_freq == 0

    def on_epoch_end(self, epoch, logs=None):
        if not self._should_check(epoch):
            self.f1_history.append(self.f1_history[-1] if self.f1_history else 0.0)
            return

        y_true, y_pred = [], []
        for xb, yb in self.val_ds:
            preds = self.model(xb, training=False).numpy()
            y_pred.append(np.argmax(preds, axis=1))
            y_true.append(np.argmax(yb.numpy(), axis=1))
        macro_f1 = float(f1_score(np.concatenate(y_true), np.concatenate(y_pred), average="macro", zero_division=0))
        self.f1_history.append(macro_f1)
        if logs is not None:
            logs["val_macro_f1"] = macro_f1
        if macro_f1 > self.best_f1:
            self.best_f1 = macro_f1
            self.model.save(self.filepath)
            print(f"\n  [MacroF1Checkpoint] epoch {epoch + 1}: val_macro_f1 = {macro_f1:.4f} ★ improved -> saved")
        else:
            print(f"\n  [MacroF1Checkpoint] epoch {epoch + 1}: val_macro_f1 = {macro_f1:.4f}  (best = {self.best_f1:.4f})")


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
f1_ckpt = _MacroF1Checkpoint(
    val_ds=val_ds,
    filepath=str(RUN_DIR / "best_model_macro_f1.keras"),
    total_epochs=EPOCHS,
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=9, restore_best_weights=True, verbose=1),
    f1_ckpt,
    lr_logger,
    epoch_timer,
]

print(f"Training on {RUNTIME_DEVICE} | epochs={EPOCHS}, batch_size={BATCH_SIZE}")
print(f"Input source : {LOGMEL_DATASET_DIR}")
print(f"Optimizer    : AdamW(cosine_annealing, warmup={WARMUP_EPOCHS}ep, lr_max={LR_MAX}, weight_decay={WEIGHT_DECAY})")

with tf.device(RUNTIME_DEVICE):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

num_epochs = len(history.history["loss"])
best_loss_epoch = min(range(num_epochs), key=lambda i: history.history["val_loss"][i])
best_f1_epoch = int(np.argmax(f1_ckpt.f1_history)) if f1_ckpt.f1_history else best_loss_epoch
best_epoch = best_f1_epoch

print(f"\nBest epoch by val_loss : {best_loss_epoch + 1}/{num_epochs}  (val_loss={history.history['val_loss'][best_loss_epoch]:.4f})")
print(f"Best epoch by macro_F1 : {best_f1_epoch + 1}/{num_epochs}  (val_macro_f1={f1_ckpt.f1_history[best_f1_epoch]:.4f})")

_section_times["8. Compile & train"] = time.perf_counter() - _t0
print(f"\nCompile & train : {_section_times['8. Compile & train']:.1f}s")

model_path = RUN_DIR / "logmel_cnn_v1.keras"
model.save(str(model_path))
np.savez(str(RUN_DIR / "norm_stats.npz"), mu=mu, std=std, genre_classes=np.array(GENRE_CLASSES))


# %% [markdown]
# ## 9. Training History

# %%
_t0 = time.perf_counter()

hist = history.history
epochs_range = range(1, len(hist["accuracy"]) + 1)

fig, (ax_acc, ax_loss, ax_f1) = plt.subplots(1, 3, figsize=(18, 4))

ax_acc.plot(epochs_range, hist["accuracy"], label="Train", linewidth=2)
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

if f1_ckpt.f1_history:
    ax_f1.plot(epochs_range, f1_ckpt.f1_history, label="Val Macro-F1", linewidth=2.5, color="darkorange")
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

fig.suptitle("Training history — Log-Mel CNN v1")
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
    y_train_true, y_train_pred, train_metrics = eval_dataset(eval_model, train_ds, GENRE_CLASSES, "TRAIN SET")
    y_val_true, y_val_pred, val_metrics = eval_dataset(eval_model, val_ds, GENRE_CLASSES, "VALIDATION SET")
    y_test_true, y_test_pred, test_metrics = eval_dataset(eval_model, test_ds, GENRE_CLASSES, "TEST SET")

print("\nPrimary metric — Macro-F1:")
for label, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
    print(f"  {label:<5} macro-f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")

cm = confusion_matrix(y_test_true, y_test_pred, labels=np.arange(N_CLASSES))
fig, ax = plt.subplots(figsize=(13, 11))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRE_CLASSES)
disp.plot(ax=ax, xticks_rotation=45, colorbar=True, cmap="Blues", values_format="d")
ax.set_title("Confusion matrix — test set (logmel_cnn_v1)")
plt.tight_layout()
plt.show()

_section_times["10. Evaluation"] = time.perf_counter() - _t0
print(f"Evaluation : {_section_times['10. Evaluation']:.2f}s")


# %% [markdown]
# ## 11. Save Run Report

# %%
_t0 = time.perf_counter()

report = {
    "run_id": f"logmel-cnn-v1-{_run_ts}",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
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
        "epochs_max": EPOCHS,
        "epochs_actual": num_epochs,
        "batch_size": BATCH_SIZE,
        "label_smoothing": LABEL_SMOOTHING,
        "weight_decay": WEIGHT_DECAY,
        "best_epoch_val_loss": best_loss_epoch + 1,
        "best_epoch_macro_f1": best_f1_epoch + 1,
        "lr_per_epoch": lr_logger.lrs,
        "seconds_per_epoch": epoch_timer.times,
    },
    "evaluation": {
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    },
}

report_path = RUN_DIR / "run_report_logmel_cnn_v1.json"
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