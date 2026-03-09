"""
baseline_mfcc_cnn.py
====================
Reproduces the MFCC → 2D CNN baseline from the official FMA repository:
  FMA/fma-repo/baselines.ipynb  §3.1 "ConvNet on MFCC"
  Reference paper: "Automatic Musical Pattern Feature Extraction Using
  Convolutional Neural Network", Li et al., IMECS 2010.

Architecture (faithful to the original):
  Input  : (13, 2582, 1)  — 13 MFCC coefficients × 2582 time frames
  Conv2D : 3  filters, kernel (13×10), stride (1×4), ReLU
  Conv2D : 15 filters, kernel (1 ×10), stride (1×4), ReLU
  Conv2D : 65 filters, kernel (1 ×10), stride (1×4), ReLU
  Flatten
  Dense  : n_classes (16), Softmax
  Optimizer : SGD lr=1e-3
  Loss      : categorical_crossentropy
  Epochs    : 20, batch_size : 16

Usage:
  python baseline_mfcc_cnn.py [--help]
  python baseline_mfcc_cnn.py --epochs 20 --batch-size 16

Outputs:
  - Console: per-epoch accuracy, final val/test accuracy + Macro-F1 + per-genre F1
  - File:    MelCNN-MGR/results/baseline_mfcc_cnn_results.txt
  - File:    MelCNN-MGR/models/baseline_mfcc_cnn.keras  (trained model)
  - Cache:   MelCNN-MGR/cache/mfcc_{split}.npy / mfcc_{split}_labels.npy
             (skip re-extraction on subsequent runs)



"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
# Locate the workspace root from this file's location
SCRIPT_DIR    = Path(__file__).resolve().parent          # …/MelCNN-MGR
WORKSPACE     = SCRIPT_DIR.parent                        # …/machine-learning-1
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"        # manifest parquets
CACHE_DIR     = SCRIPT_DIR / "cache"
RESULTS_DIR   = SCRIPT_DIR / "results"
MODELS_DIR    = SCRIPT_DIR / "models" / "mfcc_cnn"

# Default subset name — must match the suffix used by build_manifest.py
DEFAULT_SUBSET = "small"   # one of "small", "medium", "large"

# ── MFCC params (same as original) ───────────────────────────────────────────
SAMPLE_RATE  = 22050
N_MFCC       = 13
N_FFT        = 512
HOP_LENGTH   = 256
# 30-second clip @ 22050 Hz → 661 500 samples → frames = 1+(661500-512)//256 = 2582
N_FRAMES     = 2582
MFCC_SHAPE   = (N_MFCC, N_FRAMES)   # (13, 2582)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_audio_path(audio_dir: Path, track_id: int) -> Path:
    """Return the MP3 path for a given FMA track_id.
    E.g. track_id=2 → <audio_dir>/000/000002.mp3
    """
    tid_str = f"{track_id:06d}"
    return audio_dir / tid_str[:3] / f"{tid_str}.mp3"


def load_mfcc(filepath: Path) -> np.ndarray | None:
    """Load one MP3 and compute its MFCC spectrogram.

    Returns an ndarray of shape (13, 2582), or None on error.
    The MFCC is computed to match the FMA baseline exactly:
       sr=22050, n_mfcc=13, n_fft=512, hop_length=256
    Shorter clips are zero-padded; longer ones are truncated.
    """
    try:
        import librosa
        y, _ = librosa.load(str(filepath), sr=SAMPLE_RATE, mono=True, duration=30.0)
        mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE,
                                     n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        # Pad or truncate to fixed width
        if mfcc.shape[1] < N_FRAMES:
            pad = N_FRAMES - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad)), mode="constant")
        else:
            mfcc = mfcc[:, :N_FRAMES]
        return mfcc.astype(np.float32)
    except Exception as exc:
        print(f"  [WARN] Failed to load {filepath}: {exc}", file=sys.stderr)
        return None


def configure_runtime_device(tf):
    """Select CUDA GPU when available, else fall back to CPU.

    Returns:
      device_name: str (e.g., "/GPU:0" or "/CPU:0")
      backend: str ("cuda" or "cpu")
      gpu_names: list[str]
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception:
        gpus = []

    if gpus:
        # Avoid grabbing all GPU memory up-front when CUDA is available.
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        return "/GPU:0", "cuda", [gpu.name for gpu in gpus]

    return "/CPU:0", "cpu", []


# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest_splits(
    processed_dir: Path = PROCESSED_DIR,
    subset: str = DEFAULT_SUBSET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the pre-built manifest split parquets produced by build_manifest.py.

    Each parquet contains only ``reason_code == OK`` rows for the given subset,
    so rows flagged upstream as ``EXCLUDED_LABEL`` (for example,
    ``genre_top == 'Experimental'``) never appear here. Flat columns include
    ``sample_id``, ``source``, ``filepath``, ``genre_top``, ``split``,
    ``artist_id``, ``duration_s``, ``bit_rate``, ``filesize_bytes``,
    ``audio_exists``.

    Parameters
    ----------
    processed_dir:
        Directory that contains ``train_{subset}.parquet``, etc.
        Defaults to ``MelCNN-MGR/data/processed``.
    subset:
        FMA subset tag used as the filename suffix (e.g. ``"medium"`` →
        ``train_medium.parquet``).

    Returns
    -------
    (train_df, val_df, test_df) — DataFrames indexed by ``track_id`` and also
    carrying explicit ``sample_id`` / ``source`` columns for multi-source use.
    """
    def _load(name: str) -> pd.DataFrame:
        path = processed_dir / f"{name}_{subset}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest parquet not found: {path}\n"
                f"Run build_manifest.py first:\n"
                f"  python MelCNN-MGR/preprocessing/build_manifest.py"
            )
        return pd.read_parquet(path)

    train_df = _load("train")
    val_df   = _load("val")
    test_df  = _load("test")

    print(f"  Manifest — train: {len(train_df)}, "
          f"val: {len(val_df)}, test: {len(test_df)} (subset='{subset}')")
    return train_df, val_df, test_df


def resolve_sample_id(row: pd.Series, track_id: int) -> str:
    source = str(row.get("source") or "fma")
    sample_id = row.get("sample_id")
    sample_id = "" if sample_id is None else str(sample_id).strip()
    if not sample_id or sample_id.lower() == "nan":
        sample_id = f"{source}:{track_id}"
    return sample_id


def split_fingerprint(split_df: pd.DataFrame) -> str:
    hasher = hashlib.sha1()
    identities = []
    for track_id, row in split_df.iterrows():
        identities.append(resolve_sample_id(row, int(track_id)))
    for sample_id in sorted(identities):
        hasher.update(sample_id.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:12]


def extract_split(
    split_df: pd.DataFrame,
    split_name: str,
    label_enc,
    cache_dir: Path,
    subset: str = DEFAULT_SUBSET,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract or load cached MFCCs for one split.

    Uses the ``filepath`` column from the manifest parquet directly, so no
    path construction or ``get_audio_path`` call is needed.

    Returns (X, y) where:
      X : float32 ndarray  shape (N, 13, 2582)
      y : int32   ndarray  shape (N,)   — encoded genre class index
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = split_fingerprint(split_df)
        x_path = cache_dir / f"mfcc_{split_name}_{subset}_{fingerprint}.npy"
        y_path = cache_dir / f"mfcc_{split_name}_{subset}_{fingerprint}_labels.npy"

    if x_path.exists() and y_path.exists():
        print(f"  Loading {split_name} from cache …")
        X = np.load(x_path)
        y = np.load(y_path)
        print(f"    X shape: {X.shape}, y shape: {y.shape}")
        return X, y

    print(f"  Extracting MFCCs for split='{split_name}' (subset='{subset}') …")

    Xs, ys = [], []
    skipped = 0
    t0 = time.time()
    for i, (track_id, row) in enumerate(split_df.iterrows()):
        filepath = Path(row["filepath"])   # already resolved by build_manifest
        mfcc = load_mfcc(filepath)
        if mfcc is None:
            skipped += 1
            continue
        Xs.append(mfcc)
        ys.append(row["genre_top"])        # flat column name in manifest

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(split_df)} tracks processed "
                  f"({skipped} skipped), {elapsed:.0f}s elapsed")

    X = np.stack(Xs).astype(np.float32)
    # Encode genre labels to integers
    y = label_enc.transform(np.array(ys)).astype(np.int32)

    print(f"    Done: X={X.shape}, y={y.shape}, skipped={skipped}")
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"    Cached to {cache_dir}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model (faithful to FMA baselines.ipynb §3.1)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(n_classes: int):
    """Build the 2D CNN on MFCC spectrograms.

    Original architecture (modern Keras translation):
      Reshape → (13, 2582, 1)
      Conv2D(3,  (13,10), strides=(1,4)) + ReLU   → (1,  644, 3)
      Conv2D(15, (1, 10), strides=(1,4)) + ReLU   → (1,  159, 15)
      Conv2D(65, (1, 10), strides=(1,4)) + ReLU   → (1,   38, 65)
      Flatten                                       → (2470,)
      Dense(n_classes) + Softmax
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(*MFCC_SHAPE, 1), name="mfcc")    # (13,2582,1)

    x = layers.Conv2D(3,  (13, 10), strides=(1, 4), padding="valid",
                      activation="relu", name="conv1")(inputs)
    x = layers.Conv2D(15, (1,  10), strides=(1, 4), padding="valid",
                      activation="relu", name="conv2")(x)
    x = layers.Conv2D(65, (1,  10), strides=(1, 4), padding="valid",
                      activation="relu", name="conv3")(x)
    x = layers.Flatten(name="flatten")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="fc_out")(x)

    model = keras.Model(inputs, outputs, name="mfcc_2dcnn_baseline")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X, y_true_int, genre_classes, split_label: str,
                   results_lines: list):
    """Compute and print accuracy + Macro-F1 + per-genre F1."""
    import tensorflow as tf
    from sklearn.metrics import (classification_report,
                                  f1_score, accuracy_score)

    y_pred_probs = model.predict(X, batch_size=64, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    acc        = accuracy_score(y_true_int, y_pred)
    macro_f1   = f1_score(y_true_int, y_pred, average="macro", zero_division=0)
    report     = classification_report(y_true_int, y_pred,
                                        target_names=genre_classes,
                                        zero_division=0)
    line = (
        f"\n{'='*60}\n"
        f" {split_label}\n"
        f"{'='*60}\n"
        f"  Accuracy : {acc:.4f}  ({acc:.2%})\n"
        f"  Macro-F1 : {macro_f1:.4f}\n\n"
        f"Per-genre classification report:\n{report}"
    )
    print(line)
    results_lines.append(line)


def main(epochs: int = 20, batch_size: int = 16, clear_cache: bool = False):
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder

    runtime_device, backend, gpu_names = configure_runtime_device(tf)

    print(f"\n{'='*60}")
    print("  FMA Baseline Reproducer — MFCC → 2D CNN")
    print(f"{'='*60}")
    print(f"  TensorFlow : {tf.__version__}")
    print(f"  Backend    : {backend.upper()} ({runtime_device})")
    if gpu_names:
        print(f"  GPUs       : {gpu_names}")
    else:
        print("  GPUs       : none detected → CPU fallback")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch_size}")
    print(f"  Processed  : {PROCESSED_DIR}\n")

    # Optional cache clear
    if clear_cache and CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print("  Cache cleared.\n")

    # ── Load manifest split parquets ─────────────────────────────────────────
    print("Loading manifest parquets …")
    train_df, val_df, test_df = load_manifest_splits()
    all_genres = sorted(
        pd.concat([train_df, val_df, test_df])["genre_top"].unique().tolist()
    )
    n_classes = len(all_genres)
    genre_classes = all_genres
    print(f"  Genres ({n_classes}): {genre_classes}")

    label_enc = LabelEncoder()
    label_enc.fit(genre_classes)

    # ── Extract / load MFCC splits ───────────────────────────────────────────
    print("\nExtracting MFCCs (or loading from cache) …")
    X_train, y_train = extract_split(train_df, "training",   label_enc, CACHE_DIR)
    X_val,   y_val   = extract_split(val_df,   "validation", label_enc, CACHE_DIR)
    X_test,  y_test  = extract_split(test_df,  "test",       label_enc, CACHE_DIR)

    # Add channel dimension  (N, 13, 2582) → (N, 13, 2582, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    # Standardize (fit on train, apply to val/test)  — per-feature mean/std
    print("\nStandardizing features (mean=0, std=1 over training set) …")
    mu  = X_train.mean(axis=(0, 2, 3), keepdims=True)
    std = X_train.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    X_train = (X_train - mu) / std
    X_val   = (X_val   - mu) / std
    X_test  = (X_test  - mu) / std

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   n_classes)

    # ── Build model ─────────────────────────────────────────────────────────
    print("\nBuilding model …")
    with tf.device(runtime_device):
        model = build_model(n_classes)
        model.summary()

        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
        model.compile(optimizer=optimizer,
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])

    # ── Train ────────────────────────────────────────────────────────────────
    print("\nTraining …\n")
    with tf.device(runtime_device):
        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
        )

    # ── Save model ───────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "baseline_mfcc_cnn.keras"
    model.save(str(model_path))
    print(f"\nModel saved → {model_path}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\nEvaluating …")
    results_lines = []
    results_lines.append("FMA Baseline Reproducer — MFCC → 2D CNN\n")
    results_lines.append(f"Epochs={epochs}, batch_size={batch_size}\n")
    results_lines.append(
        f"Training samples : {len(X_train)}\n"
        f"Validation samples: {len(X_val)}\n"
        f"Test samples      : {len(X_test)}\n"
    )

    with tf.device(runtime_device):
        evaluate_model(model, X_train, y_train, genre_classes, "TRAIN SET",  results_lines)
        evaluate_model(model, X_val,   y_val,   genre_classes, "VALIDATION SET", results_lines)
        evaluate_model(model, X_test,  y_test,  genre_classes, "TEST SET",   results_lines)

    # Print training history summary
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]
    history_line = (
        f"\nTraining history summary:\n"
        f"  Final train accuracy : {final_train_acc:.4f}  ({final_train_acc:.2%})\n"
        f"  Final val   accuracy : {final_val_acc:.4f}  ({final_val_acc:.2%})\n"
    )
    print(history_line)
    results_lines.append(history_line)

    # ── Save results ─────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "baseline_mfcc_cnn_results.txt"
    with open(result_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))
    print(f"Results saved → {result_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce the FMA MFCC→2D CNN baseline"
    )
    parser.add_argument("--epochs",      type=int,  default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size",  type=int,  default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached MFCC arrays and re-extract")
    args = parser.parse_args()

    main(epochs=args.epochs,
         batch_size=args.batch_size,
         clear_cache=args.clear_cache)
