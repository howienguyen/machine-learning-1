#!/usr/bin/env python3
"""Convert prebuilt log-mel `.npy` datasets into split-sharded TFRecords.

This script consumes the output of:

    python MelCNN-MGR/preprocessing/2_build_log_mel_dataset.py

Expected input-root contents:

1. `logmel_config.json`
2. `logmel_manifest_train.parquet`
3. `logmel_manifest_val.parquet`
4. `logmel_manifest_test.parquet`
5. split-grouped `.npy` files referenced by those manifests

It writes a TFRecord dataset root with matching split directories:

1. `train/`
2. `val/`
3. `test/`

Each split is written as one or more shard files like:

1. `train/train-00000-of-00008.tfrecord`
2. `val/val-00000-of-00001.tfrecord`
3. `test/test-00000-of-00001.tfrecord`

It also writes TFRecord-side config and parquet manifests so downstream
training scripts can discover shard paths, shapes, labels, and genre mapping
without reopening the source `.npy` tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


SCRIPT_DIR = Path(__file__).resolve().parent
MELCNN_DIR = SCRIPT_DIR.parent
WORKSPACE = MELCNN_DIR.parent

DEFAULT_SETTINGS_PATH = MELCNN_DIR / "settings.json"
DEFAULT_SAMPLE_LENGTH_SEC = 15.0


def _load_data_sampling_settings(settings_path: Path) -> dict[str, object] | None:
    try:
        raw_text = settings_path.read_text()
    except Exception:
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            payload = ast.literal_eval(raw_text)
        except (SyntaxError, ValueError):
            return None

    config = payload.get("data_sampling_settings")
    if not isinstance(config, dict):
        return None
    return config


def load_default_sample_length_from_settings(settings_path: Path) -> float:
    config = _load_data_sampling_settings(settings_path)
    if config is None:
        return DEFAULT_SAMPLE_LENGTH_SEC

    sample_length_sec = config.get("sample_length_sec")
    if not isinstance(sample_length_sec, (int, float)) or sample_length_sec <= 0:
        return DEFAULT_SAMPLE_LENGTH_SEC

    return float(sample_length_sec)


DEFAULT_SAMPLE_LENGTH_FROM_SETTINGS = load_default_sample_length_from_settings(DEFAULT_SETTINGS_PATH)
DEFAULT_IN_ROOT = Path("/home/hsnguyen") / "model-training-data-cache" / f"logmel_dataset_{DEFAULT_SAMPLE_LENGTH_FROM_SETTINGS:g}s"
DEFAULT_OUT_ROOT = Path("/home/hsnguyen") / "model-training-data-cache" / f"logmel_dataset_{DEFAULT_SAMPLE_LENGTH_FROM_SETTINGS:g}s_tfrecord"

DEFAULT_RECORDS_PER_SHARD = 1024
DEFAULT_LOG_EVERY = 1000

INPUT_CONFIG_NAME = "logmel_config.json"
OUTPUT_CONFIG_NAME = "tfrecord_config.json"
REPORT_NAME = "tfrecord_build_report.txt"
ALL_MANIFEST_NAME = "tfrecord_manifest_all.parquet"

SPLITS = ("train", "val", "test")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-root",
        default=str(DEFAULT_IN_ROOT),
        help="Input directory produced by 2_build_log_mel_dataset.py.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DEFAULT_OUT_ROOT),
        help="Output directory for split-sharded TFRecords and manifests.",
    )
    parser.add_argument(
        "--records-per-shard",
        type=int,
        default=DEFAULT_RECORDS_PER_SHARD,
        help="Maximum number of samples per TFRecord shard.",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "gzip", "zlib"),
        default="none",
        help="Optional TFRecord compression mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split row limit for smoke tests.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Remove the output root before rebuilding TFRecords.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Progress log interval while writing samples.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _read_input_config(in_root: Path) -> dict[str, object]:
    config_path = in_root / INPUT_CONFIG_NAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Input config not found: {config_path}. Run 2_build_log_mel_dataset.py first."
        )
    try:
        return json.loads(config_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to read input config from {config_path}: {exc}") from exc


def _load_split_manifest(in_root: Path, split: str, row_limit: int | None) -> pd.DataFrame:
    manifest_path = in_root / f"logmel_manifest_{split}.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Input split manifest not found: {manifest_path}")

    frame = pd.read_parquet(manifest_path)
    required_columns = {"sample_id", "genre_top", "logmel_path", "logmel_usable", "split_dir"}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Split manifest {manifest_path} is missing required columns: {missing}")

    frame = frame[frame["logmel_usable"] == True].copy()
    frame = frame[frame["split_dir"].astype(str) == split].copy()
    frame = frame.sort_values(["genre_top", "sample_id"], kind="stable").reset_index(drop=True)
    if row_limit is not None:
        frame = frame.head(row_limit).copy()
    return frame


def _resolve_genre_classes(split_frames: list[pd.DataFrame]) -> list[str]:
    all_genres = sorted(
        {
            str(genre)
            for frame in split_frames
            for genre in frame["genre_top"].astype(str).tolist()
        }
    )
    if len(all_genres) < 2:
        raise ValueError(f"Expected at least 2 genres, found: {all_genres}")
    return all_genres


def _load_logmel_array(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing log-mel file: {path}")

    array = np.load(str(path)).astype(np.float32, copy=False)
    if array.ndim == 2:
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"Unexpected log-mel shape for {path}: {array.shape}; expected {expected_shape}")
        array = array[..., np.newaxis]
    elif array.ndim == 3:
        expected_3d = (*expected_shape, 1)
        if tuple(array.shape) != expected_3d:
            raise ValueError(f"Unexpected log-mel shape for {path}: {array.shape}; expected {expected_3d}")
    else:
        raise ValueError(f"Unexpected log-mel rank for {path}: {array.ndim}")
    return array


def _serialize_example(logmel: np.ndarray, label: int, sample_id: str, genre_top: str, split: str, source_relpath: str) -> bytes:
    serialized_tensor = tf.io.serialize_tensor(tf.convert_to_tensor(logmel, dtype=tf.float32)).numpy()
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "logmel": _bytes_feature(serialized_tensor),
                "label": _int64_feature(label),
                "sample_id": _bytes_feature(sample_id.encode("utf-8")),
                "genre_top": _bytes_feature(genre_top.encode("utf-8")),
                "split_dir": _bytes_feature(split.encode("utf-8")),
                "source_logmel_relpath": _bytes_feature(source_relpath.encode("utf-8")),
            }
        )
    )
    return example.SerializeToString()


def _compression_name(value: str) -> str:
    if value == "none":
        return ""
    if value == "gzip":
        return "GZIP"
    if value == "zlib":
        return "ZLIB"
    raise ValueError(f"Unsupported compression mode: {value}")


def _build_split_tfrecords(
    split_frame: pd.DataFrame,
    split: str,
    in_root: Path,
    out_root: Path,
    expected_shape: tuple[int, int],
    label_by_genre: dict[str, int],
    records_per_shard: int,
    compression: str,
    log_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_out_dir = out_root / split
    split_out_dir.mkdir(parents=True, exist_ok=True)

    total_records = len(split_frame)
    if total_records == 0:
        empty_samples = split_frame.copy()
        empty_shards = pd.DataFrame(columns=["split_dir", "shard_index", "records_in_shard", "tfrecord_path", "tfrecord_relpath"])
        return empty_samples, empty_shards

    shard_count = max(1, math.ceil(total_records / records_per_shard))
    sample_rows: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []
    options = tf.io.TFRecordOptions(compression_type=_compression_name(compression))

    for shard_index in range(shard_count):
        start = shard_index * records_per_shard
        end = min(total_records, start + records_per_shard)
        shard_frame = split_frame.iloc[start:end].copy()
        shard_path = split_out_dir / f"{split}-{shard_index:05d}-of-{shard_count:05d}.tfrecord"
        shard_relpath = str(shard_path.relative_to(out_root))

        with tf.io.TFRecordWriter(str(shard_path), options=options) as writer:
            for record_offset, row in enumerate(shard_frame.itertuples(index=False), start=0):
                logmel_path = Path(str(row.logmel_path)).expanduser().resolve()
                logmel = _load_logmel_array(logmel_path, expected_shape=expected_shape)
                genre_top = str(row.genre_top)
                serialized = _serialize_example(
                    logmel=logmel,
                    label=label_by_genre[genre_top],
                    sample_id=str(row.sample_id),
                    genre_top=genre_top,
                    split=split,
                    source_relpath=str(logmel_path.relative_to(in_root)) if logmel_path.is_relative_to(in_root) else str(logmel_path),
                )
                writer.write(serialized)

                row_dict = row._asdict()
                row_dict.update(
                    {
                        "label_int": int(label_by_genre[genre_top]),
                        "tfrecord_path": str(shard_path),
                        "tfrecord_relpath": shard_relpath,
                        "shard_index": shard_index,
                        "record_index_in_shard": record_offset,
                        "tfrecord_status": "ok",
                        "tfrecord_reason": "",
                    }
                )
                sample_rows.append(row_dict)

        shard_rows.append(
            {
                "split_dir": split,
                "shard_index": shard_index,
                "records_in_shard": len(shard_frame),
                "tfrecord_path": str(shard_path),
                "tfrecord_relpath": shard_relpath,
            }
        )

        processed = end
        if processed % log_every == 0 or processed == total_records:
            logging.info(
                "[TFRecord] %s %d/%d samples written | shards=%d/%d",
                split,
                processed,
                total_records,
                shard_index + 1,
                shard_count,
            )

    return pd.DataFrame(sample_rows), pd.DataFrame(shard_rows)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)


def _write_report(out_root: Path, config: dict[str, object], sample_frame: pd.DataFrame, shard_frame: pd.DataFrame) -> None:
    lines = [
        "=" * 64,
        "3_convert_npy_2_tfrecord report",
        f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "=" * 64,
        "",
        "Configuration:",
    ]
    for key, value in config.items():
        lines.append(f"  {key}: {value}")

    lines.extend(
        [
            "",
            "Dataset summary:",
            f"  Total TFRecord samples : {len(sample_frame):,}",
            f"  Total TFRecord shards  : {len(shard_frame):,}",
            "",
            "Samples by split:",
        ]
    )

    if not sample_frame.empty:
        for split, count in sample_frame.groupby("split_dir").size().sort_index().items():
            lines.append(f"  {split:<8} {count:>7,}")

        lines.extend(["", "Shards by split:"])
        for split, count in shard_frame.groupby("split_dir").size().sort_index().items():
            lines.append(f"  {split:<8} {count:>7,}")

        lines.extend(["", "Samples by split/genre:"])
        for (split, genre), count in sample_frame.groupby(["split_dir", "genre_top"]).size().sort_index().items():
            lines.append(f"  {split:<8} {genre:<20} {count:>7,}")

    lines.extend(["", "=" * 64, ""])
    (out_root / REPORT_NAME).write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if not in_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {in_root}")
    if args.records_per_shard <= 0:
        raise ValueError("--records-per-shard must be positive.")

    if args.clear_cache and out_root.exists():
        logging.info("Removing existing TFRecord output root: %s", out_root)
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    input_config = _read_input_config(in_root)
    n_mels = int(input_config["n_mels"])
    n_frames = int(input_config["n_frames"])
    expected_shape = (n_mels, n_frames)

    split_frames = [_load_split_manifest(in_root, split=split, row_limit=args.limit) for split in SPLITS]
    genre_classes = _resolve_genre_classes(split_frames)
    label_by_genre = {genre: index for index, genre in enumerate(genre_classes)}

    logging.info("Input root: %s", in_root)
    logging.info("Output root: %s", out_root)
    logging.info("Expected log-mel shape: %s", expected_shape)
    logging.info("Records per shard: %d", args.records_per_shard)
    logging.info("Compression: %s", args.compression)
    logging.info("Genre classes (%d): %s", len(genre_classes), genre_classes)

    t0 = time.time()
    split_sample_frames: list[pd.DataFrame] = []
    split_shard_frames: list[pd.DataFrame] = []
    for split, split_frame in zip(SPLITS, split_frames, strict=True):
        logging.info("[TFRecord] Converting split=%s samples=%d", split, len(split_frame))
        sample_frame, shard_frame = _build_split_tfrecords(
            split_frame=split_frame,
            split=split,
            in_root=in_root,
            out_root=out_root,
            expected_shape=expected_shape,
            label_by_genre=label_by_genre,
            records_per_shard=args.records_per_shard,
            compression=args.compression,
            log_every=args.log_every,
        )
        split_sample_frames.append(sample_frame)
        split_shard_frames.append(shard_frame)

    sample_manifest_all = pd.concat(split_sample_frames, axis=0, ignore_index=True) if split_sample_frames else pd.DataFrame()
    shard_manifest_all = pd.concat(split_shard_frames, axis=0, ignore_index=True) if split_shard_frames else pd.DataFrame()

    _write_parquet(out_root / ALL_MANIFEST_NAME, sample_manifest_all)
    _write_parquet(out_root / "tfrecord_shards_all.parquet", shard_manifest_all)
    for split, sample_frame, shard_frame in zip(SPLITS, split_sample_frames, split_shard_frames, strict=True):
        _write_parquet(out_root / f"tfrecord_manifest_{split}.parquet", sample_frame)
        _write_parquet(out_root / f"tfrecord_shards_{split}.parquet", shard_frame)

    output_config = {
        "source_logmel_root": str(in_root),
        "out_root": str(out_root),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records_per_shard": int(args.records_per_shard),
        "compression": args.compression,
        "sample_rate": int(input_config["sample_rate"]),
        "sample_length_sec": float(input_config["sample_length_sec"]),
        "n_mels": n_mels,
        "n_fft": int(input_config["n_fft"]),
        "hop_length": int(input_config["hop_length"]),
        "n_frames": n_frames,
        "logmel_shape": [n_mels, n_frames, 1],
        "genre_classes": genre_classes,
        "label_by_genre": label_by_genre,
        "feature_spec": {
            "logmel": "serialized_tensor(float32)",
            "label": "int64",
            "sample_id": "bytes",
            "genre_top": "bytes",
            "split_dir": "bytes",
            "source_logmel_relpath": "bytes",
        },
        "split_sample_counts": {
            split: int(len(frame))
            for split, frame in zip(SPLITS, split_sample_frames, strict=True)
        },
        "split_shard_counts": {
            split: int(len(frame))
            for split, frame in zip(SPLITS, split_shard_frames, strict=True)
        },
    }
    (out_root / OUTPUT_CONFIG_NAME).write_text(json.dumps(output_config, indent=2) + "\n")
    _write_report(out_root, output_config, sample_manifest_all, shard_manifest_all)

    elapsed = time.time() - t0
    logging.info("Wrote TFRecord root: %s", out_root)
    logging.info("Total TFRecord samples: %d", len(sample_manifest_all))
    logging.info("Total TFRecord shards : %d", len(shard_manifest_all))
    logging.info("Completed in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())