#!/usr/bin/env python3
"""
Load selected extra samples into the small train/val/test parquet splits.

The script reads:
1. MelCNN-MGR/settings.json for target_genres, n_extra_expected, and the
   train split ratio.
2. MelCNN-MGR/data/processed/extra_samples_for_small_dataset.json for the
   selected per-genre supplementation candidates.
3. MelCNN-MGR/data/processed/metadata_manifest_medium.parquet to enrich
   FMA-medium rows with the standard manifest columns.

Behavior:
- Only updates the target subset's split parquets (default: small).
- For each target genre, takes up to n_extra_expected selected candidates,
  applies a deterministic shuffle, and assigns rows to train / val / test.
- Uses stable integer surrogate track_ids for external datasets so the parquet
  index remains integer-valued.
- Preserves sample_id / source for multi-source compatibility.
- Skips rows that are already present, so repeated runs are idempotent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


_SCRIPT_DIR = Path(__file__).resolve().parent
_MELCNN_DIR = _SCRIPT_DIR.parent
_WORKSPACE = _MELCNN_DIR.parent

DEFAULT_SETTINGS_PATH = _MELCNN_DIR / "settings.json"
DEFAULT_PROCESSED_DIR = _MELCNN_DIR / "data" / "processed"
DEFAULT_EXTRA_SAMPLES_PATH = DEFAULT_PROCESSED_DIR / "extra_samples_for_small_dataset.json"
DEFAULT_TARGET_SUBSET = "small"
DEFAULT_RANDOM_SEED = 20260309

DEFAULT_SOURCE_NAME = "fma"
EXTERNAL_SOURCES = frozenset({"dortmund", "gtzan"})
EXPECTED_SPLIT_COLUMNS = [
    "sample_id",
    "source",
    "split",
    "subset",
    "genre_top",
    "duration_s",
    "bit_rate",
    "artist_id",
    "filepath",
    "audio_exists",
    "filesize_bytes",
    "reason_code",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load selected extra samples into train/val/test parquet splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS_PATH),
        help="Path to MelCNN-MGR settings.json.",
    )
    parser.add_argument(
        "--extra-samples-path",
        default=str(DEFAULT_EXTRA_SAMPLES_PATH),
        help="Path to extra_samples_for_small_dataset.json.",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(DEFAULT_PROCESSED_DIR),
        help="Directory containing metadata manifests and split parquet files.",
    )
    parser.add_argument(
        "--target-subset",
        default=DEFAULT_TARGET_SUBSET,
        help="Subset whose split parquet files will be updated.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Base random seed used for deterministic per-genre shuffling.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to --processed-dir and overwrites the target split files there.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def load_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in settings file: {path}") from exc

    config = payload.get("small_dataset_supplementation")
    if not isinstance(config, dict):
        raise ValueError(
            "Expected object at settings.small_dataset_supplementation in "
            f"{path}"
        )

    target_genres = config.get("target_genres")
    if not isinstance(target_genres, list) or not target_genres or not all(
        isinstance(genre, str) and genre.strip() for genre in target_genres
    ):
        raise ValueError(
            "Expected non-empty string list at "
            "settings.small_dataset_supplementation.target_genres"
        )

    n_extra_expected = config.get("n_extra_expected")
    if not isinstance(n_extra_expected, int) or n_extra_expected <= 0:
        raise ValueError(
            "Expected positive integer at "
            "settings.small_dataset_supplementation.n_extra_expected"
        )

    train_ratio = config.get("train_n_val_test_split_ratio")
    if not isinstance(train_ratio, (int, float)) or not (0 < float(train_ratio) < 1):
        raise ValueError(
            "Expected float in (0, 1) at "
            "settings.small_dataset_supplementation.train_n_val_test_split_ratio"
        )

    return {
        "target_genres": list(dict.fromkeys(target_genres)),
        "n_extra_expected": n_extra_expected,
        "train_ratio": float(train_ratio),
    }


def load_extra_samples_payload(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Extra samples JSON not found: {path}")

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in extra samples payload: {path}") from exc

    selected_tracks = payload.get("selected_tracks")
    if not isinstance(selected_tracks, dict):
        raise ValueError(f"Expected object at selected_tracks in {path}")

    return payload


def load_medium_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Medium manifest not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = [column for column in EXPECTED_SPLIT_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Medium manifest missing required columns: {missing}")
    return df


def load_split_df(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} parquet not found: {path}")

    df = pd.read_parquet(path)
    missing_columns = [column for column in EXPECTED_SPLIT_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{split_name} parquet missing required columns: {missing}")

    if df.index.name != "track_id":
        df.index.name = "track_id"

    return df


def stable_row_key(row: dict[str, object]) -> str:
    track_id = row.get("track_id")
    filepath = row.get("filepath") or row.get("relative_path") or row.get("filename") or ""
    return "|".join(
        [
            str(row.get("source") or ""),
            str(row.get("genre") or ""),
            "" if track_id is None else str(track_id),
            str(filepath),
        ]
    )


def candidate_identity_key(row: dict[str, object]) -> str:
    source = str(row.get("source") or "")
    if source in EXTERNAL_SOURCES:
        filepath = str(row.get("filepath") or row.get("relative_path") or row.get("filename") or "")
        return f"{source}|{filepath}"

    track_id = row.get("track_id")
    return f"{source}|{'' if track_id is None else int(track_id)}"


def split_row_identity_key(source: object, track_id: object, filepath: object) -> str:
    source_name = str(source or "")
    if source_name in EXTERNAL_SOURCES:
        return f"{source_name}|{'' if filepath is None else str(filepath)}"

    if track_id is None:
        return f"{source_name}|"
    return f"{source_name}|{int(track_id)}"


def genre_seed(base_seed: int, genre: str) -> int:
    digest = hashlib.sha1(genre.encode("utf-8")).digest()
    genre_hash = int.from_bytes(digest[:4], byteorder="big", signed=False)
    return base_seed + genre_hash


def deterministic_shuffle(rows: Iterable[dict[str, object]], seed: int, genre: str) -> list[dict[str, object]]:
    shuffled = list(rows)
    shuffled.sort(key=stable_row_key)
    random.Random(genre_seed(seed, genre)).shuffle(shuffled)
    return shuffled


def compute_split_counts(total: int, train_ratio: float) -> dict[str, int]:
    if total <= 0:
        return {"training": 0, "validation": 0, "test": 0}

    train_count = int(round(total * train_ratio))
    if train_count < 0:
        train_count = 0
    if train_count > total:
        train_count = total

    remainder = total - train_count
    if remainder % 2 == 1:
        train_count += 1
        remainder -= 1

    val_count = remainder // 2
    test_count = remainder // 2
    return {
        "training": train_count,
        "validation": val_count,
        "test": test_count,
    }


def deterministic_external_track_id(
    row: dict[str, object],
    used_track_ids: set[int],
) -> int:
    base_key = stable_row_key(row)
    counter = 0

    while True:
        key = f"{base_key}|{counter}" if counter else base_key
        digest = hashlib.sha1(key.encode("utf-8")).digest()
        unsigned_value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        surrogate = -max(1, unsigned_value & ((1 << 63) - 1))
        if surrogate not in used_track_ids:
            used_track_ids.add(surrogate)
            return surrogate
        counter += 1


def build_row_from_medium_manifest(
    manifest_row: pd.Series,
    track_id: int,
    split_name: str,
    target_subset: str,
) -> dict[str, object]:
    row = {column: manifest_row[column] for column in EXPECTED_SPLIT_COLUMNS}
    row["sample_id"] = f"fma-medium:{track_id}"
    row["source"] = "fma-medium"
    row["split"] = split_name
    row["subset"] = target_subset
    row["reason_code"] = "OK"
    return row


def build_row_from_external_candidate(
    candidate: dict[str, object],
    track_id: int,
    split_name: str,
    target_subset: str,
) -> dict[str, object]:
    filepath = str(candidate.get("filepath") or "")
    source = str(candidate.get("source") or "external")
    path_obj = Path(filepath) if filepath else None
    audio_exists = bool(path_obj and path_obj.exists())
    filesize_bytes = float(path_obj.stat().st_size) if audio_exists else None

    return {
        "sample_id": f"{source}:{track_id}",
        "source": source,
        "split": split_name,
        "subset": target_subset,
        "genre_top": str(candidate.get("genre") or ""),
        "duration_s": -1,
        "bit_rate": -1,
        "artist_id": -1,
        "filepath": filepath,
        "audio_exists": audio_exists,
        "filesize_bytes": filesize_bytes,
        "reason_code": "OK",
    }


def materialize_candidate_rows(
    candidate_rows: list[dict[str, object]],
    split_name: str,
    target_subset: str,
    medium_manifest: pd.DataFrame,
    used_track_ids: set[int],
    existing_sample_ids: set[str],
    existing_identity_keys: set[str],
) -> tuple[list[dict[str, object]], list[int], int]:
    rows: list[dict[str, object]] = []
    indices: list[int] = []
    skipped_existing = 0

    for candidate in candidate_rows:
        source = str(candidate.get("source") or "")
        identity_key = candidate_identity_key(candidate)
        if identity_key in existing_identity_keys:
            skipped_existing += 1
            continue

        if source == "fma-medium":
            track_id = int(candidate["track_id"])
            sample_id = f"fma-medium:{track_id}"
            if sample_id in existing_sample_ids:
                skipped_existing += 1
                continue
            if track_id not in medium_manifest.index:
                raise KeyError(f"track_id {track_id} not found in metadata_manifest_medium.parquet")
            row = build_row_from_medium_manifest(
                manifest_row=medium_manifest.loc[track_id],
                track_id=track_id,
                split_name=split_name,
                target_subset=target_subset,
            )
        elif source in EXTERNAL_SOURCES:
            track_id = deterministic_external_track_id(candidate, used_track_ids)
            sample_id = f"{source}:{track_id}"
            if sample_id in existing_sample_ids:
                skipped_existing += 1
                continue
            row = build_row_from_external_candidate(
                candidate=candidate,
                track_id=track_id,
                split_name=split_name,
                target_subset=target_subset,
            )
        else:
            raise ValueError(f"Unsupported source in selected candidate row: {source!r}")

        existing_sample_ids.add(sample_id)
        existing_identity_keys.add(identity_key)
        rows.append(row)
        indices.append(track_id)

    return rows, indices, skipped_existing


def append_rows(base_df: pd.DataFrame, rows: list[dict[str, object]], indices: list[int]) -> pd.DataFrame:
    if not rows:
        return base_df

    extra_df = pd.DataFrame(rows, index=pd.Index(indices, name="track_id"))
    extra_df = extra_df[EXPECTED_SPLIT_COLUMNS]
    return pd.concat([base_df, extra_df], axis=0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    settings_path = Path(args.settings)
    extra_samples_path = Path(args.extra_samples_path)
    processed_dir = Path(args.processed_dir)
    out_dir = Path(args.out_dir) if args.out_dir else processed_dir
    target_subset = args.target_subset

    settings = load_settings(settings_path)
    payload = load_extra_samples_payload(extra_samples_path)
    medium_manifest_path = Path(
        payload.get("medium_manifest")
        or processed_dir / "metadata_manifest_medium.parquet"
    )
    medium_manifest = load_medium_manifest(medium_manifest_path)

    train_path = processed_dir / f"train_{target_subset}.parquet"
    val_path = processed_dir / f"val_{target_subset}.parquet"
    test_path = processed_dir / f"test_{target_subset}.parquet"
    train_df = load_split_df(train_path, "train")
    val_df = load_split_df(val_path, "val")
    test_df = load_split_df(test_path, "test")

    existing_sample_ids = set(train_df["sample_id"].astype(str))
    existing_sample_ids.update(val_df["sample_id"].astype(str))
    existing_sample_ids.update(test_df["sample_id"].astype(str))

    existing_identity_keys = set(
        split_row_identity_key(source, track_id, filepath)
        for track_id, source, filepath in zip(
            train_df.index.tolist(),
            train_df["source"].tolist(),
            train_df["filepath"].tolist(),
        )
    )
    existing_identity_keys.update(
        split_row_identity_key(source, track_id, filepath)
        for track_id, source, filepath in zip(
            val_df.index.tolist(),
            val_df["source"].tolist(),
            val_df["filepath"].tolist(),
        )
    )
    existing_identity_keys.update(
        split_row_identity_key(source, track_id, filepath)
        for track_id, source, filepath in zip(
            test_df.index.tolist(),
            test_df["source"].tolist(),
            test_df["filepath"].tolist(),
        )
    )

    used_track_ids = set(int(track_id) for track_id in train_df.index.astype(int))
    used_track_ids.update(int(track_id) for track_id in val_df.index.astype(int))
    used_track_ids.update(int(track_id) for track_id in test_df.index.astype(int))

    total_added = {"training": 0, "validation": 0, "test": 0}
    total_skipped_existing = 0

    selected_tracks = payload["selected_tracks"]

    for genre in settings["target_genres"]:
        raw_candidates = selected_tracks.get(genre, [])
        if not isinstance(raw_candidates, list):
            raise ValueError(f"Expected selected_tracks[{genre!r}] to be a list")

        capped_candidates = raw_candidates[: settings["n_extra_expected"]]
        shuffled_candidates = deterministic_shuffle(capped_candidates, args.seed, genre)
        split_counts = compute_split_counts(len(shuffled_candidates), settings["train_ratio"])

        logging.info(
            "%-10s selected=%3d -> train=%3d val=%3d test=%3d",
            genre,
            len(shuffled_candidates),
            split_counts["training"],
            split_counts["validation"],
            split_counts["test"],
        )

        cursor = 0
        train_candidates = shuffled_candidates[cursor: cursor + split_counts["training"]]
        cursor += split_counts["training"]
        val_candidates = shuffled_candidates[cursor: cursor + split_counts["validation"]]
        cursor += split_counts["validation"]
        test_candidates = shuffled_candidates[cursor: cursor + split_counts["test"]]

        train_rows, train_indices, skipped = materialize_candidate_rows(
            candidate_rows=train_candidates,
            split_name="training",
            target_subset=target_subset,
            medium_manifest=medium_manifest,
            used_track_ids=used_track_ids,
            existing_sample_ids=existing_sample_ids,
            existing_identity_keys=existing_identity_keys,
        )
        total_skipped_existing += skipped
        val_rows, val_indices, skipped = materialize_candidate_rows(
            candidate_rows=val_candidates,
            split_name="validation",
            target_subset=target_subset,
            medium_manifest=medium_manifest,
            used_track_ids=used_track_ids,
            existing_sample_ids=existing_sample_ids,
            existing_identity_keys=existing_identity_keys,
        )
        total_skipped_existing += skipped
        test_rows, test_indices, skipped = materialize_candidate_rows(
            candidate_rows=test_candidates,
            split_name="test",
            target_subset=target_subset,
            medium_manifest=medium_manifest,
            used_track_ids=used_track_ids,
            existing_sample_ids=existing_sample_ids,
            existing_identity_keys=existing_identity_keys,
        )
        total_skipped_existing += skipped

        train_df = append_rows(train_df, train_rows, train_indices)
        val_df = append_rows(val_df, val_rows, val_indices)
        test_df = append_rows(test_df, test_rows, test_indices)

        total_added["training"] += len(train_rows)
        total_added["validation"] += len(val_rows)
        total_added["test"] += len(test_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_train_path = out_dir / f"train_{target_subset}.parquet"
    out_val_path = out_dir / f"val_{target_subset}.parquet"
    out_test_path = out_dir / f"test_{target_subset}.parquet"

    train_df.to_parquet(out_train_path, engine="pyarrow", index=True)
    val_df.to_parquet(out_val_path, engine="pyarrow", index=True)
    test_df.to_parquet(out_test_path, engine="pyarrow", index=True)

    logging.info(
        "Added rows -> train=%d val=%d test=%d | skipped_existing=%d",
        total_added["training"],
        total_added["validation"],
        total_added["test"],
        total_skipped_existing,
    )
    logging.info("Wrote %s", out_train_path)
    logging.info("Wrote %s", out_val_path)
    logging.info("Wrote %s", out_test_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())