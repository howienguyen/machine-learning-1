#!/usr/bin/env python3
"""
Collect extra candidate samples for the small dataset from multiple sources.

The script looks for a configurable target-genre list in three places:
1. FMA metadata rows from the exact medium subset whose track_id is not present
   in metadata_manifest_small.parquet.
2. additional_datasets/dortmund-university genre folders.
3. additional_datasets/gtzan genre folders.

It writes a JSON report containing:
- all discovered candidates per source
- a capped per-genre selection (up to --n-extra-expected)
- folder coverage notes for each external dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from build_manifest import _COL_GENRE_TOP, _COL_SUBSET, load_tracks


_SCRIPT_DIR = Path(__file__).resolve().parent
_MELCNN_DIR = _SCRIPT_DIR.parent
_WORKSPACE = _MELCNN_DIR.parent

DEFAULT_METADATA_ROOT = _WORKSPACE / "FMA" / "fma_metadata"
DEFAULT_SMALL_MANIFEST = _MELCNN_DIR / "data" / "processed" / "metadata_manifest_small.parquet"
DEFAULT_MEDIUM_MANIFEST = _MELCNN_DIR / "data" / "processed" / "metadata_manifest_medium.parquet"
DEFAULT_DORTMUND_ROOT = _WORKSPACE / "additional_datasets" / "dortmund-university"
DEFAULT_GTZAN_ROOT = _WORKSPACE / "additional_datasets" / "gtzan"
DEFAULT_OUTPUT_PATH = _MELCNN_DIR / "data" / "processed" / "extra_samples_for_small_dataset.json"
DEFAULT_SETTINGS_PATH = _MELCNN_DIR / "settings.json"

SOURCE_PRIORITY = ["fma-medium", "dortmund", "gtzan"]
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".au", ".flac", ".ogg", ".aiff", ".aif"}

_DORTMUND_FOLDER_TO_GENRES = {
    "blues": ["Blues"],
    "jazz": ["Jazz"],
    "pop": ["Pop"],
    "raphiphop": ["Hip-Hop"],
    "folkcountry": ["Folk", "Country"],
}

_GTZAN_FOLDER_TO_GENRE = {
    "hiphop": "Hip-Hop",
    "pop": "Pop",
    "rock": "Rock",
    "classical": "Classical",
    "jazz": "Jazz",
    "country": "Country",
    "blues": "Blues",
}


def load_supplementation_settings(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Settings file not found: {path}")

    try:
        raw_settings = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in settings file: {path}") from exc

    config = raw_settings.get("small_dataset_supplementation")
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
            f"settings.small_dataset_supplementation.target_genres in {path}"
        )

    n_extra_expected = config.get("n_extra_expected")
    if not isinstance(n_extra_expected, int) or n_extra_expected <= 0:
        raise ValueError(
            "Expected positive integer at "
            f"settings.small_dataset_supplementation.n_extra_expected in {path}"
        )

    return {
        "target_genres": target_genres,
        "n_extra_expected": n_extra_expected,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect extra candidate samples for the small dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--settings",
        default=str(DEFAULT_SETTINGS_PATH),
        help="Path to MelCNN-MGR settings.json.",
    )
    parser.add_argument(
        "--metadata-root",
        default=str(DEFAULT_METADATA_ROOT),
        help="Path to FMA/fma_metadata.",
    )
    parser.add_argument(
        "--small-manifest",
        default=str(DEFAULT_SMALL_MANIFEST),
        help="Path to metadata_manifest_small.parquet used to exclude existing small-set tracks.",
    )
    parser.add_argument(
        "--medium-manifest",
        default=str(DEFAULT_MEDIUM_MANIFEST),
        help="Path to metadata_manifest_medium.parquet used to resolve FMA-medium filepaths.",
    )
    parser.add_argument(
        "--dortmund-root",
        default=str(DEFAULT_DORTMUND_ROOT),
        help="Path to additional_datasets/dortmund-university.",
    )
    parser.add_argument(
        "--gtzan-root",
        default=str(DEFAULT_GTZAN_ROOT),
        help="Path to additional_datasets/gtzan.",
    )
    parser.add_argument(
        "--out-path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--n-extra-expected",
        type=int,
        default=None,
        help="Maximum number of selected tracks per target genre.",
    )
    parser.add_argument(
        "--target-genres",
        nargs="+",
        default=None,
        help="Target genre_top labels to search for.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def _normalize_genre(value: str) -> str:
    return value.strip().casefold().replace("_", " ").replace("-", " ")


def _json_ready(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def _list_audio_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS
    )


def _load_small_track_ids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Small manifest not found: {path}")
    manifest = pd.read_parquet(path)
    if "subset" not in manifest.columns:
        raise ValueError(f"Expected 'subset' column in small manifest: {path}")
    exact_small = manifest[manifest["subset"] == "small"]
    return set(int(track_id) for track_id in exact_small.index.astype(int))


def _load_medium_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Medium manifest not found: {path}")

    manifest = pd.read_parquet(path)
    required_columns = {"subset", "genre_top", "filepath", "audio_exists", "reason_code"}
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns in medium manifest {path}: {missing}")

    return manifest


def collect_fma_medium_candidates(
    metadata_root: Path,
    medium_manifest_path: Path,
    small_track_ids: set[int],
    target_genres: list[str],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    tracks = load_tracks(metadata_root)
    target_set = set(target_genres)
    medium_manifest = _load_medium_manifest(medium_manifest_path)

    medium_candidates = medium_manifest[
        (medium_manifest["subset"] == "medium")
        & (medium_manifest["genre_top"].isin(target_set))
        & (~medium_manifest.index.isin(small_track_ids))
    ][["filepath", "audio_exists", "reason_code"]].copy()

    medium = tracks[tracks.index.isin(medium_candidates.index)].copy()

    found_by_genre: dict[str, list[dict[str, object]]] = {genre: [] for genre in target_genres}
    if not medium.empty:
        medium = medium.sort_index()
        for track_id, row in medium.iterrows():
            manifest_row = medium_candidates.loc[track_id]
            genre = str(row[_COL_GENRE_TOP])
            found_by_genre[genre].append(
                {
                    "source": "fma-medium",
                    "genre": genre,
                    "track_id": int(track_id),
                    "subset": _json_ready(row.get(_COL_SUBSET)),
                    "title": _json_ready(row.get(("track", "title"))),
                    "artist_name": _json_ready(row.get(("artist", "name"))),
                    "album_title": _json_ready(row.get(("album", "title"))),
                    "filepath": _json_ready(manifest_row.get("filepath")),
                    "audio_exists": _json_ready(manifest_row.get("audio_exists")),
                    "reason_code": _json_ready(manifest_row.get("reason_code")),
                }
            )

    note = {
        "source": "fma-medium",
        "rows_scanned": int(len(tracks)),
        "medium_manifest": str(medium_manifest_path),
        "medium_manifest_candidate_rows": int(len(medium_candidates)),
        "medium_candidate_rows": int(len(medium)),
        "message": (
            "No exact-medium FMA tracks matched the requested genres after excluding "
            "track_ids already present in metadata_manifest_small.parquet."
            if medium.empty else
            "Collected exact-medium FMA candidates not present in the small manifest."
        ),
    }
    return found_by_genre, note


def collect_external_candidates(
    dataset_root: Path,
    source_name: str,
    target_genres: list[str],
    folder_mapping: dict[str, str | list[str]],
) -> tuple[dict[str, list[dict[str, object]]], list[dict[str, object]]]:
    found_by_genre: dict[str, list[dict[str, object]]] = {genre: [] for genre in target_genres}
    folder_notes: list[dict[str, object]] = []

    if not dataset_root.exists():
        folder_notes.append(
            {
                "source": source_name,
                "folder": None,
                "status": "missing_root",
                "path": str(dataset_root),
            }
        )
        return found_by_genre, folder_notes

    normalized_targets = {_normalize_genre(genre): genre for genre in target_genres}
    target_set = set(target_genres)

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue

        mapped = folder_mapping.get(child.name.casefold())
        if mapped is None:
            mapped_genres: list[str] = []
        elif isinstance(mapped, list):
            mapped_genres = [genre for genre in mapped if genre in target_set]
        else:
            mapped_genres = [mapped] if mapped in target_set else []

        if not mapped_genres:
            normalized_name = _normalize_genre(child.name)
            mapped_genre = normalized_targets.get(normalized_name)
            mapped_genres = [mapped_genre] if mapped_genre else []

        files = _list_audio_files(child)
        folder_notes.append(
            {
                "source": source_name,
                "folder": child.name,
                "status": "matched" if mapped_genres else "ignored",
                "mapped_genres": mapped_genres,
                "audio_file_count": len(files),
                "path": str(child),
            }
        )

        for genre in mapped_genres:
            for path in files:
                found_by_genre[genre].append(
                    {
                        "source": source_name,
                        "genre": genre,
                        "track_id": None,
                        "filepath": str(path),
                        "relative_path": str(path.relative_to(_WORKSPACE)),
                        "folder": child.name,
                        "filename": path.name,
                    }
                )

    return found_by_genre, folder_notes


def build_selected_tracks(
    all_found: dict[str, dict[str, list[dict[str, object]]]],
    target_genres: list[str],
    n_extra_expected: int,
) -> dict[str, list[dict[str, object]]]:
    selected: dict[str, list[dict[str, object]]] = {}

    for genre in target_genres:
        combined: list[dict[str, object]] = []
        for source in SOURCE_PRIORITY:
            combined.extend(all_found[source][genre])
        selected[genre] = combined[:n_extra_expected]

    return selected


def build_summary(
    all_found: dict[str, dict[str, list[dict[str, object]]]],
    selected: dict[str, list[dict[str, object]]],
    target_genres: list[str],
    n_extra_expected: int,
) -> dict[str, object]:
    per_genre = {}
    for genre in target_genres:
        source_counts = {
            source: len(all_found[source][genre])
            for source in SOURCE_PRIORITY
        }
        per_genre[genre] = {
            "source_counts": source_counts,
            "total_found": sum(source_counts.values()),
            "selected_count": len(selected[genre]),
            "selected_shortfall": max(0, n_extra_expected - len(selected[genre])),
        }

    return {
        "n_extra_expected": n_extra_expected,
        "per_genre": per_genre,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    settings_path = Path(args.settings)
    settings = load_supplementation_settings(settings_path)

    metadata_root = Path(args.metadata_root)
    small_manifest = Path(args.small_manifest)
    medium_manifest = Path(args.medium_manifest)
    dortmund_root = Path(args.dortmund_root)
    gtzan_root = Path(args.gtzan_root)
    out_path = Path(args.out_path)
    target_genres = list(dict.fromkeys(args.target_genres or settings["target_genres"]))
    n_extra_expected = args.n_extra_expected or settings["n_extra_expected"]

    if n_extra_expected <= 0:
        raise ValueError("--n-extra-expected must be > 0")

    small_track_ids = _load_small_track_ids(small_manifest)
    logging.info("Loaded %d small-manifest track_ids", len(small_track_ids))

    fma_found, fma_note = collect_fma_medium_candidates(
        metadata_root=metadata_root,
        medium_manifest_path=medium_manifest,
        small_track_ids=small_track_ids,
        target_genres=target_genres,
    )
    dortmund_found, dortmund_notes = collect_external_candidates(
        dataset_root=dortmund_root,
        source_name="dortmund",
        target_genres=target_genres,
        folder_mapping=_DORTMUND_FOLDER_TO_GENRES,
    )
    gtzan_found, gtzan_notes = collect_external_candidates(
        dataset_root=gtzan_root,
        source_name="gtzan",
        target_genres=target_genres,
        folder_mapping=_GTZAN_FOLDER_TO_GENRE,
    )

    all_found = {
        "fma-medium": fma_found,
        "dortmund": dortmund_found,
        "gtzan": gtzan_found,
    }
    selected = build_selected_tracks(
        all_found=all_found,
        target_genres=target_genres,
        n_extra_expected=n_extra_expected,
    )
    summary = build_summary(
        all_found=all_found,
        selected=selected,
        target_genres=target_genres,
        n_extra_expected=n_extra_expected,
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_genres": target_genres,
        "settings_path": str(settings_path),
        "small_manifest": str(small_manifest),
        "medium_manifest": str(medium_manifest),
        "metadata_root": str(metadata_root),
        "n_extra_expected": n_extra_expected,
        "source_priority_for_selection": SOURCE_PRIORITY,
        "notes": {
            "fma_medium": fma_note,
            "dortmund": dortmund_notes,
            "gtzan": gtzan_notes,
        },
        "summary": summary,
        "all_found_by_source": all_found,
        "selected_tracks": selected,
    }
    write_json(out_path, payload)

    for genre in target_genres:
        logging.info(
            "%-10s selected=%3d found(fma=%3d dortmund=%3d gtzan=%3d)",
            genre,
            len(selected[genre]),
            len(fma_found[genre]),
            len(dortmund_found[genre]),
            len(gtzan_found[genre]),
        )
    logging.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())