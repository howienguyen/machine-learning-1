"""Download Jamendo tracks by requested primary genres.

README-style usage examples:

    python download_by_genre_limits.py

    python download_by_genre_limits.py \
        --mode retry-failed \
        --failed-tsv genre_downloads/failed_20260310-120000-pid1234.tsv

Fresh-download mode reads the metadata TSV, applies the current genre and
tag-count filters, writes selection TSV outputs, and downloads the chosen tracks.

Retry mode reads a prior failed_*.tsv file and retries only those track ids.
"""

import argparse
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
CLIENT_ID = "b097337c"   # get one from developer.jamendo.com
CLIENT_SECRET = "d317766582bad5ceab2ee17cc12c5456"


# Requested maximum per genre
GENRE_LIMITS = {
    "country": 150,
    "pop": 200,
    "blues": 350,

    # "hiphop": 100,
    # "folk": 100,
    # "metal": 100,  # treat as rock
    # "electronic": 100,
    # "classical": 100,
    # "jazz": 100,
}

# Restrict eligible tracks to rows carrying at most this many genre tags.
# Set to 1 to keep only single-genre tracks.
MAX_GENRE_TAGS_PER_TRACK = 2

OUT_DIR = "genre_downloads"
META_FILE = "additional_datasets/mtg-jamendo-dataset-repo/data/autotagging_genre.tsv"   # or data/autotagging.tsv
TIMEOUT = 30
SLEEP_BETWEEN = 0.2
RUN_ID = f"{time.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}"

os.makedirs(OUT_DIR, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and download Jamendo tracks by primary genre, or retry a prior "
            "failed_*.tsv download batch."
        ),
        epilog=(
            "Examples:\n"
            "  python download_by_genre_limits.py\n"
            "  python download_by_genre_limits.py --mode retry-failed "
            "--failed-tsv genre_downloads/failed_20260310-120000-pid1234.tsv"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["download", "retry-failed"],
        default="download",
        help="Download a fresh selection, or retry track ids listed in a failed TSV.",
    )
    parser.add_argument(
        "--failed-tsv",
        type=str,
        default=None,
        help="Path to a failed_*.tsv file to retry when --mode retry-failed is used.",
    )
    return parser.parse_args()


# ----------------------------
# HELPERS
# ----------------------------
def load_metadata_table(path: str) -> pd.DataFrame:
    """Load Jamendo metadata TSV where the last tags field may contain extra tab-separated values."""
    meta_path = Path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n\r").split("\t")
        if len(header) < 6:
            raise RuntimeError(f"Unexpected metadata header in {meta_path}: {header}")

        fixed_prefix = header[:-1]
        tag_column = header[-1]
        rows = []

        for line_number, raw_line in enumerate(fh, start=2):
            stripped = raw_line.rstrip("\n\r")
            if not stripped:
                continue

            parts = stripped.split("\t")
            if len(parts) < len(header):
                parts.extend([""] * (len(header) - len(parts)))

            row = parts[: len(fixed_prefix)]
            tags = parts[len(fixed_prefix) :]
            row.append("\t".join(tag for tag in tags if tag))

            if len(row) != len(header):
                raise RuntimeError(
                    f"Failed to normalize metadata row at line {line_number}: {parts}"
                )
            rows.append(row)

    return pd.DataFrame(rows, columns=[*fixed_prefix, tag_column])


def normalize_genre_name(name: str) -> str:
    return name.strip().lower()


def normalize_track_id_value(value) -> int:
    text = str(value).strip()
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(f"Could not parse numeric track id from value: {value!r}")
    return int(match.group(1))


def genre_pattern(genre: str) -> str:
    """
    Match official tag style like genre---pop,
    but also allow plain word fallback.
    """
    g = re.escape(normalize_genre_name(genre))
    return rf"genre---{g}|\b{g}\b"


def extract_genre_tags(raw_tags: str) -> list[str]:
    text = str(raw_tags or "")
    tags = []
    seen = set()

    for part in text.split("\t"):
        normalized = part.strip().lower()
        if not normalized:
            continue

        match = re.match(r"genre---(.+)$", normalized)
        if match:
            genre_name = match.group(1).strip()
            if genre_name and genre_name not in seen:
                tags.append(genre_name)
                seen.add(genre_name)

    return tags


def primary_genre_from_tags(tags: list[str]) -> str | None:
    if not tags:
        return None
    return tags[0]


def detect_track_id_column(df: pd.DataFrame) -> str:
    possible = [c for c in df.columns if "track" in c.lower() and "id" in c.lower()]
    if not possible:
        raise RuntimeError(f"Could not find track id column. Columns: {list(df.columns)}")
    return possible[0]


def detect_tag_column(df: pd.DataFrame, target_genres) -> str:
    possible_tag_cols = [c for c in df.columns if "tag" in c.lower() or "genre" in c.lower()]
    for c in possible_tag_cols:
        sample = df[c].astype(str).head(200).tolist()
        sample_text = " | ".join(sample).lower()
        if any(g in sample_text for g in target_genres):
            return c

    string_cols = [c for c in df.columns if df[c].dtype == "object"]
    for c in string_cols:
        sample = df[c].astype(str).head(200).tolist()
        sample_text = " | ".join(sample).lower()
        if any(g in sample_text for g in target_genres):
            return c

    raise RuntimeError(f"Could not find tag/genre column. Columns: {list(df.columns)}")


def build_track_selection(
    df: pd.DataFrame,
    track_id_col: str,
    tag_col: str,
) -> tuple[dict[str, list[int]], set[int], dict[int, set[str]], pd.DataFrame]:
    df = df.copy()
    df[tag_col] = df[tag_col].astype(str)
    df["genre_tags"] = df[tag_col].map(extract_genre_tags)
    df["genre_tag_count"] = df["genre_tags"].map(len)
    df["primary_genre"] = df["genre_tags"].map(primary_genre_from_tags)

    eligible_df = df[df["genre_tag_count"] <= MAX_GENRE_TAGS_PER_TRACK].copy()

    print(
        f"Eligible rows with <= {MAX_GENRE_TAGS_PER_TRACK} genre tag(s): "
        f"{len(eligible_df)} / {len(df)}"
    )

    selected_per_genre: dict[str, list[int]] = {}
    all_selected_track_ids: set[int] = set()
    track_to_genres: dict[int, set[str]] = {}

    for genre, limit in GENRE_LIMITS.items():
        genre_norm = normalize_genre_name(genre)
        genre_df = eligible_df[eligible_df["primary_genre"].eq(genre_norm)].copy()
        genre_df["normalized_track_id"] = genre_df[track_id_col].map(normalize_track_id_value)
        genre_df = genre_df.sort_values(
            by=["genre_tag_count", "normalized_track_id"],
            ascending=[True, True],
            kind="stable",
        )
        genre_track_ids = (
            genre_df.drop_duplicates(subset=[track_id_col], keep="first")["normalized_track_id"]
            .head(limit)
            .tolist()
        )

        selected_per_genre[genre_norm] = genre_track_ids

        for track_id in genre_track_ids:
            all_selected_track_ids.add(track_id)
            track_to_genres.setdefault(track_id, set()).add(genre_norm)

        excluded_multi_genre = int(
            df[df["primary_genre"].eq(genre_norm)]["genre_tag_count"].gt(MAX_GENRE_TAGS_PER_TRACK).sum()
        )
        available_candidates = int(genre_df[track_id_col].nunique())
        print(
            f"{genre_norm}: selected {len(genre_track_ids)} tracks (limit {limit}) from "
            f"{available_candidates} primary-genre candidate(s); "
            f"excluded {excluded_multi_genre} multi-genre rows"
        )

    print(f"\nTotal unique tracks across all genres: {len(all_selected_track_ids)}")
    return selected_per_genre, all_selected_track_ids, track_to_genres, eligible_df


def write_selection_outputs(
    selected_per_genre: dict[str, list[int]],
    track_to_genres: dict[int, set[str]],
    track_id_col: str,
) -> None:
    for genre, track_ids in selected_per_genre.items():
        out_tsv = os.path.join(OUT_DIR, f"{genre}_track_ids_{RUN_ID}.tsv")
        pd.DataFrame({track_id_col: track_ids}).to_csv(out_tsv, sep="\t", index=False)

    mapping_rows = []
    for track_id, genres in track_to_genres.items():
        mapping_rows.append({
            "track_id": track_id,
            "genres": ",".join(sorted(genres)),
        })

    pd.DataFrame(mapping_rows).to_csv(
        os.path.join(OUT_DIR, f"selected_track_genre_mapping_{RUN_ID}.tsv"),
        sep="\t",
        index=False,
    )


def load_failed_downloads(path: str) -> pd.DataFrame:
    failed_path = Path(path)
    if not failed_path.exists():
        raise FileNotFoundError(f"Failed-download TSV not found: {failed_path}")

    failed_df = pd.read_csv(failed_path, sep="\t")
    if "track_id" not in failed_df.columns:
        raise RuntimeError(
            f"Expected a 'track_id' column in failed TSV: {failed_path}"
        )

    failed_df = failed_df.copy()
    failed_df["track_id"] = failed_df["track_id"].map(normalize_track_id_value)
    return failed_df


def resolve_output_genre_for_retry(
    track_id: int,
    failed_row: pd.Series,
    track_to_genres: dict[int, set[str]],
) -> str | None:
    output_genre = failed_row.get("output_genre")
    if pd.notna(output_genre) and str(output_genre).strip():
        return normalize_genre_name(str(output_genre))

    assigned_genres = track_to_genres.get(track_id, set())
    if not assigned_genres:
        return None
    return sorted(assigned_genres)[0]


def download_tracks(
    session: requests.Session,
    track_ids: list[int],
    track_to_genres: dict[int, set[str]],
    failed_rows: dict[int, pd.Series] | None = None,
) -> tuple[int, int, list[dict[str, object]]]:
    downloaded = 0
    failed: list[dict[str, object]] = []
    already_have = 0

    for genre in GENRE_LIMITS:
        os.makedirs(os.path.join(OUT_DIR, normalize_genre_name(genre)), exist_ok=True)

    for track_id in tqdm(track_ids, desc="Downloading tracks"):
        try:
            assigned_genres = sorted(track_to_genres.get(track_id, []))
            failed_row = failed_rows.get(track_id) if failed_rows else None

            if failed_row is not None:
                primary_output_genre = resolve_output_genre_for_retry(track_id, failed_row, track_to_genres)
            else:
                primary_output_genre = assigned_genres[0] if assigned_genres else None

            if not primary_output_genre:
                failed.append(
                    {
                        "track_id": track_id,
                        "assigned_genres": ",".join(assigned_genres),
                        "output_genre": None,
                        "error": "no assigned requested genre",
                    }
                )
                continue

            out_path = os.path.join(OUT_DIR, primary_output_genre, f"{track_id}.mp3")

            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                already_have += 1
                continue

            download_url = get_download_url(session, track_id)
            if not download_url:
                failed.append(
                    {
                        "track_id": track_id,
                        "assigned_genres": ",".join(assigned_genres),
                        "output_genre": primary_output_genre,
                        "error": "no download url",
                    }
                )
                continue

            download_file(session, download_url, out_path)
            downloaded += 1
            time.sleep(SLEEP_BETWEEN)

        except Exception as exc:
            failed.append(
                {
                    "track_id": track_id,
                    "assigned_genres": ",".join(sorted(track_to_genres.get(track_id, []))),
                    "output_genre": primary_output_genre if "primary_output_genre" in locals() else None,
                    "error": str(exc),
                }
            )

    return downloaded, already_have, failed


def write_failed_downloads(failed: list[dict[str, object]]) -> None:
    if not failed:
        return

    pd.DataFrame(failed).to_csv(
        os.path.join(OUT_DIR, f"failed_{RUN_ID}.tsv"),
        sep="\t",
        index=False,
    )
    print(f"Saved failures to failed_{RUN_ID}.tsv")


def get_download_url(session: requests.Session, track_id: int) -> str | None:
    url = "https://api.jamendo.com/v3.0/tracks/"
    params = {
        "client_id": CLIENT_ID,
        "format": "json",
        "id": str(track_id),
        "audioformat": "mp31",
        "audiodlformat": "mp31",
        "limit": 1,
    }
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()

    results = data.get("results", [])
    if not results:
        return None

    item = results[0]
    return item.get("audiodownload") or item.get("audio")


def download_file(session: requests.Session, url: str, out_path: str):
    with session.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with open(out_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 128):
                if chunk:
                    handle.write(chunk)


def main() -> int:
    args = parse_args()

    df = load_metadata_table(META_FILE)
    target_genres = [normalize_genre_name(g) for g in GENRE_LIMITS.keys()]
    track_id_col = detect_track_id_column(df)
    tag_col = detect_tag_column(df, target_genres)

    print(f"Using track ID column: {track_id_col}")
    print(f"Using tag/genre column: {tag_col}")
    print(f"Run ID: {RUN_ID}")
    print(f"Mode: {args.mode}")

    selected_per_genre, all_selected_track_ids, track_to_genres, _ = build_track_selection(
        df,
        track_id_col,
        tag_col,
    )

    failed_rows_by_track_id: dict[int, pd.Series] | None = None
    if args.mode == "download":
        write_selection_outputs(selected_per_genre, track_to_genres, track_id_col)
        track_ids_to_download = sorted(all_selected_track_ids)
    else:
        if not args.failed_tsv:
            raise RuntimeError("--failed-tsv is required when --mode retry-failed is used.")

        failed_df = load_failed_downloads(args.failed_tsv)
        failed_rows_by_track_id = {
            int(row.track_id): pd.Series(row._asdict())
            for row in failed_df.itertuples(index=False)
        }
        track_ids_to_download = sorted(failed_rows_by_track_id.keys())
        print(f"Retrying {len(track_ids_to_download)} failed track(s) from {args.failed_tsv}")

    session = requests.Session()
    downloaded, already_have, failed = download_tracks(
        session=session,
        track_ids=track_ids_to_download,
        track_to_genres=track_to_genres,
        failed_rows=failed_rows_by_track_id,
    )

    print(f"\nDownloaded new files: {downloaded}")
    print(f"Already existed: {already_have}")
    print(f"Failed: {len(failed)}")

    write_failed_downloads(failed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())