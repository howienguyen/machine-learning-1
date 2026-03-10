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
    "country": 300,
    "pop": 400,
    "blues": 700,
}

# Restrict eligible tracks to rows carrying at most this many genre tags.
# Set to 1 to keep only single-genre tracks.
MAX_GENRE_TAGS_PER_TRACK = 2

OUT_DIR = "genre_downloads"
META_FILE = "additional_datasets/mtg-jamendo-dataset-repo/data/autotagging_genre.tsv"   # or data/autotagging.tsv
TIMEOUT = 30
SLEEP_BETWEEN = 0.2

os.makedirs(OUT_DIR, exist_ok=True)


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


# ----------------------------
# STEP 1: READ METADATA
# ----------------------------
df = load_metadata_table(META_FILE)

target_genres = [normalize_genre_name(g) for g in GENRE_LIMITS.keys()]
track_id_col = detect_track_id_column(df)
tag_col = detect_tag_column(df, target_genres)

print(f"Using track ID column: {track_id_col}")
print(f"Using tag/genre column: {tag_col}")

df[tag_col] = df[tag_col].astype(str)
df["genre_tags"] = df[tag_col].map(extract_genre_tags)
df["genre_tag_count"] = df["genre_tags"].map(len)
df["primary_genre"] = df["genre_tags"].map(primary_genre_from_tags)

eligible_df = df[df["genre_tag_count"] <= MAX_GENRE_TAGS_PER_TRACK].copy()

print(
    f"Eligible rows with <= {MAX_GENRE_TAGS_PER_TRACK} genre tag(s): "
    f"{len(eligible_df)} / {len(df)}"
)


# ----------------------------
# STEP 2: SELECT TRACKS PER GENRE
# ----------------------------
selected_per_genre = {}
all_selected_track_ids = set()
track_to_genres = {}

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

for genre, track_ids in selected_per_genre.items():
    out_tsv = os.path.join(OUT_DIR, f"{genre}_track_ids.tsv")
    pd.DataFrame({track_id_col: track_ids}).to_csv(out_tsv, sep="\t", index=False)

mapping_rows = []
for track_id, genres in track_to_genres.items():
    mapping_rows.append({
        "track_id": track_id,
        "genres": ",".join(sorted(genres)),
    })

pd.DataFrame(mapping_rows).to_csv(
    os.path.join(OUT_DIR, "selected_track_genre_mapping.tsv"),
    sep="\t",
    index=False,
)


# ----------------------------
# STEP 3: DOWNLOAD TRACKS
# ----------------------------
session = requests.Session()

downloaded = 0
failed = []
already_have = 0

for genre in GENRE_LIMITS:
    os.makedirs(os.path.join(OUT_DIR, normalize_genre_name(genre)), exist_ok=True)

for track_id in tqdm(sorted(all_selected_track_ids), desc="Downloading tracks"):
    try:
        assigned_genres = sorted(track_to_genres.get(track_id, []))
        if not assigned_genres:
            failed.append((track_id, "no assigned requested genre"))
            continue

        primary_output_genre = assigned_genres[0]
        out_path = os.path.join(OUT_DIR, primary_output_genre, f"{track_id}.mp3")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            already_have += 1
            continue

        download_url = get_download_url(session, track_id)
        if not download_url:
            failed.append((track_id, "no download url"))
            continue

        download_file(session, download_url, out_path)
        downloaded += 1
        time.sleep(SLEEP_BETWEEN)

    except Exception as exc:
        failed.append((track_id, str(exc)))

print(f"\nDownloaded new files: {downloaded}")
print(f"Already existed: {already_have}")
print(f"Failed: {len(failed)}")

if failed:
    pd.DataFrame(failed, columns=["track_id", "error"]).to_csv(
        os.path.join(OUT_DIR, "failed.tsv"),
        sep="\t",
        index=False,
    )
    print("Saved failures to failed.tsv")