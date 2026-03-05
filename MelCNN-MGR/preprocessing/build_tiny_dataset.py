#!/usr/bin/env python3
"""
build_tiny_dataset.py
=====================
Create a tiny preprocessed FMA dataset from existing split parquets.

The script samples rows from:
  train_{source}.parquet
  val_{source}.parquet
  test_{source}.parquet

and writes:
  train_{output_subset}.parquet
  val_{output_subset}.parquet
  test_{output_subset}.parquet

Default sampling:
  --train-per-genre 70   # 70 samples per genre in the train split
  --val-per-genre   15    # 15 samples per genre in the val split
  --test-per-genre  15    # 15 samples per genre in the test split

The script discovers all genres present in each split independently and
samples up to N rows per genre. If a genre has fewer than N rows a warning
is logged and all available rows for that genre are used.

Examples
--------
	# Use medium as source (default 70/15/15 per genre)
	python MelCNN-MGR/preprocessing/build_tiny_dataset.py --source-subset medium

	# Larger sample: 20 train / 5 val / 5 test per genre
	python MelCNN-MGR/preprocessing/build_tiny_dataset.py --source-subset medium \
	  --train-per-genre 20 --val-per-genre 5 --test-per-genre 5

	# Custom output suffix
	python MelCNN-MGR/preprocessing/build_tiny_dataset.py --source-subset medium --output-subset tiny_medium

	# Custom source file paths (any parquet triplet)
	python MelCNN-MGR/preprocessing/build_tiny_dataset.py \
	  --train-path MelCNN-MGR/data/processed/train_custom.parquet \
	  --val-path   MelCNN-MGR/data/processed/val_custom.parquet \
	  --test-path  MelCNN-MGR/data/processed/test_custom.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import re

# ── Paths (relative to this file) ─────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent           # …/preprocessing
_MELCNN_DIR = _SCRIPT_DIR.parent                        # …/MelCNN-MGR

DEFAULT_PROCESSED_DIR = _MELCNN_DIR / "data" / "processed"
DEFAULT_OUTPUT_SUBSET = "tiny"
DEFAULT_SEED = 42
DEFAULT_GENRE_COL = "genre_top"

DEFAULT_TRAIN_PER_GENRE = 70
DEFAULT_VAL_PER_GENRE = 15
DEFAULT_TEST_PER_GENRE = 15


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Build tiny split parquets from existing FMA split parquets.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)

	parser.add_argument(
		"--processed-dir",
		default=str(DEFAULT_PROCESSED_DIR),
		help="Directory containing split parquets (train_*.parquet, val_*.parquet, test_*.parquet).",
	)

	parser.add_argument(
		"--source-subset",
		default="small",
		help="Source subset to read from train_{subset}.parquet / val_{subset}.parquet / test_{subset}.parquet."
			 " If omitted, auto-detects medium first, then small.",
	)

	parser.add_argument("--train-path", default=None, help="Explicit source train parquet path.")
	parser.add_argument("--val-path", default=None, help="Explicit source val parquet path.")
	parser.add_argument("--test-path", default=None, help="Explicit source test parquet path.")

	parser.add_argument(
		"--train-per-genre", type=int, default=DEFAULT_TRAIN_PER_GENRE,
		help="Rows to sample per genre in the train split.",
	)
	parser.add_argument(
		"--val-per-genre", type=int, default=DEFAULT_VAL_PER_GENRE,
		help="Rows to sample per genre in the val split.",
	)
	parser.add_argument(
		"--test-per-genre", type=int, default=DEFAULT_TEST_PER_GENRE,
		help="Rows to sample per genre in the test split.",
	)

	parser.add_argument(
		"--output-subset",
		default=DEFAULT_OUTPUT_SUBSET,
		help="Output suffix used to write train_{output_subset}.parquet etc.",
	)

	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument(
		"--genre-col",
		default=DEFAULT_GENRE_COL,
		help="Column name holding the genre label. Used for per-genre sampling.",
	)
	parser.add_argument(
		"--log-level",
		default="INFO",
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
	)

	return parser.parse_args(argv)


def _all_or_none(values: list[object | None]) -> bool:
	return all(v is not None for v in values) or all(v is None for v in values)


def resolve_source_paths(args: argparse.Namespace, processed_dir: Path) -> tuple[Path, Path, Path, str]:
	explicit = [args.train_path, args.val_path, args.test_path]
	if not _all_or_none(explicit):
		raise ValueError("Provide either all of --train-path/--val-path/--test-path, or none of them.")

	if all(explicit):
		train_path = Path(args.train_path)
		val_path = Path(args.val_path)
		test_path = Path(args.test_path)
		source_desc = "explicit_paths"
		return train_path, val_path, test_path, source_desc

	if args.source_subset:
		subset = args.source_subset
		train_path = processed_dir / f"train_{subset}.parquet"
		val_path = processed_dir / f"val_{subset}.parquet"
		test_path = processed_dir / f"test_{subset}.parquet"
		source_desc = f"subset:{subset}"
		return train_path, val_path, test_path, source_desc

	# Auto-detect source subset: prefer medium, then small
	for subset in ("medium", "small"):
		train_path = processed_dir / f"train_{subset}.parquet"
		val_path = processed_dir / f"val_{subset}.parquet"
		test_path = processed_dir / f"test_{subset}.parquet"
		if train_path.exists() and val_path.exists() and test_path.exists():
			source_desc = f"subset:{subset}"
			return train_path, val_path, test_path, source_desc

	raise FileNotFoundError(
		"Could not auto-detect source split parquets. "
		"Provide --source-subset (e.g. small/medium) or explicit --train-path/--val-path/--test-path."
	)


def load_split(path: Path, name: str, genre_col: str = DEFAULT_GENRE_COL) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"{name} parquet not found: {path}")
	df = pd.read_parquet(path)
	if len(df) == 0:
		raise ValueError(f"{name} parquet is empty: {path}")
	logging.info(
		"Loaded %-5s | %6d rows × %d cols",
		name, len(df), len(df.columns),
	)
	if genre_col in df.columns:
		n_genres = df[genre_col].nunique()
		top = df[genre_col].value_counts().head(3)
		top_str = ", ".join(f"{g}({c})" for g, c in top.items())
		logging.info(
			"       %-5s | %d genres  top-3: %s%s",
			"", n_genres, top_str,
			" …" if n_genres > 3 else "",
		)
	return df


def sample_split_per_genre(
	df: pd.DataFrame,
	n_per_genre: int,
	seed: int,
	genre_col: str,
) -> tuple[pd.DataFrame, str, dict[str, int]]:
	"""Sample up to *n_per_genre* rows for every genre found in *df[genre_col]*.

	Returns
	-------
	sampled : pd.DataFrame
		The sampled rows, sorted by original index.
	method : str
		Human-readable description of the sampling method used.
	genre_counts : dict[str, int]
		Mapping of genre → number of rows actually sampled.
	"""
	if n_per_genre <= 0:
		raise ValueError(f"n_per_genre must be > 0, got {n_per_genre}.")

	if genre_col not in df.columns or df[genre_col].isna().all():
		# No usable genre column — fall back to a flat random sample.
		n_fallback = min(n_per_genre * 8, len(df))  # rough guess: 8 genres
		sampled = df.sample(n=n_fallback, random_state=seed)
		logging.warning(
			"Genre column %r not found or all-NaN; falling back to flat random sample of %d rows.",
			genre_col, n_fallback,
		)
		return sampled.sort_index(), f"flat_random(n={n_fallback}, no {genre_col})", {}

	genres = sorted(df[genre_col].dropna().unique())
	logging.info("  %d genres found: %s", len(genres), genres)
	logging.info("  Sampling up to %d rows/genre …", n_per_genre)

	parts: list[pd.DataFrame] = []
	genre_counts: dict[str, int] = {}
	short_genres: list[str] = []

	for genre in genres:
		genre_df = df[df[genre_col] == genre]
		available = len(genre_df)
		n = min(n_per_genre, available)
		if n < n_per_genre:
			short_genres.append(f"{genre}({available}/{n_per_genre})")
		parts.append(genre_df.sample(n=n, random_state=seed))
		genre_counts[str(genre)] = n

	if short_genres:
		logging.warning(
			"  Short genres (available/requested): %s",
			", ".join(short_genres),
		)

	# Print a compact per-genre table at DEBUG level so it doesn't clutter INFO runs.
	max_genre_len = max(len(g) for g in genre_counts)
	logging.debug("  %-*s  sampled", max_genre_len, "genre")
	logging.debug("  %s", "-" * (max_genre_len + 10))
	for g, cnt in genre_counts.items():
		logging.debug("  %-*s  %d", max_genre_len, g, cnt)

	sampled = pd.concat(parts).sort_index()
	total = len(sampled)
	logging.info(
		"  → %d total rows sampled (%d genres, avg %.1f/genre)",
		total, len(genres), total / len(genres),
	)
	method = f"per_genre({genre_col}, up_to_{n_per_genre}/genre, {len(genres)} genres)"
	return sampled, method, genre_counts


def write_report(
	report_path: Path,
	source_desc: str,
	train_src: pd.DataFrame,
	val_src: pd.DataFrame,
	test_src: pd.DataFrame,
	train_out: pd.DataFrame,
	val_out: pd.DataFrame,
	test_out: pd.DataFrame,
	method_train: str,
	method_val: str,
	method_test: str,
	train_genre_counts: dict[str, int],
	val_genre_counts: dict[str, int],
	test_genre_counts: dict[str, int],
	genre_col: str,
) -> None:
	lines: list[str] = []
	sep = "=" * 68
	lines += [
		sep,
		"tiny dataset build report",
		f"Generated : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
		f"Source    : {source_desc}",
		sep,
		"",
		"── Source sizes ─────────────────────────────────────────────────────",
		f"  train: {len(train_src):>7,}",
		f"  val  : {len(val_src):>7,}",
		f"  test : {len(test_src):>7,}",
		"",
		"── Output sizes ─────────────────────────────────────────────────────",
		f"  train: {len(train_out):>7,}  ({method_train})",
		f"  val  : {len(val_out):>7,}  ({method_val})",
		f"  test : {len(test_out):>7,}  ({method_test})",
		"",
	]

	# Per-genre breakdown table (train / val / test columns)
	all_genres = sorted(
		set(train_genre_counts) | set(val_genre_counts) | set(test_genre_counts)
	)
	if all_genres:
		lines += ["── Per-genre sample counts ──────────────────────────────────────────"]
		lines.append(f"  {'Genre':<28} {'train':>7}  {'val':>5}  {'test':>5}")
		lines.append("  " + "-" * 52)
		for genre in all_genres:
			tr = train_genre_counts.get(genre, 0)
			vl = val_genre_counts.get(genre, 0)
			te = test_genre_counts.get(genre, 0)
			lines.append(f"  {genre:<28} {tr:>7,}  {vl:>5,}  {te:>5,}")
		lines.append("  " + "-" * 52)
		lines.append(
			f"  {'TOTAL':<28} {sum(train_genre_counts.values()):>7,}"
			f"  {sum(val_genre_counts.values()):>5,}"
			f"  {sum(test_genre_counts.values()):>5,}"
		)
		lines.append("")

	lines += [sep]
	report_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
	args = parse_args(argv)

	logging.basicConfig(
		level=getattr(logging, args.log_level),
		format="%(asctime)s %(levelname)-8s %(message)s",
		datefmt="%H:%M:%S",
	)

	t0 = time.time()
	processed_dir = Path(args.processed_dir)
	processed_dir.mkdir(parents=True, exist_ok=True)

	train_path, val_path, test_path, source_desc = resolve_source_paths(args, processed_dir)
	logging.info("Source train: %s", train_path)
	logging.info("Source val  : %s", val_path)
	logging.info("Source test : %s", test_path)

	logging.info("─── Loading source splits ───────────────────────────────────")
	train_src = load_split(train_path, "train", args.genre_col)
	val_src   = load_split(val_path,   "val",   args.genre_col)
	test_src  = load_split(test_path,  "test",  args.genre_col)

	logging.info("─── Sampling train split ───────────────────────────────────")
	train_out, method_train, train_genre_counts = sample_split_per_genre(
		train_src, args.train_per_genre, args.seed + 11, args.genre_col
	)
	logging.info("─── Sampling val split ─────────────────────────────────────")
	val_out, method_val, val_genre_counts = sample_split_per_genre(
		val_src, args.val_per_genre, args.seed + 17, args.genre_col
	)
	logging.info("─── Sampling test split ────────────────────────────────────")
	test_out, method_test, test_genre_counts = sample_split_per_genre(
		test_src, args.test_per_genre, args.seed + 23, args.genre_col
	)

	requested_output_subset = args.output_subset.strip()
	if not requested_output_subset:
		raise ValueError("--output-subset must be non-empty.")

	def _sanitize_output_subset(subset: str) -> str:
		s = subset.strip()
		if not s:
			return DEFAULT_OUTPUT_SUBSET
		# strip trailing _vNNN or _vN.NN style suffixes (e.g. _v311, _v3.11)
		sanitized = re.sub(r'_v\d+(?:\.\d+)*$', '', s)
		if not sanitized:
			return DEFAULT_OUTPUT_SUBSET
		return sanitized

	output_subset = _sanitize_output_subset(requested_output_subset)
	if output_subset != requested_output_subset:
		logging.info("Sanitized --output-subset: %r -> %r", requested_output_subset, output_subset)

	train_out_path = processed_dir / f"train_{output_subset}.parquet"
	val_out_path = processed_dir / f"val_{output_subset}.parquet"
	test_out_path = processed_dir / f"test_{output_subset}.parquet"
	config_path = processed_dir / f"tiny_dataset_config_{output_subset}.json"
	report_path = processed_dir / f"tiny_dataset_report_{output_subset}.txt"

	train_out.to_parquet(train_out_path, engine="pyarrow", index=True)
	val_out.to_parquet(val_out_path, engine="pyarrow", index=True)
	test_out.to_parquet(test_out_path, engine="pyarrow", index=True)

	logging.info("─── Writing outputs ────────────────────────────────────────")
	logging.info("Wrote train → %s  (%d rows)", train_out_path.name, len(train_out))
	logging.info("Wrote val   → %s  (%d rows)", val_out_path.name,   len(val_out))
	logging.info("Wrote test  → %s  (%d rows)", test_out_path.name,  len(test_out))

	config = {
		"source_desc": source_desc,
		"source_train_path": str(train_path),
		"source_val_path": str(val_path),
		"source_test_path": str(test_path),
		"requested_output_subset": requested_output_subset,
		"output_subset": output_subset,
		"train_per_genre": args.train_per_genre,
		"val_per_genre": args.val_per_genre,
		"test_per_genre": args.test_per_genre,
		"genre_col": args.genre_col,
		"seed": args.seed,
		"sampling_methods": {
			"train": method_train,
			"val": method_val,
			"test": method_test,
		},
		"genre_counts": {
			"train": train_genre_counts,
			"val": val_genre_counts,
			"test": test_genre_counts,
		},
		"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
	}
	with open(config_path, "w", encoding="utf-8") as fh:
		json.dump(config, fh, indent=2)
	logging.info("Wrote config → %s", config_path.name)

	write_report(
		report_path=report_path,
		source_desc=source_desc,
		train_src=train_src,
		val_src=val_src,
		test_src=test_src,
		train_out=train_out,
		val_out=val_out,
		test_out=test_out,
		method_train=method_train,
		method_val=method_val,
		method_test=method_test,
		train_genre_counts=train_genre_counts,
		val_genre_counts=val_genre_counts,
		test_genre_counts=test_genre_counts,
		genre_col=args.genre_col,
	)
	logging.info("Wrote report → %s", report_path.name)

	# ── Final summary ────────────────────────────────────────────────────
	total_out = len(train_out) + len(val_out) + len(test_out)
	n_genres = len(train_genre_counts) or len(val_genre_counts) or len(test_genre_counts)
	logging.info("─── Summary ────────────────────────────────────────────────")
	logging.info("  Source subset   : %s", source_desc)
	logging.info("  Output subset   : %s", output_subset)
	logging.info("  Genres found : %d", n_genres)
	logging.info(
		"  Rows out     : train=%d  val=%d  test=%d  (total=%d)",
		len(train_out), len(val_out), len(test_out), total_out,
	)
	logging.info(
		"  Budget/genre : train=%d  val=%d  test=%d",
		args.train_per_genre, args.val_per_genre, args.test_per_genre,
	)
	logging.info("  Output dir   : %s", processed_dir)
	logging.info("  Elapsed      : %.2fs", time.time() - t0)
	return 0


if __name__ == "__main__":
	sys.exit(main())

