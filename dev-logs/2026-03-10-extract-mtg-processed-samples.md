# Dev Log — 2026-03-10 — `extract_mtg_processed_samples.py` Segment Selection Update

## Scope

This log covers the updates made to `extract_mtg_processed_samples.py` for the MTG/Jamendo post-download sample-extraction workflow.

Updated artifact:

1. `extract_mtg_processed_samples.py`

## Summary

The extractor now treats the configured segment count as a maximum rather than a fixed requirement.

Key changes:

1. renamed the CLI option from `--num-segments` to `--max-num-segments`
2. kept `--min-duration-seconds` as the first eligibility gate
3. added per-file segment-count downgrading based on audio duration
4. expanded the script help text and added a README-style usage example in the module docstring

## New CLI behavior

### `--max-num-segments`

The script no longer assumes that every eligible file should always produce the same number of segments.

Instead, `--max-num-segments` defines the upper bound for a single file.

### Duration-based segment selection

For each audio file:

1. if `duration_seconds <= min_duration_seconds`, the file is skipped
2. otherwise, if `max_num_segments <= 1`, the file uses that value directly
3. otherwise, if `duration_seconds <= 30`, the file uses 1 segment
4. otherwise, if `30 < duration_seconds <= 60`, the file uses 2 segments
5. otherwise, the file uses `max_num_segments`

This preserves the existing minimum-duration filter while allowing shorter eligible files to contribute fewer clips.

## Implementation notes

### New helper

A new helper, `choose_num_segments(duration_seconds, max_num_segments)`, now centralizes the per-file segment-count decision.

### Start-time generation

The existing `choose_start_times(...)` logic still determines where each segment begins.

That means the update changes how many segments are requested for a file, but it does not change the spacing strategy itself.

### User-facing output

The per-file status line now reports the actual number of extracted segments rather than always echoing the configured maximum.

## Example behavior

Given:

- `--max-num-segments 3`
- `--min-duration-seconds 30`
- `--segment-seconds 11`
- `--edge-buffer-seconds 20`

The extractor now behaves as follows:

1. a 28-second file is skipped
2. a 30-second file is skipped
3. a 31-second file is processed with 2 segments
4. a 45-second file is processed with 2 segments
5. a 60-second file is processed with 2 segments
6. a 61-second file is processed with 3 segments

## Documentation update

The top-level module docstring now includes a README-style usage example:

```bash
python extract_mtg_processed_samples.py \
    --input-dir genre_downloads \
    --output-dir mtg-processed-samples \
    --segment-seconds 11 \
    --max-num-segments 3 \
    --min-duration-seconds 30 \
    --edge-buffer-seconds 20
```

The argparse help text was also expanded so the duration-based downgrading rule is visible directly from `--help`.
