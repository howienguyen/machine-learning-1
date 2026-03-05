# Dev Log — libmpg123 Dequantization Warning Fix

Tue Mar  4 (UTC 2026)

**Files changed:**
- `MelCNN-MGR/notebooks/baseline_mfcc_cnn_v2.ipynb` (config cell, load_mfcc, extract_split, section 4 markdown, MFCC extraction cell)
- `docs/MelCNN-MGR-Preprocessing.md` (new section: "Audio decode quality")

---

## Problem observed

During MFCC extraction on FMA, the following message appeared on stdout/stderr:

```
[src/libmpg123/layer3.c:INT123_do_layer3():1844] error: dequantization failed!
```

It flooded the output (one line per corrupt frame, dozens of lines per corrupt track) and
raised the question of whether it had a negative impact on the training data.

---

## Root cause

**Layer III (MP3) uses Huffman coding for spectral data in each audio frame (~26 ms).**
When a frame's Huffman packet is corrupt or malformed (usually from a bad download, partial
write, or upstream encoder bug), libmpg123 cannot reconstruct the PCM samples.  It:

1. Writes `"error: dequantization failed!"` directly to **C-level stderr (OS fd 2)**.
2. Substitutes zeros for the affected frame and continues decoding.

Critically, **no Python exception is raised**.  `librosa.load()` returns a normal-looking
`np.ndarray` with the correct length.  The MFCC matrix computed from it has the correct shape
`(13, 2582)` but wrong values (silence) for the affected time windows.

Without any guard, the track was counted as successfully extracted and cached — it silently
inflated the training set with subtly corrupted features.

---

## Impact assessment

| Concern | Finding |
|---|---|
| Crash | None — libmpg123 recovers automatically |
| Shape correctness | Always `(13, 2582)`, correct |
| Value correctness | **No** — affected ~26 ms frames are zeroed |
| Python exception | Not raised → track is NOT added to `skipped` without the fix |
| Proportion in FMA | ~1–5 % of tracks; one corrupt track may produce dozens of warning lines |
| Effect on training | Minor noise source; unlikely to dominate a 13 522-track train set |
| Reproducibility | Deterministic — same file → same corrupt frames every run |

---

## Fix applied

### Detection: fd-level stderr capture

The only reliable way to detect C-library stderr messages inside a Python process is to
temporarily replace **OS file descriptor 2** with a pipe before calling `librosa.load()`,
then read the pipe after the call completes.  `contextlib.redirect_stderr()` cannot do this
because it only redirects the Python `sys.stderr` object, not the underlying OS file descriptor.

```python
def _load_audio_with_stderr_capture(filepath, sr, mono=True, duration=30.0):
    import fcntl as _fcntl
    r_fd, w_fd = os.pipe()
    old_fd = os.dup(2)
    sys.stderr.flush()
    os.dup2(w_fd, 2)   # wire fd 2 → write-end of pipe
    os.close(w_fd)
    try:
        y, sr_out = librosa.load(str(filepath), sr=sr, mono=mono, duration=duration)
    finally:
        os.dup2(old_fd, 2)   # restore immediately (even on exception)
        os.close(old_fd)
        _fcntl.fcntl(r_fd, _fcntl.F_SETFL,
                     _fcntl.fcntl(r_fd, _fcntl.F_GETFL) | os.O_NONBLOCK)
        try:
            buf = os.read(r_fd, 131_072)
        except BlockingIOError:
            buf = b""
        os.close(r_fd)
    return y, sr_out, buf.decode("utf-8", errors="replace")
```

Key safety properties:
- fd 2 is **always restored** in `finally` regardless of exceptions — Jupyter output is never lost.
- The pipe read-end is set **non-blocking** so `os.read()` returns `b""` instead of hanging when
  no warnings were produced.
- The pipe buffer (131 072 bytes) is large enough for any realistic libmpg123 output per track.

### API change: `load_mfcc` now returns `(mfcc, is_clean)`

```python
# Before
def load_mfcc(filepath: Path) -> "np.ndarray | None":
    ...

# After
def load_mfcc(filepath: Path) -> "tuple[np.ndarray | None, bool]":
    # is_clean = False  ↔  libmpg123 emitted an "error:" / "dequantization failed" warning
```

`is_clean = False` does **not** mean the MFCC is `None` — the array is still returned so the
caller can decide whether to keep or skip the track.

### New `skip_degraded` parameter on `extract_split`

```python
def extract_split(
    split_df, split_name, label_enc, cache_dir, subset,
    skip_degraded: bool = False,   # ← NEW
) -> tuple:
```

Progress output now shows both counts separately:

```
  [training] Extracting MFCCs for 6,400 tracks …
    500/6400 (8%)  skipped=1  degraded(kept)=12  — 47s elapsed
    ...
    Done : X=(6385, 13, 2582), y=(6385,)
    Skipped  (unreadable)                      : 1
    Degraded (corrupt frames, kept)            : 14  (0.2 %)
```

### New `SKIP_DEGRADED` config knob (config cell)

```python
# SKIP_DEGRADED = False  →  keep degraded tracks (faithful FMA baseline)
# SKIP_DEGRADED = True   →  exclude degraded tracks (cleaner input for new models)
SKIP_DEGRADED = False
```

The flag is passed through to `extract_split()` in the MFCC extraction cell.
**It only takes effect during extraction** — existing `.npy` caches are loaded as-is.
Changing it requires `CLEAR_CACHE = True` to rebuild the cache from scratch.

---

## Documentation updated

- `docs/MelCNN-MGR-Preprocessing.md` — new section "Audio decode quality: the
  `dequantization failed` warning", covering root cause, impact table, fix summary, and
  `SKIP_DEGRADED` behaviour table.
- Section 4 markdown cell in `baseline_mfcc_cnn_v2.ipynb` — inline explanation of the
  warning, its cause, impact in FMA, and the `SKIP_DEGRADED` control table.
- Config cell docstring in `baseline_mfcc_cnn_v2.ipynb` — explains both `False` and `True`
  values of `SKIP_DEGRADED` with guidance on when to use each.
- `_load_audio_with_stderr_capture`, `load_mfcc`, `extract_split` — full docstrings
  explaining the fd-redirect technique and the `is_clean` / `skip_degraded` semantics.

---

## Recommendation summary

| Scenario | `SKIP_DEGRADED` |
|---|---|
| Faithful FMA baseline (Goal 1 comparison) | `False` — match original evaluation conditions |
| New model variants (log-mel CNN, etc.) | `True` — avoid unexplained noise sources |
| Debugging / profiling only | `False` — maximum data, observe degraded count |
