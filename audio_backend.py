"""audio_backend.py — Audio processing backend for web_audio_capture.py.

This module is intentionally decoupled from the Flask layer.
Integration points called by web_audio_capture.py:

    on_capture_started(sample_rate, channels)
        Called once when the WASAPI loopback stream opens successfully.

    on_chunk(pcm_bytes, sample_rate, channels)
        Called for every raw PCM chunk (~FRAMES_PER_BUFFER frames, int16).

    on_capture_stopped()
        Called when the capture thread exits (either user-initiated or on error).

    on_clip_served(wav_bytes, sample_rate, channels)
        Called when a /clip.wav response is about to be sent to the browser.

TODO: Replace the `pass` stubs below with real processing (e.g. VAD, music
      detection, EQ, keyword recognition, …).
"""

from __future__ import annotations

import logging
import math
import array
import sys
import time

logger = logging.getLogger("audio_backend")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _duration_s(num_bytes: int, sample_rate: int, channels: int) -> float:
    """Convert a raw PCM byte count to a duration in seconds (int16 samples)."""
    bytes_per_frame = channels * 2  # 16-bit = 2 bytes per sample
    if bytes_per_frame <= 0 or sample_rate <= 0:
        return 0.0
    return num_bytes / bytes_per_frame / sample_rate


def _rms(pcm_bytes: bytes) -> float:
    """RMS amplitude normalised to 0..1 for 16-bit little-endian PCM."""
    if not pcm_bytes:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm_bytes)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    acc = sum(int(s) * int(s) for s in samples)
    return min(1.0, math.sqrt(acc / len(samples)) / 32768.0)


# ---------------------------------------------------------------------------
# State (reset on each capture session)
# ---------------------------------------------------------------------------

_session: dict = {}


def _reset_session(sample_rate: int, channels: int) -> None:
    _session.update(
        sample_rate=sample_rate,
        channels=channels,
        chunk_count=0,
        total_bytes=0,
        start_time=time.monotonic(),
    )


# ---------------------------------------------------------------------------
# Public hooks
# ---------------------------------------------------------------------------

def on_capture_started(sample_rate: int, channels: int) -> None:
    """Called once when the WASAPI loopback stream opens successfully."""
    _reset_session(sample_rate, channels)
    logger.info(
        "[BACKEND] Capture started — sample_rate=%d Hz  channels=%d",
        sample_rate,
        channels,
    )
    # TODO: initialise feature extractors, open output files, warm-up models, etc.


def on_chunk(pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
    """Called for every raw PCM int16 chunk from the loopback stream.

    This is called on the capture thread, so keep it fast.
    Offload heavy work to a separate thread/process if needed.
    """
    _session["chunk_count"] = _session.get("chunk_count", 0) + 1
    _session["total_bytes"] = _session.get("total_bytes", 0) + len(pcm_bytes)

    rms = _rms(pcm_bytes)
    elapsed = time.monotonic() - _session.get("start_time", time.monotonic())
    duration = _duration_s(len(pcm_bytes), sample_rate, channels)

    logger.debug(
        "[BACKEND] Chunk #%d  size=%d B  duration=%.3f s  rms=%.4f  elapsed=%.1f s",
        _session["chunk_count"],
        len(pcm_bytes),
        duration,
        rms,
        elapsed,
    )

    # TODO: run VAD, send to feature extractor ring-buffer, etc.


def on_capture_stopped() -> None:
    """Called when the capture session ends (user stop or error)."""
    elapsed = time.monotonic() - _session.get("start_time", time.monotonic())
    total_bytes = _session.get("total_bytes", 0)
    chunk_count = _session.get("chunk_count", 0)
    sr = _session.get("sample_rate", 0)
    ch = _session.get("channels", 0)
    total_duration = _duration_s(total_bytes, sr, ch)

    logger.info(
        "[BACKEND] Capture stopped — chunks=%d  total_pcm=%d B  "
        "audio_duration=%.2f s  wall_time=%.2f s",
        chunk_count,
        total_bytes,
        total_duration,
        elapsed,
    )
    # TODO: flush pending processing, save session metadata, etc.


def on_clip_served(wav_bytes: bytes, sample_rate: int, channels: int) -> None:
    """Called just before a /clip.wav WAV snapshot is sent to the browser.

    wav_bytes includes the 44-byte WAV header.
    """
    pcm_size = max(0, len(wav_bytes) - 44)
    duration = _duration_s(pcm_size, sample_rate, channels)

    logger.info(
        "[BACKEND] Clip served — wav_size=%d B  pcm_size=%d B  duration=%.2f s",
        len(wav_bytes),
        pcm_size,
        duration,
    )
    # TODO: submit clip to music-detection / EQ model pipeline, etc.
