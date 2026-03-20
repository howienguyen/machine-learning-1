
"""Web audio capture demo app.

This Flask app captures system audio (Windows WASAPI loopback), maintains a
short rolling buffer for playback and reconnect replay, and sends fixed-size
snapshots to a separate inference service. It supports two inference event
types produced by the server:

- "partial_result": a live, rolling prediction produced while capture is
    running. Partial results are emitted repeatedly during a session (client
    sends snapshots at `SEND_INTERVAL_SEC`) and are stored in
    `_InferenceState.last_partial`.
- "final_result": a terminal prediction produced when the capture session is
    being stopped. The capture loop flushes one last snapshot and calls
    finalize(), which results in a `final_result` stored in
    `_InferenceState.last_final`.

Notes:
- The UI polls `/inference/status` frequently to refresh `last_partial` and
    `last_final` values; `partial` is for live updates and `final` marks the
    session-closing label (they may be identical if the last partial used the
    same audio window).
- This module is designed for demonstration and local development; it
    intentionally keeps a compact, explicit behavior for save/replay and
    reconnect handling.
"""

# Run guide:
# 1. Start the inference service first, for example:
#    python MelCNN-MGR/inference_web_service/app.py --model-dir MelCNN-MGR/models/logmel-cnn-v2_1-YYYYMMDD-HHMMSS
# 2. Then start this Flask capture app, optionally overriding the REST target or snapshot-send interval:
#    MELCNN_INFERENCE_API_URL=http://127.0.0.1:8000/predict MELCNN_INFERENCE_SEND_INTERVAL_SEC=3 python MelCNN-MGR/demo-app/web_audio_capture_v1.py
# 3. Open http://127.0.0.1:5000 in a browser and use Capture/Stop to stream system audio to the inference service.

import struct
import threading
import time
from collections import deque
import json
import os
import array
import audioop
import math
import random
import sys
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flask import Flask, Response, render_template_string, send_from_directory
try:
    import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]
    PYAUDIO_IMPORT_ERROR = None
except Exception as exc:
    pyaudio = None
    PYAUDIO_IMPORT_ERROR = exc
import requests

if sys.platform != "win32":
    print("This program uses audio_backend which only works on Windows.")
    sys.exit(1)

import utils.audio_backend as audio_backend
LOGGER = logging.getLogger("melcnn.demo_app.web_audio_capture")

# Local capture chunk size in frames returned by the WASAPI loopback stream.
FRAMES_PER_BUFFER = 4096
# Rolling local clip length kept for playback/reconnect replay purposes.
CLIP_SECONDS = 60
# Outbound inference format required by the service/model path.
INFERENCE_SAMPLE_RATE = 22050
INFERENCE_CHANNELS = 1
# Preferred capture quality for replay/saved clips. Stream opening still falls
# back to other device-supported formats if needed.
PREFERRED_CAPTURE_SAMPLE_RATES = (48000, 44100, 22050)
PREFERRED_CAPTURE_CHANNELS = (2, 1)
SETTINGS_PATH = WORKSPACE_ROOT / "MelCNN-MGR" / "settings.json"


def _load_min_inference_seconds(settings_path: Path) -> float:
    default_seconds = 10.0
    try:
        payload = json.loads(settings_path.read_text())
    except Exception:
        return default_seconds

    config = payload.get("data_sampling_settings")
    if not isinstance(config, dict):
        return default_seconds

    sample_length_sec = config.get("sample_length_sec")
    if not isinstance(sample_length_sec, (int, float)) or sample_length_sec <= 0:
        return default_seconds

    return float(sample_length_sec)


def _format_seconds_label(seconds: float) -> str:
    return f"{seconds:g}"


# Model inference contract currently requires a full n-second window.
MIN_INFERENCE_SECONDS = _load_min_inference_seconds(SETTINGS_PATH)

# Inference service REST endpoint consumed by this capture client. If only a
# websocket URL is provided, it is converted to the matching /predict URL.
def _resolve_inference_api_url() -> str:
    raw_value = os.environ.get("MELCNN_INFERENCE_API_URL")
    if not raw_value:
        raw_value = os.environ.get("MELCNN_INFERENCE_WS_URL", "http://172.17.233.113:8000/predict")

    if "://" not in raw_value:
        raw_value = f"http://{raw_value}"

    parsed = urlsplit(raw_value)
    scheme = parsed.scheme.lower()
    if scheme in {"ws", "wss"}:
        scheme = "https" if scheme == "wss" else "http"
        path = "/predict" if not parsed.path or parsed.path == "/ws/stream" else parsed.path
        return urlunsplit((scheme, parsed.netloc, path, parsed.query, parsed.fragment))
    path = parsed.path or "/predict"
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, parsed.fragment))


INFERENCE_API_URL = _resolve_inference_api_url()
# Inference mode requested from the server for each stream session.
INFERENCE_MODE = "single_crop"
# Server-side partial-result cadence once a full latest-window snapshot is available.
EMIT_INTERVAL_SEC = 1
# Client-side cadence for sending the latest MIN_INFERENCE_SECONDS snapshot to the service.
SEND_INTERVAL_SEC = 6
# REST connect timeout for service probes and inference requests.
REST_CONNECT_TIMEOUT_SEC = 15.0

# REST response timeout for inference requests.
REST_REQUEST_TIMEOUT_SEC = 15.0

# Background retry cadence used when the inference service is unavailable.
RECONNECT_RETRY_INTERVAL_SEC = 10
# Maximum number of consecutive reconnect attempts before inference is disabled for the current capture session.
RECONNECT_MAX_ATTEMPTS = 6
# Amount of recent PCM to replay after reconnect so the service can restore one full inference window.
RECONNECT_REPLAY_SECONDS = 10.0
MAX_GENRE_RESULT_NOTES = 6
TREND_WINDOW_SECONDS = 180
MAX_TREND_HISTORY_POINTS = 512
DEMO_APP_IMAGES_DIR = (WORKSPACE_ROOT / "MelCNN-MGR" / "demo-app" / "images").resolve()
CAPTURED_AUDIO_DIR = (WORKSPACE_ROOT / "MelCNN-MGR" / "data" / "tmp_demo_app").resolve()
MAX_SAVED_CAPTURED_CLIPS = 10
SAVE_LATEST_CAPTURED_CLIPS = False

app = Flask(__name__)


# Endpoints polled at high frequency that would otherwise flood the console.
_SUPPRESS_ENDPOINTS = ("/level", "/inference/status")


class _SuppressLevelEndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(f"GET {ep}" in msg for ep in _SUPPRESS_ENDPOINTS)


def _rms_level_int16(pcm_bytes: bytes) -> float:
    """Returns an RMS level normalized to 0..1 for 16-bit PCM."""
    if not pcm_bytes:
        return 0.0

    samples = array.array('h')
    samples.frombytes(pcm_bytes)
    if sys.byteorder == 'big':
        samples.byteswap()

    if not samples:
        return 0.0

    acc = 0
    for s in samples:
        acc += int(s) * int(s)
    rms = math.sqrt(acc / len(samples))
    return float(min(1.0, max(0.0, rms / 32768.0)))


def _convert_pcm16_for_inference(
    pcm_bytes: bytes,
    input_sample_rate: int,
    input_channels: int,
) -> bytes:
    """Downmixes to mono and resamples PCM16 little-endian audio for inference."""
    if not pcm_bytes or input_sample_rate <= 0 or input_channels <= 0:
        return b""

    frame_width = 2 * input_channels
    usable_bytes = (len(pcm_bytes) // frame_width) * frame_width
    if usable_bytes <= 0:
        return b""

    pcm_bytes = pcm_bytes[:usable_bytes]
    if input_channels > 1:
        pcm_bytes = audioop.tomono(pcm_bytes, 2, 0.5, 0.5)

    if input_sample_rate != INFERENCE_SAMPLE_RATE:
        pcm_bytes, _ = audioop.ratecv(
            pcm_bytes,
            2,
            1,
            input_sample_rate,
            INFERENCE_SAMPLE_RATE,
            None,
        )

    return pcm_bytes

def create_wav_header(sample_rate, channels, data_size_bytes):
    """Generates a standard WAV header for a finite PCM payload."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    data_size = int(data_size_bytes)
    chunk_size = 36 + data_size

    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', chunk_size, b'WAVE',
        b'fmt ', 16, 1, channels, sample_rate, byte_rate, block_align, bits_per_sample,
        b'data', data_size
    )


def _pcm_payload_to_wav_bytes(pcm_bytes: bytes, sample_rate: int, channels: int) -> bytes:
    if not pcm_bytes:
        return b""
    return create_wav_header(sample_rate, channels, len(pcm_bytes)) + pcm_bytes


def _local_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_filename_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return cleaned or "capture"


def _trim_saved_captured_clips(directory: Path, keep_latest: int) -> None:
    wav_files = sorted(directory.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale_path in wav_files[keep_latest:]:
        stale_path.unlink(missing_ok=True)


def _save_captured_clip(wav_bytes: bytes, sample_rate: int, channels: int, source_tag: str = "capture") -> Path | None:
    if not SAVE_LATEST_CAPTURED_CLIPS:
        return None
    if not wav_bytes:
        return None
    CAPTURED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    millis = int((time.time() % 1) * 1000)
    clip_name = f"{timestamp}-{millis:03d}-{_sanitize_filename_component(source_tag)}-{sample_rate}hz-{channels}ch.wav"
    clip_path = CAPTURED_AUDIO_DIR / clip_name
    clip_path.write_bytes(wav_bytes)
    _trim_saved_captured_clips(CAPTURED_AUDIO_DIR, MAX_SAVED_CAPTURED_CLIPS)
    LOGGER.info("Saved outbound clip path=%s bytes=%s sample_rate=%s channels=%s", clip_path, len(wav_bytes), sample_rate, channels)
    return clip_path


def _demo_asset_url(filename: str) -> str:
    return f"/demo-assets/{filename}"


def _choose_falling_asset(default_filename: str) -> str:
    if random.random() < 0.25:
        return random.choice(["music1.png", "music2.png"])
    return default_filename


def _build_falling_asset_sets() -> list[list[str]]:
    base_assets = [
        "leaves1.png",
        "leaves2.png",
        "leaves3.png",
        "leaves4.png",
        "leaves1.png",
        "leaves2.png",
        "leaves3.png",
        "leaves4.png",
    ]
    return [
        [_demo_asset_url(_choose_falling_asset(asset_name)) for asset_name in base_assets]
        for _ in range(3)
    ]

class _RollingAudioBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._chunks = deque()
        self._chunk_end_time_ms = deque()
        self._max_chunks = 1
        self.sample_rate = None
        self.channels = None

    def configure(self, sample_rate, channels):
        with self._lock:
            self.sample_rate = int(sample_rate)
            self.channels = int(channels)
            # Keep enough chunks for ~CLIP_SECONDS.
            chunks_per_second = self.sample_rate / FRAMES_PER_BUFFER
            self._max_chunks = max(1, int(CLIP_SECONDS * chunks_per_second) + 1)
            self._chunks = deque(maxlen=self._max_chunks)
            self._chunk_end_time_ms = deque(maxlen=self._max_chunks)

    def clear(self):
        with self._lock:
            self._chunks.clear()
            self._chunk_end_time_ms.clear()

    def push(self, pcm_bytes, chunk_end_time_ms=None):
        with self._lock:
            self._chunks.append(pcm_bytes)
            if chunk_end_time_ms is None:
                chunk_end_time_ms = int(time.time() * 1000)
            self._chunk_end_time_ms.append(int(chunk_end_time_ms))

    def snapshot_wav(self):
        with self._lock:
            if not self._chunks or not self.sample_rate or not self.channels:
                return None
            pcm = b"".join(self._chunks)
            header = create_wav_header(self.sample_rate, self.channels, len(pcm))
            return header + pcm

    def snapshot_chunks(self, max_seconds=None):
        with self._lock:
            chunks = list(self._chunks)
            sample_rate = self.sample_rate
        if not chunks:
            return []
        if max_seconds is None or not sample_rate:
            return chunks
        chunks_per_second = sample_rate / FRAMES_PER_BUFFER
        keep_chunks = max(1, int(math.ceil(float(max_seconds) * chunks_per_second)))
        return chunks[-keep_chunks:]

    def snapshot_end_time_ms(self, max_seconds=None):
        with self._lock:
            chunk_end_time_ms = list(self._chunk_end_time_ms)
            sample_rate = self.sample_rate
        if not chunk_end_time_ms:
            return None
        if max_seconds is None or not sample_rate:
            return int(chunk_end_time_ms[-1])
        chunks_per_second = sample_rate / FRAMES_PER_BUFFER
        keep_chunks = max(1, int(math.ceil(float(max_seconds) * chunks_per_second)))
        selected = chunk_end_time_ms[-keep_chunks:]
        if not selected:
            return None
        return int(selected[-1])

    def snapshot_pcm(self, max_seconds=None):
        # Returns one contiguous latest-window PCM payload for websocket upload.
        chunks = self.snapshot_chunks(max_seconds=max_seconds)
        if not chunks:
            return b""
        return b"".join(chunks)

    def buffered_seconds(self):
        with self._lock:
            if not self._chunks or not self.sample_rate or not self.channels:
                return 0.0
            total_bytes = sum(len(chunk) for chunk in self._chunks)
            bytes_per_second = self.sample_rate * self.channels * 2
        if bytes_per_second <= 0:
            return 0.0
        return float(total_bytes / bytes_per_second)


rolling_buffer = _RollingAudioBuffer()


def _latest_inference_snapshot_pcm() -> bytes:
    """Return the newest inference-ready PCM snapshot from the rolling buffer.

    The inference service currently expects a full `MIN_INFERENCE_SECONDS`
    window in mono 16-bit PCM at `INFERENCE_SAMPLE_RATE`. This helper takes the
    latest capture-window PCM from `_RollingAudioBuffer` and converts it into
    the model-serving contract used for both rolling partial requests and the
    final session flush.
    """
    with rolling_buffer._lock:
        sample_rate = rolling_buffer.sample_rate
        channels = rolling_buffer.channels
    raw_pcm = rolling_buffer.snapshot_pcm(MIN_INFERENCE_SECONDS)
    return _convert_pcm16_for_inference(
        raw_pcm,
        int(sample_rate or 0),
        int(channels or 0),
    )


def _latest_inference_captured_time_ms() -> int | None:
    return rolling_buffer.snapshot_end_time_ms(MIN_INFERENCE_SECONDS)


def _latest_inference_replay_chunks() -> list[bytes]:
    """Return replay payloads used to restore one full inference window.

    The REST client only needs the latest converted snapshot right now, so this
    returns a single-item list when buffered audio is available. The list shape
    keeps the reconnect/finalize interfaces aligned with older streaming-style
    replay logic.
    """
    payload = _latest_inference_snapshot_pcm()
    return [payload] if payload else []


class _InferenceState:
    """Thread-safe container holding current inference session state.

    Fields of interest:
    - `last_partial`: most recent payload from an in-session `partial_result`.
    - `last_final`: most recent payload from a session-ending `final_result`.

    `partial_result` represents ongoing live guesses from the latest window
    (updated repeatedly while capturing). `final_result` represents the
    definitive prediction when the capture session is flushed and finalized.
    Both entries are plain dict payloads received from the inference service
    and are updated under an internal lock to be safe for concurrent reads by
    the Flask UI polling thread.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.connected = False
            self.streaming = False
            self.reconnecting = False
            self.transport = 'rest'
            self.ws_url = INFERENCE_API_URL
            self.mode = INFERENCE_MODE
            self.sample_rate = None
            self.channels = None
            self.frames_per_buffer = FRAMES_PER_BUFFER
            self.send_interval_sec = SEND_INTERVAL_SEC
            self.reconnect_interval_sec = RECONNECT_RETRY_INTERVAL_SEC
            self.reconnect_max_attempts = RECONNECT_MAX_ATTEMPTS
            self.received_chunks = 0
            self.sent_payloads = 0
            self.last_payload_seconds = 0.0
            self.buffer_seconds = 0.0
            self.reconnect_count = 0
            self.reconnect_attempt = 0
            self.inference_disabled = False
            self.last_backoff_seconds = 0.0
            self.last_partial = None
            self.last_final = None
            self.last_server_event = None
            self.last_error = None
            self.last_detail = None
            self.genre_result_notes = {}
            self.prediction_trend_history = deque(maxlen=MAX_TREND_HISTORY_POINTS)

    def _record_prediction_note_locked(self, payload):
        genre = str(payload.get('genre') or '').strip()
        if not genre:
            return
        notes = self.genre_result_notes.get(genre)
        if notes is None:
            notes = deque(maxlen=MAX_GENRE_RESULT_NOTES)
            self.genre_result_notes[genre] = notes
        notes.append(
            {
                'timestamp': _local_timestamp(),
                'event': payload.get('event'),
                'genre': genre,
                'confidence': payload.get('confidence'),
                'received_chunks': payload.get('received_chunks'),
                'buffer_seconds': payload.get('buffer_seconds'),
            }
        )

    def _record_prediction_trend_locked(self, payload):
        probs = payload.get('probs')
        genre_classes = payload.get('genre_classes')
        if not isinstance(probs, list) or not isinstance(genre_classes, list):
            return
        if len(probs) != len(genre_classes):
            return
        ts_epoch = float(time.time())
        self.prediction_trend_history.append(
            {
                'timestamp': payload.get('timestamp') or _local_timestamp(),
                'ts_epoch': ts_epoch,
                'event': payload.get('event'),
                'genre': payload.get('genre'),
                'probs': probs,
                'genre_classes': genre_classes,
            }
        )

    def on_session_started(self, sample_rate, channels):
        with self._lock:
            self.connected = True
            self.streaming = True
            self.reconnecting = False
            self.sample_rate = int(sample_rate)
            self.channels = int(channels)
            self.frames_per_buffer = FRAMES_PER_BUFFER
            self.send_interval_sec = SEND_INTERVAL_SEC
            self.reconnect_interval_sec = RECONNECT_RETRY_INTERVAL_SEC
            self.reconnect_max_attempts = RECONNECT_MAX_ATTEMPTS
            self.last_error = None
            self.last_detail = None
            self.last_server_event = 'started'
            self.inference_disabled = False

    def on_payload_sent(self, payload_seconds):
        with self._lock:
            self.sent_payloads += 1
            self.last_payload_seconds = float(payload_seconds)

    def on_reconnecting(self, attempt, delay):
        with self._lock:
            self.connected = False
            self.streaming = True
            self.reconnecting = True
            self.reconnect_attempt = int(attempt)
            self.last_backoff_seconds = float(delay)
            self.last_server_event = 'reconnecting'
            self.last_detail = f'Reconnect attempt {attempt} with backoff {delay:.2f}s'

    def on_reconnected(self):
        with self._lock:
            self.connected = True
            self.streaming = True
            self.reconnecting = False
            self.reconnect_count += 1
            self.reconnect_attempt = 0
            self.last_backoff_seconds = 0.0
            self.last_server_event = 'reconnected'
            self.last_detail = 'REST inference requests resumed successfully.'
            self.last_error = None

    def on_transport_lost(self, message):
        with self._lock:
            self.connected = False
            self.streaming = True
            self.reconnecting = True
            self.last_server_event = 'transport_lost'
            self.last_detail = str(message)

    def on_reconnect_scheduled(self, delay, detail):
        with self._lock:
            self.connected = False
            self.reconnecting = True
            self.last_backoff_seconds = float(delay)
            self.last_server_event = 'retry_scheduled'
            self.last_detail = str(detail)

    def on_inference_disabled(self, detail):
        with self._lock:
            self.connected = False
            self.streaming = False
            self.reconnecting = False
            self.inference_disabled = True
            self.last_server_event = 'inference_disabled'
            self.last_detail = str(detail)
            self.last_error = str(detail)

    def on_server_message(self, payload):
        with self._lock:
            self.last_server_event = payload.get('event')
            self.received_chunks = int(payload.get('received_chunks', self.received_chunks or 0))
            self.buffer_seconds = float(payload.get('buffer_seconds', self.buffer_seconds or 0.0))
            self.last_detail = payload.get('detail', self.last_detail)
            event = payload.get('event')
            if event == 'partial_result':
                if not payload.get('timestamp'):
                    payload = dict(payload)
                    payload['timestamp'] = _local_timestamp()
                self.last_partial = payload
                self._record_prediction_note_locked(payload)
                self._record_prediction_trend_locked(payload)
                self.last_error = None
            elif event == 'final_result':
                if not payload.get('timestamp'):
                    payload = dict(payload)
                    payload['timestamp'] = _local_timestamp()
                self.last_final = payload
                self.streaming = False
                self._record_prediction_note_locked(payload)
                self._record_prediction_trend_locked(payload)
                self.last_error = None
            elif event == 'started':
                self.connected = True
                self.streaming = True
                self.reconnecting = False
                self.last_error = None
            elif event == 'error':
                self.last_error = payload.get('detail') or 'Unknown websocket error'

    def on_error(self, message):
        with self._lock:
            self.last_error = str(message)
            self.streaming = False
            self.connected = False
            self.reconnecting = False

    def on_closed(self):
        with self._lock:
            self.streaming = False
            self.connected = False
            self.reconnecting = False

    def snapshot(self):
        with self._lock:
            cutoff = time.time() - TREND_WINDOW_SECONDS
            return {
                'connected': self.connected,
                'streaming': self.streaming,
                'reconnecting': self.reconnecting,
                'transport': self.transport,
                'ws_url': self.ws_url,
                'mode': self.mode,
                'sample_rate': self.sample_rate,
                'channels': self.channels,
                'frames_per_buffer': self.frames_per_buffer,
                'send_interval_sec': self.send_interval_sec,
                'reconnect_interval_sec': self.reconnect_interval_sec,
                'reconnect_max_attempts': self.reconnect_max_attempts,
                'received_chunks': self.received_chunks,
                'sent_payloads': self.sent_payloads,
                'last_payload_seconds': self.last_payload_seconds,
                'buffer_seconds': self.buffer_seconds,
                'reconnect_count': self.reconnect_count,
                'reconnect_attempt': self.reconnect_attempt,
                'inference_disabled': self.inference_disabled,
                'last_backoff_seconds': self.last_backoff_seconds,
                'last_server_event': self.last_server_event,
                'last_error': self.last_error,
                'last_detail': self.last_detail,
                'last_partial': self.last_partial,
                'last_final': self.last_final,
                'genre_result_notes': {
                    genre: list(notes)
                    for genre, notes in self.genre_result_notes.items()
                },
                'trend_window_seconds': TREND_WINDOW_SECONDS,
                'prediction_trend_history': [
                    point for point in self.prediction_trend_history
                    if float(point.get('ts_epoch', 0.0) or 0.0) >= cutoff
                ],
            }


inference_state = _InferenceState()


class _StreamingInferenceClient:
    def __init__(self, api_url, mode, emit_interval_sec):
        self.api_url = api_url
        self.mode = mode
        self.emit_interval_sec = float(emit_interval_sec)
        self.sample_rate = None
        self.channels = None
        self.frames_per_buffer = FRAMES_PER_BUFFER
        self.health_url = self._build_health_url(api_url)
        self._session = requests.Session()
        self._replay_chunks_provider = None
        self._ever_connected = False
        self._request_count = 0
        self._failure_count = 0
        self._available = False
        self._inference_disabled = False

    @staticmethod
    def _build_health_url(api_url):
        parsed = urlsplit(api_url)
        return urlunsplit((parsed.scheme, parsed.netloc, '/health', parsed.query, parsed.fragment))

    def configure_stream(self, sample_rate, channels, frames_per_buffer):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frames_per_buffer = int(frames_per_buffer)

    def connect(self, sample_rate, channels, frames_per_buffer):
        self.configure_stream(sample_rate, channels, frames_per_buffer)
        LOGGER.info(
            "REST client connect requested api_url=%s sample_rate=%s channels=%s frames_per_buffer=%s mode=%s emit_interval_sec=%.3f",
            self.api_url,
            self.sample_rate,
            self.channels,
            self.frames_per_buffer,
            self.mode,
            self.emit_interval_sec,
        )
        self._probe_service("Initial connection scheduled.")

    def _probe_service(self, reason):
        try:
            response = self._session.get(
                self.health_url,
                timeout=(REST_CONNECT_TIMEOUT_SEC, REST_REQUEST_TIMEOUT_SEC),
            )
            response.raise_for_status()
            payload = response.json()
            LOGGER.info(
                "REST health ok api_url=%s sample_rate=%s clip_duration=%s detail=%s",
                self.api_url,
                payload.get('sample_rate'),
                payload.get('clip_duration'),
                reason,
            )
            if self._ever_connected:
                if not self._available:
                    inference_state.on_reconnected()
            else:
                inference_state.on_session_started(self.sample_rate, self.channels)
                self._ever_connected = True
            self._available = True
            self._failure_count = 0
            return True
        except Exception as exc:
            LOGGER.warning("REST health probe failed api_url=%s reason=%s error=%s", self.api_url, reason, exc)
            self._available = False
            inference_state.on_transport_lost(f'Rest health probe failed: {exc}')
            return False

    def _handle_request_failure(self, detail):
        self._available = False
        self._failure_count += 1
        LOGGER.error("REST inference request failed api_url=%s failures=%s detail=%s", self.api_url, self._failure_count, detail)
        if self._failure_count >= RECONNECT_MAX_ATTEMPTS:
            self._inference_disabled = True
            inference_state.on_inference_disabled(
                'Inference service could not be reached after '
                f'{RECONNECT_MAX_ATTEMPTS} retries. Capture will continue without inference '
                'for the rest of this session.'
            )
            return
        inference_state.on_reconnecting(self._failure_count, RECONNECT_RETRY_INTERVAL_SEC)
        inference_state.on_transport_lost(detail)
        inference_state.on_reconnect_scheduled(
            RECONNECT_RETRY_INTERVAL_SEC,
            f'{detail} Retrying REST inference on next send interval.',
        )

    def _mark_request_success(self):
        if self._ever_connected:
            if not self._available:
                inference_state.on_reconnected()
        else:
            inference_state.on_session_started(self.sample_rate, self.channels)
            self._ever_connected = True
        self._available = True
        self._failure_count = 0

    def _post_snapshot(self, pcm_bytes, event_name, replay_chunks_provider=None, captured_time_ms=None):
        if replay_chunks_provider is not None:
            self._replay_chunks_provider = replay_chunks_provider
        if self._inference_disabled:
            return False

        wav_bytes = _pcm_payload_to_wav_bytes(pcm_bytes, self.sample_rate, self.channels)
        _save_captured_clip(
            wav_bytes,
            sample_rate=int(self.sample_rate or 0),
            channels=int(self.channels or 0),
            source_tag=f"sent-{event_name}",
        )
        payload_seconds = 0.0
        if self.sample_rate and self.channels:
            payload_seconds = len(pcm_bytes) / (self.sample_rate * self.channels * 2)

        LOGGER.info(
            "REST send snapshot api_url=%s bytes=%s wav_bytes=%s approx_seconds=%.3f event=%s",
            self.api_url,
            len(pcm_bytes),
            len(wav_bytes),
            payload_seconds,
            event_name,
        )

        try:
            form_data = {'mode_form': self.mode}
            if captured_time_ms is not None:
                form_data['captured_time_ms'] = str(int(captured_time_ms))
            response = self._session.post(
                self.api_url,
                data=form_data,
                files={'file': ('live_snapshot.wav', wav_bytes, 'audio/wav')},
                timeout=(REST_CONNECT_TIMEOUT_SEC, REST_REQUEST_TIMEOUT_SEC),
            )
            response.raise_for_status()
            model_payload = response.json()
        except Exception as exc:
            self._handle_request_failure(f'REST /predict failed: {exc}')
            return False

        self._mark_request_success()
        inference_state.on_payload_sent(payload_seconds)
        self._request_count += 1

        client_payload = dict(model_payload)
        client_payload.update(
            {
                'event': event_name,
                'received_chunks': self._request_count,
                'buffer_seconds': round(payload_seconds, 3),
                'is_warmup': False,
                'detail': 'REST /predict response',
            }
        )
        LOGGER.info(
            "REST inference result api_url=%s event=%s genre=%s confidence=%.4f",
            self.api_url,
            event_name,
            client_payload.get('genre'),
            float(client_payload.get('confidence', 0.0) or 0.0),
        )
        inference_state.on_server_message(client_payload)
        return True

    def request_manual_retry(self):
        self._inference_disabled = False
        self._failure_count = 0
        inference_state.on_reconnect_scheduled(0.0, 'Manual retry requested. Retrying REST inference now.')
        self._probe_service('Manual retry requested.')
        if self._replay_chunks_provider is not None:
            replay_chunks = self._replay_chunks_provider()
            if replay_chunks:
                self._post_snapshot(
                    replay_chunks[-1],
                    event_name='partial_result',
                    captured_time_ms=_latest_inference_captured_time_ms(),
                )

    def send_chunk(self, pcm_bytes, replay_chunks_provider=None):
        return self._post_snapshot(
            pcm_bytes,
            event_name='partial_result',
            replay_chunks_provider=replay_chunks_provider,
            captured_time_ms=_latest_inference_captured_time_ms(),
        )

    def finalize(self, replay_chunks_provider=None):
        if self._replay_chunks_provider is not None and replay_chunks_provider is None:
            replay_chunks_provider = self._replay_chunks_provider
        payload = b''
        if replay_chunks_provider is not None:
            replay_chunks = replay_chunks_provider()
            if replay_chunks:
                payload = replay_chunks[-1]
        if not payload:
            return
        self._post_snapshot(
            payload,
            event_name='final_result',
            replay_chunks_provider=replay_chunks_provider,
            captured_time_ms=_latest_inference_captured_time_ms(),
        )

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass
        LOGGER.info("REST client closed api_url=%s", self.api_url)
        inference_state.on_closed()


class _CaptureManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = None
        self._is_capturing = False
        self.last_error = None
        self._level = 0.0
        self._inference_client = None

    def is_capturing(self):
        with self._lock:
            return self._is_capturing

    def start(self):
        with self._lock:
            if self._is_capturing:
                return
            self.last_error = None
            self._level = 0.0
            inference_state.reset()
            self._stop_event = threading.Event()
            self._is_capturing = True
            self._thread = threading.Thread(target=self._capture_loop, args=(self._stop_event,), daemon=True)
            self._thread.start()
        LOGGER.info("Capture start requested")

    def stop(self):
        thread = None
        stop_event = None
        with self._lock:
            if not self._is_capturing:
                return
            self._is_capturing = False
            thread = self._thread
            stop_event = self._stop_event
            self._thread = None
            self._stop_event = None

        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=2.0)
        LOGGER.info("Capture stop requested")

    def level(self):
        with self._lock:
            return float(self._level)

    def request_inference_retry(self):
        with self._lock:
            inference_client = self._inference_client
        if inference_client is None:
            return False, 'Inference retry is only available while capture is active.'
        inference_client.request_manual_retry()
        return True, 'Manual inference retry requested.'

    def _capture_loop(self, stop_event: threading.Event):
        if pyaudio is None:
            self.last_error = RuntimeError(
                f"pyaudiowpatch is required for WASAPI loopback capture: {PYAUDIO_IMPORT_ERROR}"
            )
            inference_state.on_error(self.last_error)
            LOGGER.error("Capture startup failed: %s", self.last_error)
            return

        p = pyaudio.PyAudio()
        stream = None
        inference_client = None
        pending_started_at = None
        try:
            stream, sample_rate, channels = _open_wasapi_loopback_stream(p)
            LOGGER.info(
                "Capture stream opened sample_rate=%s channels=%s frames_per_buffer=%s",
                sample_rate,
                channels,
                FRAMES_PER_BUFFER,
            )
            rolling_buffer.configure(sample_rate=sample_rate, channels=channels)
            rolling_buffer.clear()
            audio_backend.on_capture_started(sample_rate, channels)
            inference_client = _StreamingInferenceClient(INFERENCE_API_URL, INFERENCE_MODE, EMIT_INTERVAL_SEC)
            inference_client.connect(
                sample_rate=INFERENCE_SAMPLE_RATE,
                channels=INFERENCE_CHANNELS,
                frames_per_buffer=FRAMES_PER_BUFFER,
            )
            with self._lock:
                self._inference_client = inference_client

            while not stop_event.is_set():
                try:
                    data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                except OSError as exc:
                    if stop_event.is_set():
                        LOGGER.warning("Capture stream read interrupted during shutdown: %s", exc)
                    else:
                        LOGGER.warning("Capture stream read warning; closing capture loop gracefully: %s", exc)
                    break
                except Exception:
                    if stop_event.is_set():
                        LOGGER.warning("Capture stream read interrupted during shutdown", exc_info=True)
                        break
                    raise
                rolling_buffer.push(data, chunk_end_time_ms=int(time.time() * 1000))
                audio_backend.on_chunk(data, sample_rate, channels)
                if pending_started_at is None:
                    pending_started_at = time.monotonic()

                should_flush = (time.monotonic() - pending_started_at) >= SEND_INTERVAL_SEC
                if should_flush:
                    buffered_seconds = rolling_buffer.buffered_seconds()
                    if buffered_seconds >= MIN_INFERENCE_SECONDS:
                        # Upload the latest model-sized window in reduced 22050 Hz
                        # mono PCM16 format.
                        payload = _latest_inference_snapshot_pcm()
                        if payload:
                            LOGGER.info(
                                "Capture flush buffered_seconds=%.3f send_interval_sec=%.3f snapshot_bytes=%s",
                                buffered_seconds,
                                SEND_INTERVAL_SEC,
                                len(payload),
                            )
                            inference_client.send_chunk(
                                payload,
                                replay_chunks_provider=_latest_inference_replay_chunks,
                            )
                    else:
                        LOGGER.info(
                            "Capture flush skipped buffered_seconds=%.3f minimum_required=%.3f",
                            buffered_seconds,
                            MIN_INFERENCE_SECONDS,
                        )
                    pending_started_at = None

                # RMS level for UI meter (0..1). Smoothed for stability.
                try:
                    normalized = _rms_level_int16(data)
                except Exception:
                    normalized = 0.0

                with self._lock:
                    # Peak-hold with faster decay for responsiveness.
                    self._level = max(normalized, self._level * 0.60)
        except Exception as e:
            self.last_error = e
            inference_state.on_error(e)
            LOGGER.exception("Capture loop failed")
        finally:
            if inference_client is not None:
                try:
                    buffered_seconds = rolling_buffer.buffered_seconds()
                    if buffered_seconds >= MIN_INFERENCE_SECONDS:
                        payload = _latest_inference_snapshot_pcm()
                        if payload:
                            LOGGER.info(
                                "Capture final flush buffered_seconds=%.3f snapshot_bytes=%s",
                                buffered_seconds,
                                len(payload),
                            )
                            inference_client.send_chunk(
                                payload,
                                replay_chunks_provider=_latest_inference_replay_chunks,
                            )
                        inference_client.finalize(
                            replay_chunks_provider=_latest_inference_replay_chunks
                        )
                finally:
                    inference_client.close()
            with self._lock:
                self._inference_client = None
                if self._thread is threading.current_thread():
                    self._thread = None
                    self._stop_event = None
                    self._is_capturing = False
            audio_backend.on_capture_stopped()
            try:
                if stream is not None:
                    try:
                        stream.stop_stream()
                    except OSError as exc:
                        LOGGER.warning("Capture stream stop warning: %s", exc)
                    except Exception:
                        LOGGER.warning("Capture stream stop warning", exc_info=True)
                    try:
                        stream.close()
                    except OSError as exc:
                        LOGGER.warning("Capture stream close warning: %s", exc)
                    except Exception:
                        LOGGER.warning("Capture stream close warning", exc_info=True)
            finally:
                p.terminate()
            LOGGER.info("Capture stream closed")


capture_manager = _CaptureManager()


def _open_wasapi_loopback_stream(p):
    # 1. Access the Windows WASAPI host
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as e:
        raise Exception("WASAPI is not available on this system.") from e

    # 2. Get the default output device (your primary speakers/headphones)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

    # 3. Search for the loopback interface associated with those speakers
    loopback_device = None
    for loopback in p.get_loopback_device_info_generator():
        if default_speakers["name"] in loopback["name"]:
            loopback_device = loopback
            break

    if not loopback_device:
        raise Exception("Could not find WASAPI loopback device.")

    # Prefer common playback formats first for better browser compatibility.
    def _unique_positive_ints(values):
        results = []
        for v in values:
            try:
                vi = int(v)
            except (TypeError, ValueError):
                continue
            if vi > 0 and vi not in results:
                results.append(vi)
        return results

    candidate_channels = _unique_positive_ints(
        [
            *PREFERRED_CAPTURE_CHANNELS,
            default_speakers.get("maxOutputChannels"),
            loopback_device.get("maxInputChannels"),
        ]
    )
    candidate_rates = _unique_positive_ints(
        [
            loopback_device.get("defaultSampleRate"),
            default_speakers.get("defaultSampleRate"),
            *PREFERRED_CAPTURE_SAMPLE_RATES,
        ]
    )

    stream = None
    last_error = None
    chosen_rate = None
    chosen_channels = None
    attempted_pairs = []

    for rate in candidate_rates:
        for ch in candidate_channels:
            try:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                    input=True,
                    input_device_index=loopback_device["index"],
                )
                chosen_rate = rate
                chosen_channels = ch
                break
            except (ValueError, OSError) as e:
                attempted_pairs.append(f"{rate}Hz/{ch}ch -> {e}")
                last_error = e
            except Exception as e:
                attempted_pairs.append(f"{rate}Hz/{ch}ch -> {e}")
                last_error = e
        if stream is not None:
            break

    if stream is None:
        attempted_ch = ", ".join(map(str, candidate_channels)) if candidate_channels else "(none)"
        attempted_rates = ", ".join(map(str, candidate_rates)) if candidate_rates else "(none)"
        attempted_detail = " | ".join(attempted_pairs[-6:]) if attempted_pairs else repr(last_error)
        raise Exception(
            "Failed to open WASAPI loopback stream. "
            f"Attempted rates: {attempted_rates}. Attempted channels: {attempted_ch}. "
            f"Loopback device: {loopback_device.get('name')}. Last error: {last_error}. "
            f"Recent attempts: {attempted_detail}"
        )

    return stream, int(chosen_rate), int(chosen_channels)


def _wait_for_buffer_ready(timeout_s=3.0):
    # Give the capture thread a moment to start filling the buffer.
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if capture_manager.last_error is not None:
            raise capture_manager.last_error
        if rolling_buffer.snapshot_wav() is not None:
            return
        time.sleep(0.05)

    raise TimeoutError("Audio buffer not ready")


def _wait_for_capture_started(timeout_s=3.0):
    # Capture startup should mean the loopback stream opened successfully,
    # not that audible PCM has already arrived.
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if capture_manager.last_error is not None:
            raise capture_manager.last_error
        if not capture_manager.is_capturing():
            time.sleep(0.05)
            continue
        with rolling_buffer._lock:
            if rolling_buffer.sample_rate and rolling_buffer.channels:
                return
        time.sleep(0.05)

    raise TimeoutError("Capture stream did not initialize")
    
@app.route('/clip.wav')
def clip_wav():
    wav_bytes = rolling_buffer.snapshot_wav()
    if wav_bytes is None:
        msg = "Audio capture is active, but no audio has been captured yet. Play some sound, wait ~1-2s, then Stop."
        if capture_manager.last_error is not None:
            msg += f" Capture error: {capture_manager.last_error}"
        return Response(msg, status=503, mimetype="text/plain")

    with rolling_buffer._lock:
        sr = rolling_buffer.sample_rate
        ch = rolling_buffer.channels
    audio_backend.on_clip_served(wav_bytes, sr or 0, ch or 0)
    return Response(wav_bytes, mimetype="audio/wav")


@app.route('/stream.wav')
def stream_wav_compat():
    # Backward-compatible alias for older/cached pages.
    # Still serves a finite "last 18 seconds" clip (not a live stream).
    return clip_wav()


def _json_response(payload, status=200):
    return Response(json.dumps(payload), status=status, mimetype="application/json")


@app.route('/level')
def audio_level():
    return _json_response(
        {
            "capturing": capture_manager.is_capturing(),
            "level": capture_manager.level(),
        }
    )


@app.route('/capture/status')
def capture_status():
    return _json_response(
        {
            "capturing": capture_manager.is_capturing(),
            "last_error": str(capture_manager.last_error) if capture_manager.last_error else None,
        }
    )


@app.route('/capture/start', methods=['POST'])
def capture_start():
    capture_manager.start()
    try:
        _wait_for_capture_started(timeout_s=3.0)
    except Exception as e:
        capture_manager.stop()
        return _json_response({"capturing": False, "error": str(e)}, status=500)

    return _json_response({"capturing": True})


@app.route('/capture/stop', methods=['POST'])
def capture_stop():
    capture_manager.stop()
    return _json_response({"capturing": False})


@app.route('/inference/status')
def inference_status():
    payload = inference_state.snapshot()
    payload['capturing'] = capture_manager.is_capturing()
    payload['capture_error'] = str(capture_manager.last_error) if capture_manager.last_error else None
    return _json_response(payload)


@app.route('/inference/retry', methods=['POST'])
def inference_retry():
    ok, detail = capture_manager.request_inference_retry()
    payload = inference_state.snapshot()
    payload['capturing'] = capture_manager.is_capturing()
    payload['capture_error'] = str(capture_manager.last_error) if capture_manager.last_error else None
    payload['detail'] = detail
    if not ok:
        return _json_response(payload, status=400)
    return _json_response(payload)


@app.route('/demo-assets/<path:filename>')
def demo_assets(filename):
    return send_from_directory(DEMO_APP_IMAGES_DIR, filename)

@app.route('/')
def index():
    # Minimal UI: user starts/stops capture; playback plays the latest captured clip.
    falling_asset_sets = _build_falling_asset_sets()
    return render_template_string('''
        <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Music Genre Prediction</title>
                <link rel="icon" type="image/png" href="{{ app_icon_url }}" />
                <style>
                    :root {
                        --bg: #181b1f;
                        --panel: #1A2433;
                        --panel-2: #141C28;
                        --text: #F4F0F2;
                        --muted: #B7B2BC;
                        --border: #32363b;
                        --cyan: #82D7DD;
                        --blue: #5D84B6;
                        --coral: #F58F73;
                        --pink: #EEA0B8;
                        --gold: #DDB85B;
                        --spring: #8FD88B;
                        --teal-deep: #4FA7A2;
                        --cream: #E8D6C8;
                    }

                    * { box-sizing: border-box; }
                    html, body { height: 100%; }
                    body {
                        position: relative;
                        min-height: 100vh;
                        margin: 0;
                        background: radial-gradient(820px 520px at 14% 6%, rgba(238, 160, 184, 0.12), transparent 56%),
                                    radial-gradient(760px 460px at 84% 16%, rgba(143, 216, 139, 0.11), transparent 58%),
                                    radial-gradient(980px 620px at 58% 100%, rgba(93, 132, 182, 0.17), transparent 62%),
                                    radial-gradient(660px 340px at 48% 26%, rgba(245, 143, 115, 0.08), transparent 64%),
                                    var(--bg);
                        color: var(--text);
                        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
                        line-height: 1.0;
                    }

                    body::before {
                        content: "";
                        position: fixed;
                        top: -24px;
                        left: 0;
                        right: 0;
                        height: min(34vh, 300px);
                        background: linear-gradient(rgba(24, 27, 31, 0.10), rgba(24, 27, 31, 0.52)),
                                    url("{{ hero_bg_url }}") no-repeat center top / cover;
                        filter: blur(10px);
                        opacity: 0.5;
                        transform: scale(1.03);
                        transform-origin: top center;
                        pointer-events: none;
                        z-index: 0;
                    }

                    .leaf-stage {
                        position: fixed;
                        inset: 0;
                        overflow: hidden;
                        pointer-events: none;
                        opacity: 0;
                        transition: opacity 220ms ease;
                        z-index: 0;
                    }

                    body.capture-active .leaf-stage {
                        opacity: 1;
                    }

                    .leaf-set {
                        position: absolute;
                        inset: 0;
                        pointer-events: none;
                    }

                    .leaf-set div {
                        position: absolute;
                        top: -12%;
                        display: block;
                        will-change: transform, opacity;
                    }

                    .leaf-set img {
                        display: block;
                        width: 56px;
                        height: auto;
                        filter: drop-shadow(0 6px 10px rgba(0, 0, 0, 0.18));
                        user-select: none;
                        -webkit-user-drag: none;
                    }

                    .leaf-set div:nth-child(1) {
                        left: 20%;
                        animation: leaf-fall 15s linear infinite;
                        animation-delay: -7s;
                    }

                    .leaf-set div:nth-child(2) {
                        left: 50%;
                        animation: leaf-fall 20s linear infinite;
                        animation-delay: -5s;
                    }

                    .leaf-set div:nth-child(3) {
                        left: 70%;
                        animation: leaf-fall 20s linear infinite;
                        animation-delay: 0s;
                    }

                    .leaf-set div:nth-child(4) {
                        left: 0%;
                        animation: leaf-fall 15s linear infinite;
                        animation-delay: -5s;
                    }

                    .leaf-set div:nth-child(5) {
                        left: 85%;
                        animation: leaf-fall 18s linear infinite;
                        animation-delay: -10s;
                    }

                    .leaf-set div:nth-child(6) {
                        left: 32%;
                        animation: leaf-fall 17s linear infinite;
                        animation-delay: -12s;
                    }

                    .leaf-set div:nth-child(7) {
                        left: 12%;
                        animation: leaf-fall 13s linear infinite;
                        animation-delay: -3s;
                    }

                    .leaf-set div:nth-child(8) {
                        left: 64%;
                        animation: leaf-fall 16s linear infinite;
                        animation-delay: -9s;
                    }

                    .leaf-set.alt-a {
                        transform: scale(2) rotateY(180deg);
                        filter: blur(2px);
                    }

                    .leaf-set.alt-b {
                        transform: scale(0.8) rotateX(180deg);
                        filter: blur(4px);
                    }

                    @keyframes leaf-fall {
                        0% {
                            opacity: 0;
                            top: -10%;
                            transform: translateX(20px) rotate(0deg);
                        }

                        10% {
                            opacity: 1;
                        }

                        20% {
                            transform: translateX(-20px) rotate(45deg);
                        }

                        40% {
                            transform: translateX(-20px) rotate(90deg);
                        }

                        60% {
                            transform: translateX(20px) rotate(180deg);
                        }

                        80% {
                            transform: translateX(-20px) rotate(180deg);
                        }

                        100% {
                            opacity: 0;
                            top: 110%;
                            transform: translateX(-20px) rotate(225deg);
                        }
                    }

                    .wrap {
                        position: relative;
                        z-index: 1;
                        max-width: 860px;
                        margin: 0 auto;
                        padding: 10px 10px;
                    }

                    .card {
                        position: relative;
                        overflow: hidden;
                        background: linear-gradient(180deg, rgba(40, 44, 51, 0.77), rgba(40, 42, 46, 0.78));
                        border: 1px solid var(--border);
                        border-radius: 9px;
                        padding: 16px 16px;
                        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.03);
                    }

                    .card::before {
                        content: "";
                        position: absolute;
                        inset: 0;
                        background: linear-gradient(rgba(24, 27, 31, 0.88), rgba(24, 27, 31, 0.88)),
                                    url("{{ hero_bg_url }}") no-repeat center center / cover;
                        filter: blur(10px);
                        opacity: 0.4;
                        transform: scale(1.06);
                        transform-origin: center center;
                        pointer-events: none;
                        z-index: 0;
                    }

                    .card > * {
                        position: relative;
                        z-index: 1;
                    }

                    h2 {
                        margin: 0 0 6px;
                        font-size: 20px;
                        letter-spacing: 0.2px;
                    }

                    .sub {
                        margin: 0 0 18px;
                        color: var(--muted);
                        font-size: 13px;
                    }

                    .hero-card {
                        position: relative;
                        overflow: hidden;
                        background: linear-gradient(135deg, rgba(93, 132, 182, 0.14), rgba(245, 143, 115, 0.096) 42%, rgba(143, 216, 139, 0.08));
                        padding: 12px 14px;
                    }

                    .hero-card::before {
                        background: linear-gradient(rgba(24, 27, 31, 0.82), rgba(24, 27, 31, 0.82)),
                                    url("{{ hero_bg_url }}") no-repeat center center / cover;
                        opacity: 0.34;
                    }

                    .hero-card > .hero-card-overlay {
                        position: absolute;
                        inset: 0;
                        background: url("{{ hero_bg_url }}") no-repeat center center / cover;
                        opacity: 0.20;
                        pointer-events: none;
                        z-index: 0;
                    }

                    .hero-card-grid {
                        display: grid;
                        grid-template-columns: minmax(0, 1fr) auto;
                        gap: 8px;
                        align-items: center;
                        text-align: left;
                    }

                    .hero-card-copy {
                        min-width: 0;
                    }

                    .hero-card-logo-column {
                        display: flex;
                        justify-content: flex-end;
                        align-items: center;
                    }

                    .hero-logo {
                        width: min(72px, 17vw);
                        height: auto;
                        display: block;
                        margin: 0;
                    }

                    @media (max-width: 640px) {
                        .hero-card-grid {
                            grid-template-columns: 1fr;
                            gap: 6px;
                            text-align: center;
                        }

                        .hero-card-logo-column {
                            justify-content: center;
                            order: -1;
                        }
                    }

                    .hero-card::after {
                        content: "";
                        position: absolute;
                        inset: auto -40px -55px auto;
                        width: 180px;
                        height: 180px;
                        background: radial-gradient(circle, rgba(130, 215, 221, 0.18), transparent 62%);
                        pointer-events: none;
                    }

                    .row {
                        display: flex;
                        gap: 6px;
                        align-items: center;
                        justify-content: center;
                        flex-wrap: wrap;
                        margin-top: 6px;
                    }

                    button {
                        appearance: none;
                        border: 1px solid rgba(130, 215, 221, 0.55);
                        background: linear-gradient(180deg, rgba(130, 215, 221, 1), rgba(93, 191, 198, 1));
                        color: #10202B;
                        font-weight: 600;
                        border-radius: 10px;
                        padding: 10px 14px;
                        cursor: pointer;
                        box-shadow: 0 10px 24px rgba(130, 215, 221, 0.18);
                    }

                    button:hover {
                        border-color: rgba(238, 160, 184, 0.75);
                    }

                    button:disabled {
                        cursor: not-allowed;
                        opacity: 0.7;
                    }

                    audio {
                        width: min(560px, 100%);
                        border-radius: 10px;
                        border: none;
                        background: transparent;
                        box-shadow: none;
                        color-scheme: dark;
                        accent-color: var(--cyan);
                    }

                    audio::-webkit-media-controls-panel {
                        background: linear-gradient(180deg, rgba(22, 28, 36, 0.96), rgba(12, 17, 24, 0.98));
                    }

                    audio::-webkit-media-controls-enclosure {
                        border-radius: 10px;
                        background: transparent;
                    }

                    .meter {
                        width: min(560px, 100%);
                        margin: 16px auto 0;
                        text-align: left;
                    }

                    .meter-label {
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        color: var(--muted);
                        font-size: 12px;
                        margin-bottom: 8px;
                    }

                    .bar {
                        height: 12px;
                        background: rgba(232, 214, 200, 0.08);
                        border: 1px solid var(--border);
                        border-radius: 999px;
                        overflow: hidden;
                    }

                    .bar > div {
                        height: 12px;
                        width: 0%;
                        background: linear-gradient(90deg, var(--cyan), var(--spring), var(--gold), var(--pink), var(--coral));
                        transition: width 90ms linear;
                    }

                    .badge {
                        display: inline-block;
                        padding: 3px 8px;
                        border-radius: 999px;
                        font-size: 11px;
                        border: 1px solid rgba(130, 215, 221, 0.30);
                        color: var(--text);
                        background: linear-gradient(180deg, rgba(93, 132, 182, 0.22), rgba(79, 167, 162, 0.16));
                    }

                    .grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                        gap: 12px;
                        margin-top: 18px;
                        text-align: left;
                    }

                    .stat {
                        background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.022));
                        border: 1px solid var(--border);
                        border-radius: 12px;
                        padding: 12px;
                    }

                    .stat:nth-child(5n + 1) {
                        border-color: rgba(130, 215, 221, 0.28);
                    }

                    .stat:nth-child(5n + 2) {
                        border-color: rgba(93, 132, 182, 0.30);
                    }

                    .stat:nth-child(5n + 3) {
                        border-color: rgba(245, 143, 115, 0.28);
                    }

                    .stat:nth-child(5n + 4) {
                        border-color: rgba(238, 160, 184, 0.28);
                    }

                    .stat:nth-child(5n + 5) {
                        border-color: rgba(143, 216, 139, 0.28);
                    }

                    .stat .label {
                        color: var(--muted);
                        font-size: 12px;
                        margin-bottom: 6px;
                    }

                    .stat .value {
                        font-size: 16px;
                        font-weight: 600;
                    }

                    .label-inline-meta {
                        margin-left: 6px;
                        # color: var(--muted);
                        font-size: 16px;
                        font-weight: 500;
                        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                    }

                    .prediction-header {
                        opacity: 0.7;
                        margin-bottom: 12px;
                        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                    }

                    .prediction-line {
                        transition: opacity 200ms ease;
                        opacity: 1;
                        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                        font-size: 16px;
                    }

                    .result-box {
                        margin-top: 16px;
                        padding: 14px;
                        border-radius: 12px;
                        border: 1px solid var(--border);
                        background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.022));
                        text-align: left;
                    }

                    .prediction-fade-body {
                        transition: opacity 240ms ease;
                        opacity: 1;
                    }

                    .mono {
                        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                        font-size: 14px;
                    }

                    .notes-list {
                        margin: 0;
                        padding: 0;
                        list-style: none;
                        display: grid;
                        gap: 8px;
                    }

                    .notes-item {
                        padding: 10px 12px;
                        border-radius: 10px;
                        border: 1px solid rgba(130, 215, 221, 0.18);
                        background: linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(255, 255, 255, 0.02));
                    }

                    .notes-item strong {
                        display: inline-block;
                        margin-right: 8px;
                    }

                    .notes-meta {
                        color: var(--muted);
                        margin-top: 4px;
                    }

                    .trend-canvas {
                        width: 100%;
                        height: 260px;
                        display: block;
                        border-radius: 12px;
                        border: 1px solid rgba(130, 215, 221, 0.16);
                        background: linear-gradient(180deg, rgba(13, 20, 30, 0.9), rgba(18, 28, 40, 0.92));
                    }

                    .trend-legend {
                        display: flex;
                        flex-wrap: wrap;
                        gap: 8px;
                        margin-top: 12px;
                    }

                    .trend-chip {
                        display: inline-flex;
                        align-items: center;
                        gap: 6px;
                        padding: 4px 8px;
                        border-radius: 999px;
                        border: 1px solid rgba(255, 255, 255, 0.08);
                        background: rgba(255, 255, 255, 0.04);
                        color: var(--text);
                        font-size: 11px;
                    }

                    .trend-chip-swatch {
                        width: 10px;
                        height: 10px;
                        border-radius: 999px;
                        display: inline-block;
                    }
                </style>
            </head>
            <body style="font-family: sans-serif; text-align: center; margin-top: 0px;">
                <div class="leaf-stage" aria-hidden="true">
                    <div class="leaf-set">
                        {% for asset_url in falling_asset_sets[0] %}
                        <div><img src="{{ asset_url }}" alt="" /></div>
                        {% endfor %}
                    </div>
                    <div class="leaf-set alt-a">
                        {% for asset_url in falling_asset_sets[1] %}
                        <div><img src="{{ asset_url }}" alt="" /></div>
                        {% endfor %}
                    </div>
                    <div class="leaf-set alt-b">
                        {% for asset_url in falling_asset_sets[2] %}
                        <div><img src="{{ asset_url }}" alt="" /></div>
                        {% endfor %}
                    </div>
                </div>
                <div class="wrap">
                    <div class="card hero-card" style="margin-bottom: 12px; text-align: center;">
                        <div class="hero-card-overlay" aria-hidden="true"></div>
                        <div class="hero-card-grid">
                            <div class="hero-card-copy">
                                <div style="font-size: 18px; font-weight: 700; letter-spacing: 0.1px; line-height: 1.05;">🎧𝄞✮˚.⋆ Final Project: Music Genre Prediction using Log-Mel & CNN ♬⋆.˚</div>
                                <div style="font-size: 14px; margin-top: 6px;"><strong>by Nguyen Sy Hung, 2026</strong></div>
                                <div style="color: var(--muted); font-size: 14px; margin-top: 6px;"><strong>Machine Learning 1, MLE501.22, FSB | Lecturer: PhD. Truc Thi Kim, Nguyen</strong></div>
                            </div>
                            <div class="hero-card-logo-column">
                                <img class="hero-logo" src="{{ hero_logo_url }}" alt="Music Genre Prediction logo" />
                            </div>
                        </div>
                    </div>
                    <div class="card">
                        <div class="row">
                            <button id="captureBtn">Capture</button>
                            <button id="retryBtn">Retry Inference</button>
                            <span class="badge" id="wsBadge">API Idle</span>
                            <span class="badge">Mode: <span id="modeLabel">---</span></span>
                            <span class="badge" id="statusBadge">Idle</span>
                        </div>

                        <div class="meter">
                            <!-- <div class="meter-label">
                                <span>Audio Level</span>
                            </div> -->
                            <div class="bar">
                                <div id="levelBar"></div>
                            </div>
                        </div>

                        <div class="result-box">
                            <div class="label prediction-header"><strong>{{ latest_prediction_label }}</strong><span class="label-inline-meta" id="partialPredictionAge"></span></div>
                            <div class="prediction-fade-body" id="partialPredictionBody">
                                <div class="value prediction-line" style="margin-bottom: 6px;">
                                  <span id="partialPrediction">Waiting for inference...</span> *
                                  <span style="color: var(--muted); font-size: 12px;" id="partialTopK"></span>
                                </div>
                                
                                <div style="color: var(--muted);" class="mono" id="partialTimestamp">---</div>
                            </div>
                        </div>

                        <!--
                        <div class="result-box">
                            <div class="label">Latest Final Prediction</div>
                            <div class="value" id="finalPrediction">---</div>
                            <div class="mono" id="finalTopK"></div>
                            <div class="mono" id="finalTimestamp">---</div>
                        </div> -->

                        <div class="result-box">
                            <canvas id="trendCanvas" class="trend-canvas"></canvas>
                            <div class="trend-legend mono" id="trendLegend">Waiting for trend data...</div>
                        </div>

                        <div class="row" style="margin-top: 16px;">
                            <audio id="player" controls></audio>
                        </div>


                        <div class="grid">
                            <div class="stat">
                                <div class="label">Inference Mode</div>
                                <div class="value" id="inferenceMode">---</div>
                            </div>
                            <div class="stat">
                                <div class="label">Stream Format</div>
                                <div class="value" id="streamFormat">---</div>
                            </div>
                            <div class="stat">
                                <div class="label">Inference Requests</div>
                                <div class="value" id="receivedChunks">0</div>
                            </div>
                            <div class="stat">
                                <div class="label">Buffered Seconds</div>
                                <div class="value" id="bufferSeconds">0.0</div>
                            </div>
                            <div class="stat">
                                <div class="label">Send Interval</div>
                                <div class="value" id="sendInterval">---</div>
                            </div>
                            <!--<div class="stat">
                                <div class="label">Reconnects</div>
                                <div class="value" id="reconnectCount">0</div>
                            </div>-->
                            <div class="stat">
                                <div class="label">Sent Payloads</div>
                                <div class="value" id="sentPayloads">0</div>
                            </div>
                        </div>


                        <div class="result-box">
                            <div class="label">Service Status</div>
                            <div class="mono" id="serviceStatus">No messages yet.</div>
                        </div>

                        <div class="result-box">
                            <div class="label">Recent Genre Notes</div>
                            <div class="mono" id="genreNotes">No genre notes yet.</div>
                        </div>

                    </div>
                </div>

                <script>
                                    const player = document.getElementById('player');
                                    const captureBtn = document.getElementById('captureBtn');
                                    const retryBtn = document.getElementById('retryBtn');
                                    const levelBar = document.getElementById('levelBar');
                                    const statusBadge = document.getElementById('statusBadge');
                                    const wsBadge = document.getElementById('wsBadge');
                                    const modeLabel = document.getElementById('modeLabel');
                                    const inferenceMode = document.getElementById('inferenceMode');
                                    const streamFormat = document.getElementById('streamFormat');
                                    const receivedChunks = document.getElementById('receivedChunks');
                                    const bufferSeconds = document.getElementById('bufferSeconds');
                                    const sendInterval = document.getElementById('sendInterval');
                                    const reconnectCount = document.getElementById('reconnectCount');
                                    const sentPayloads = document.getElementById('sentPayloads');
                                    const partialPrediction = document.getElementById('partialPrediction');
                                    const partialPredictionBody = document.getElementById('partialPredictionBody');
                                    const partialPredictionAge = document.getElementById('partialPredictionAge');
                                    const partialTopK = document.getElementById('partialTopK');
                                    const partialTimestamp = document.getElementById('partialTimestamp');
                                    const finalPrediction = document.getElementById('finalPrediction');
                                    const finalTopK = document.getElementById('finalTopK');
                                    const finalTimestamp = document.getElementById('finalTimestamp');
                                    const serviceStatus = document.getElementById('serviceStatus');
                                    const genreNotes = document.getElementById('genreNotes');
                                    const trendCanvas = document.getElementById('trendCanvas');
                                    const trendLegend = document.getElementById('trendLegend');
                                    let capturing = false;
                                    let levelPoll = null;
                                    let inferencePoll = null;
                                    let analyserLoopRunning = false;
                                    let levelSmoothed = 0;
                                    let meterMode = 'idle'; // 'capture' | 'playback' | 'idle'
                                    let partialPredictionUpdatedAtMs = null;
                                    let partialPredictionTimestampValue = null;
                                    let partialPredictionFadeTimer = null;
                                    let partialPredictionAgeTimer = null;
                                    const PARTIAL_PREDICTION_BLINK_OPACITY = 0.6;
                                    const PARTIAL_PREDICTION_BLINK_PERIOD_MS = 130;
                                    const PARTIAL_PREDICTION_DIM_AFTER_3S_OPACITY = 0.60;
                                    const PARTIAL_PREDICTION_DIM_AFTER_6S_OPACITY = 0.40;
                                    const PARTIAL_PREDICTION_DIM_AFTER_15S_OPACITY = 0.30;
                                    const PARTIAL_PREDICTION_DIM_AFTER_45S_OPACITY = 0.20;
                                    const PARTIAL_PREDICTION_DIM_AFTER_60S_OPACITY = 0.15;
                                    const DEFAULT_LEAF_SELECTOR = '.leaf-set img';
                                    const FALLING_ASSET_LAYER_COUNT = 3;
                                    const FALLING_ASSET_IMAGES_PER_LAYER = 8;
                                    const FALLING_ASSET_LIBRARY = {
                                        'Rock': ['rock1.png', 'rock2.png', 'rock3.png', 'rock4.png'],
                                        'Metal': ['metal1.png', 'metal2.png', 'metal3.png', 'metal4.png'],
                                        'Hip-Hop': ['hip-hop1.png', 'hip-hop2.png', 'hip-hop3.png', 'hip-hop4.png'],
                                        'Pop': ['pop1.png', 'pop2.png', 'pop3.png', 'pop4.png'],
                                        'Country': ['country1.png', 'country2.png', 'country3.png', 'country4.png'],
                                        'Classical': ['classical.png', 'classical2.png', 'classical3.png', 'classical4.png'],
                                        'Jazz': ['jazz1.png', 'jazz2.png', 'jazz3.png', 'jazz4.png'],
                                        'Blues': ['blues1.png', 'blues2.png', 'blues3.png', 'blues4.png'],
                                        'Bolero': ['bolero2.png', 'bolero3.png', 'bolero4.png'],
                                        'Speech': [],
                                    };
                                    const trendPalette = ['#3574a1', '#F58F73', '#7eb37b', '#916240', '#795e8a', '#413c9e', '#852a3a', '#cf9717', '#ba2a1a', '#636363'];
                                    let defaultFallingAssetSets = [];
                                    let activeFallingGenreKey = null;
                                    let fallingAssetRequestId = 0;
                                    const fallingAssetPreloadCache = new Map();
                                    let previous_predicted_genre = null;
                                    let latest_predicted_genre = null;

                                    function parseTimestampMs(value) {
                                        if (value === null || value === undefined || value === '') return null;
                                        if (typeof value === 'number' && Number.isFinite(value)) return value;

                                        const raw = String(value).trim();
                                        if (/^\d{13}$/.test(raw)) return Number(raw);
                                        if (/^\d{10}$/.test(raw)) return Number(raw) * 1000;

                                        const localMatch = raw.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?$/);
                                        if (localMatch) {
                                            const [, year, month, day, hour, minute, second, millisecond] = localMatch;
                                            return new Date(
                                                Number(year),
                                                Number(month) - 1,
                                                Number(day),
                                                Number(hour),
                                                Number(minute),
                                                Number(second),
                                                Number((millisecond || '0').padEnd(3, '0')),
                                            ).getTime();
                                        }

                                        const parsed = Date.parse(raw);
                                        return Number.isFinite(parsed) ? parsed : null;
                                    }

                                    function formatLocalTimestamp(value, includeMilliseconds = false) {
                                        if (!value) return '---';
                                        const parsedMs = parseTimestampMs(value);
                                        if (parsedMs === null) return String(value);
                                        const parsed = new Date(parsedMs);

                                        const pad2 = number => String(number).padStart(2, '0');
                                        const base = `${parsed.getFullYear()}-${pad2(parsed.getMonth() + 1)}-${pad2(parsed.getDate())} ${pad2(parsed.getHours())}:${pad2(parsed.getMinutes())}:${pad2(parsed.getSeconds())}`;
                                        if (!includeMilliseconds) return base;
                                        return `${base}.${String(parsed.getMilliseconds()).padStart(3, '0')}`;
                                    }

                                    function formatPredictionAgeLabel(timestampValue, nowMs = Date.now()) {
                                        const timestampMs = parseTimestampMs(timestampValue);
                                        if (timestampMs === null) return '';
                                        const ageSeconds = Math.max(0, Math.floor((nowMs - timestampMs) / 1000));
                                        if (ageSeconds < 60) {
                                            return `(${ageSeconds} second${ageSeconds === 1 ? '' : 's'} ago)`;
                                        }
                                        const ageMinutes = Math.floor(ageSeconds / 60);
                                        const remainingSeconds = ageSeconds % 60;
                                        return `(${ageMinutes}m ${remainingSeconds}s ago)`;
                                    }

                                    function updatePartialPredictionAgeLabel() {
                                        if (!partialPredictionAge) return;
                                        partialPredictionAge.textContent = formatPredictionAgeLabel(partialPredictionTimestampValue);
                                    }

                                    function ensurePartialPredictionAgeTimer() {
                                        if (partialPredictionAgeTimer) return;
                                        partialPredictionAgeTimer = setInterval(updatePartialPredictionAgeLabel, 1000);
                                    }

                                    function computePartialPredictionLineOpacity(nowMs) {
                                        if (!partialPredictionUpdatedAtMs) return 1;
                                        const ageMs = Math.max(0, nowMs - partialPredictionUpdatedAtMs);
                                        if (ageMs <= 3 * 1000) {
                                            return Math.floor(nowMs / PARTIAL_PREDICTION_BLINK_PERIOD_MS) % 2 === 0
                                                ? 1
                                                : PARTIAL_PREDICTION_BLINK_OPACITY;
                                        }
                                        return 1;
                                    }

                                    function computePartialPredictionBodyOpacity(nowMs) {
                                        if (!partialPredictionUpdatedAtMs) return 1;
                                        const ageMs = Math.max(0, nowMs - partialPredictionUpdatedAtMs);
                                        if (ageMs > 60 * 1000) return PARTIAL_PREDICTION_DIM_AFTER_60S_OPACITY;
                                        if (ageMs > 45 * 1000) return PARTIAL_PREDICTION_DIM_AFTER_45S_OPACITY;
                                        if (ageMs > 15 * 1000) return PARTIAL_PREDICTION_DIM_AFTER_15S_OPACITY;
                                        if (ageMs > 6 * 1000) return PARTIAL_PREDICTION_DIM_AFTER_6S_OPACITY;
                                        if (ageMs > 3 * 1000) return PARTIAL_PREDICTION_DIM_AFTER_3S_OPACITY;
                                        return 1;
                                    }

                                    function applyPartialPredictionFade() {
                                        if (partialPrediction) {
                                            const lineOpacity = computePartialPredictionLineOpacity(Date.now());
                                            partialPrediction.style.opacity = lineOpacity.toFixed(3);
                                            if (partialTopK) {
                                                partialTopK.style.opacity = lineOpacity.toFixed(3);
                                            }
                                        }
                                        if (partialPredictionBody) {
                                            const bodyOpacity = computePartialPredictionBodyOpacity(Date.now());
                                            partialPredictionBody.style.opacity = bodyOpacity.toFixed(3);
                                        }
                                    }

                                    function ensurePartialPredictionFadeTimer() {
                                        if (partialPredictionFadeTimer) return;
                                        partialPredictionFadeTimer = setInterval(applyPartialPredictionFade, 250);
                                    }

                                    function formatTopK(items) {
                                        if (!items || !items.length) return '---';
                                        return items.map(item => `${mapGenreLabel(item.genre)} (${(item.probability * 100).toFixed(1)}%)`).join(' | ');
                                    }

                                    function formatSecondaryTopK(items) {
                                        if (!items || items.length < 2) return '---';
                                        const secondItem = items[1];
                                        return `${mapGenreLabel(secondItem.genre)} (${Number(secondItem.probability).toFixed(2)})`;
                                    }

                                    function mapGenreLabel(genre) {
                                        if (!genre && genre !== 0) return genre;
                                        const g = String(genre);
                                        // Render a friendly label for Borelo/Bolero while keeping
                                        // the original genre value in the underlying data.
                                        if (g === 'Borelo' || g === 'Bolero') return 'Borelo Việt Nam';
                                        return g;
                                    }

                                    function normalizeGenreKey(genre) {
                                        if (!genre && genre !== 0) return '';
                                        const g = String(genre).trim();
                                        if (g === 'Borelo') return 'Bolero';
                                        return g;
                                    }

                                    function formatDecoratedGenreLabel(genre) {
                                        const label = mapGenreLabel(genre) || '---';
                                        const emojis = getGenreEmojiSuffix(genre);
                                        return emojis ? `${label} ${emojis}` : label;
                                    }

                                    function getGenreEmojiSuffix(genre) {
                                        const emojiMap = {
                                            'Rock': '🤘🏼 🎸',
                                            'Metal': '🤘🏼 🎸 😈 💀',
                                            'Hip-Hop': '📀 🎧 🤸🏾 📼',
                                            'Pop': '✨ 🎤 🎵',
                                            'Country': '🎸 𐚁 👢 🐎',
                                            'Classical': '🎻',
                                            'Jazz': '🎷🎺 ♪♫',
                                            'Blues': '🎸♬🍸🌃',
                                            'Bolero': '𝄞 🎭 🍂 🌿',
                                            'Speech': '🗣 💬',
                                        };
                                        return emojiMap[normalizeGenreKey(genre)] || '';
                                    }

                                    function formatDecoratedGenreLabelHtml(genre) {
                                        const label = mapGenreLabel(genre) || '---';
                                        const emojis = getGenreEmojiSuffix(genre);
                                        return emojis ? `<strong>${label}</strong> ${emojis}` : `<strong>${label}</strong>`;
                                    }

                                    function captureDefaultFallingAssetSets() {
                                        if (defaultFallingAssetSets.length) return;
                                        defaultFallingAssetSets = Array.from(document.querySelectorAll('.leaf-set')).map(layerElement =>
                                            Array.from(layerElement.querySelectorAll('img')).map(img => img.getAttribute('src') || '')
                                        );
                                    }

                                    function primeFallingAssetLibrary() {
                                        Object.values(FALLING_ASSET_LIBRARY)
                                            .flat()
                                            .forEach(filename => preloadImage(`/demo-assets/${filename}`));
                                    }

                                    function getFallingAssetFilenamesForGenre(genre) {
                                        const key = normalizeGenreKey(genre);
                                        return FALLING_ASSET_LIBRARY[key] || [];
                                    }

                                    function expandAssetList(filenames, count) {
                                        if (!filenames.length) return [];
                                        const urls = [];
                                        for (let idx = 0; idx < count; idx++) {
                                            const filename = filenames[idx % filenames.length];
                                            urls.push(`/demo-assets/${filename}`);
                                        }
                                        return urls;
                                    }

                                    function buildFallingAssetSetsForGenre(genre) {
                                        const filenames = getFallingAssetFilenamesForGenre(genre);
                                        if (!filenames.length) return null;

                                        return Array.from({ length: FALLING_ASSET_LAYER_COUNT }, (_, layerIndex) => {
                                            const rotated = filenames.map((_, idx) => filenames[(idx + layerIndex) % filenames.length]);
                                            return expandAssetList(rotated, FALLING_ASSET_IMAGES_PER_LAYER);
                                        });
                                    }

                                    function applyFallingAssetSets(assetSets) {
                                        if (!Array.isArray(assetSets) || !assetSets.length) return;
                                        const layers = document.querySelectorAll('.leaf-set');
                                        layers.forEach((layerElement, layerIndex) => {
                                            const urls = assetSets[layerIndex] || [];
                                            const images = layerElement.querySelectorAll('img');
                                            images.forEach((img, imageIndex) => {
                                                if (urls[imageIndex]) {
                                                    img.src = urls[imageIndex];
                                                }
                                            });
                                        });
                                    }

                                    function restartFallingAnimations() {
                                        document.querySelectorAll('.leaf-set').forEach(layer => {
                                            Array.from(layer.children).forEach(div => {
                                                layer.replaceChild(div.cloneNode(true), div);
                                            });
                                        });
                                    }

                                    function restoreDefaultFallingAssets() {
                                        if (!defaultFallingAssetSets.length) return;
                                        applyFallingAssetSets(defaultFallingAssetSets);
                                        activeFallingGenreKey = null;
                                    }

                                    function preloadImage(url) {
                                        if (fallingAssetPreloadCache.has(url)) {
                                            return fallingAssetPreloadCache.get(url);
                                        }

                                        const promise = new Promise(resolve => {
                                            const image = new Image();
                                            image.onload = () => resolve(true);
                                            image.onerror = () => resolve(false);
                                            image.src = url;
                                        });
                                        fallingAssetPreloadCache.set(url, promise);
                                        return promise;
                                    }

                                    async function updateFallingAssetsForGenre(genre) {
                                        const normalizedGenre = normalizeGenreKey(genre);
                                        if (!normalizedGenre) {
                                            restoreDefaultFallingAssets();
                                            return;
                                        }
                                        if (activeFallingGenreKey === normalizedGenre) return;

                                        const assetSets = buildFallingAssetSetsForGenre(normalizedGenre);
                                        if (!assetSets) {
                                            restoreDefaultFallingAssets();
                                            return;
                                        }

                                        const requestId = ++fallingAssetRequestId;

                                        applyFallingAssetSets(assetSets);
                                        restartFallingAnimations();
                                        activeFallingGenreKey = normalizedGenre;

                                        const loaded = await Promise.all(assetSets.flat().map(preloadImage));
                                        if (requestId !== fallingAssetRequestId) return;

                                        if (!loaded.every(Boolean)) {
                                            restoreDefaultFallingAssets();
                                            return;
                                        }
                                    }

                                    function syncPredictionGenreState(genre) {
                                        latest_predicted_genre = normalizeGenreKey(genre) || null;
                                        if (latest_predicted_genre === previous_predicted_genre) return;
                                        previous_predicted_genre = latest_predicted_genre;

                                        if (!latest_predicted_genre) {
                                            restoreDefaultFallingAssets();
                                            return;
                                        }
                                        const assetSets = buildFallingAssetSetsForGenre(latest_predicted_genre);
                                        if (!assetSets) {
                                            restoreDefaultFallingAssets();
                                            return;
                                        }
                                        applyFallingAssetSets(assetSets);
                                        restartFallingAnimations();
                                        activeFallingGenreKey = latest_predicted_genre;
                                    }

                                    function renderGenreNotesMap(notesMap) {
                                        if (!notesMap || !Object.keys(notesMap).length) {
                                            genreNotes.textContent = 'No genre notes yet.';
                                            return;
                                        }

                                        const entries = Object.entries(notesMap)
                                            .flatMap(([genre, notes]) => {
                                                if (!Array.isArray(notes)) return [];
                                                return notes.map(note => ({ ...note, genre: note.genre || genre }));
                                            })
                                            .sort((a, b) => {
                                                const left = parseTimestampMs(a.timestamp) ?? 0;
                                                const right = parseTimestampMs(b.timestamp) ?? 0;
                                                return right - left;
                                            })
                                            .slice(0, 10)
                                            .map(note => {
                                                const confidence = Number(note.confidence || 0);
                                                const chunks = note.received_chunks ?? '-';
                                                const bufferSeconds = Number(note.buffer_seconds || 0).toFixed(2);
                                                return `
                                                    <li class="notes-item">
                                                        <strong>${mapGenreLabel(note.genre) || '---'}</strong>
                                                        <span>${note.event || 'result'} • ${(confidence * 100).toFixed(1)}%</span>
                                                        <div class="notes-meta">${formatLocalTimestamp(note.timestamp)} | requests ${chunks} | buffer ${bufferSeconds}s</div>
                                                    </li>
                                                `;
                                            })
                                            .join('');

                                        genreNotes.innerHTML = `<ul class="notes-list">${entries}</ul>`;
                                    }

                                    function getTrendColor(index) {
                                        return trendPalette[index % trendPalette.length];
                                    }

                                    function renderTrendPlot(history, trendWindowSeconds = 180) {
                                        if (!trendCanvas) return;

                                        const rect = trendCanvas.getBoundingClientRect();
                                        const width = Math.max(320, Math.floor(rect.width || 640));
                                        const height = Math.max(220, Math.floor(rect.height || 260));
                                        const dpr = window.devicePixelRatio || 1;
                                        trendCanvas.width = Math.floor(width * dpr);
                                        trendCanvas.height = Math.floor(height * dpr);

                                        const ctx = trendCanvas.getContext('2d');
                                        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                                        ctx.clearRect(0, 0, width, height);

                                        const padding = { top: 16, right: 18, bottom: 26, left: 36 };
                                        const plotWidth = width - padding.left - padding.right;
                                        const plotHeight = height - padding.top - padding.bottom;
                                        const nowSec = Date.now() / 1000;
                                        const points = Array.isArray(history) ? history.filter(point => Array.isArray(point.probs) && Array.isArray(point.genre_classes)) : [];

                                        ctx.fillStyle = 'rgba(244, 240, 242, 0.75)';
                                        ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

                                        if (!points.length) {
                                            ctx.fillText('Waiting for trend data...', padding.left, padding.top + 20);
                                            trendLegend.textContent = 'Waiting for trend data...';
                                            return;
                                        }

                                        const latestPoint = points[points.length - 1];
                                        const genreClasses = Array.isArray(latestPoint.genre_classes) ? latestPoint.genre_classes : [];
                                        const seriesMap = new Map(genreClasses.map((genre, idx) => [genre, { color: getTrendColor(idx), points: [] }]));

                                        for (const point of points) {
                                            const ts = Number(point.ts_epoch || 0);
                                            const ageSec = nowSec - ts;
                                            if (ageSec < 0 || ageSec > trendWindowSeconds) continue;
                                            for (let idx = 0; idx < genreClasses.length; idx++) {
                                                const genre = genreClasses[idx];
                                                const series = seriesMap.get(genre);
                                                if (!series) continue;
                                                const probability = Number(point.probs[idx] || 0);
                                                series.points.push({ ageSec, probability });
                                            }
                                        }

                                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
                                        ctx.lineWidth = 1;
                                        for (let gridIdx = 0; gridIdx <= 4; gridIdx++) {
                                            const y = padding.top + (plotHeight * gridIdx) / 4;
                                            ctx.beginPath();
                                            ctx.moveTo(padding.left, y);
                                            ctx.lineTo(width - padding.right, y);
                                            ctx.stroke();
                                            const label = `${(1 - gridIdx / 4).toFixed(2)}`;
                                            ctx.fillText(label, 4, y + 4);
                                        }
                                        for (let gridIdx = 0; gridIdx <= 6; gridIdx++) {
                                            const x = padding.left + (plotWidth * gridIdx) / 6;
                                            ctx.beginPath();
                                            ctx.moveTo(x, padding.top);
                                            ctx.lineTo(x, height - padding.bottom);
                                            ctx.stroke();
                                            const secondsAgo = Math.round(trendWindowSeconds - (trendWindowSeconds * gridIdx) / 6);
                                            ctx.fillText(`-${secondsAgo}s`, x - 12, height - 8);
                                        }

                                        for (const [genre, series] of seriesMap.entries()) {
                                            if (!series.points.length) continue;
                                            const sortedPoints = series.points.slice().sort((a, b) => b.ageSec - a.ageSec);
                                            ctx.strokeStyle = series.color;
                                            ctx.lineWidth = 2;
                                            ctx.beginPath();
                                            sortedPoints.forEach((point, idx) => {
                                                const x = padding.left + ((trendWindowSeconds - point.ageSec) / trendWindowSeconds) * plotWidth;
                                                const y = padding.top + (1 - point.probability) * plotHeight;
                                                if (idx === 0) ctx.moveTo(x, y);
                                                else ctx.lineTo(x, y);
                                            });
                                            ctx.stroke();
                                        }

                                        const legendHtml = Array.from(seriesMap.entries())
                                            .map(([genre, series]) => `<span class="trend-chip"><span class="trend-chip-swatch" style="background:${series.color}"></span>${mapGenreLabel(genre)}</span>`)
                                            .join('');
                                        trendLegend.innerHTML = legendHtml || 'Waiting for trend data...';
                                    }

                                    function updateInferenceUI(data) {
                                        wsBadge.textContent = data.connected ? 'API Ready' : 'API Unreachable';
                                        if (data.reconnecting) {
                                            wsBadge.textContent = `API Retry (${data.reconnect_attempt || 0})`;
                                        }
                                        if (data.inference_disabled) {
                                            wsBadge.textContent = 'API Disabled';
                                        }
                                        inferenceMode.textContent = data.mode || '---';
                                        const sr = data.sample_rate || '---';
                                        const ch = data.channels || '---';
                                        const fpb = data.frames_per_buffer || '---';
                                        streamFormat.textContent = `${sr} Hz / ${ch} ch / ${fpb}`;
                                        sendInterval.textContent = `${Number(data.send_interval_sec || 0).toFixed(2)} s`;
                                        receivedChunks.textContent = String(data.received_chunks || 0);
                                        sentPayloads.textContent = String(data.sent_payloads || 0);
                                        bufferSeconds.textContent = Number(data.buffer_seconds || 0).toFixed(2);
                                        if (reconnectCount) {
                                            reconnectCount.textContent = String(data.reconnect_count || 0);
                                        }
                                        retryBtn.disabled = !data.capturing || data.connected || data.reconnecting;

                                        const partial = data.last_partial;
                                        if (partial) {
                                            const warmup = partial.is_warmup ? ' [warmup]' : '';
                                            const capturedTimeValue = partial.captured_time_ms || partial.timestamp || null;
                                            partialPredictionTimestampValue = capturedTimeValue;
                                            syncPredictionGenreState(partial.genre);
                                            if (partialPrediction) {
                                                partialPrediction.innerHTML = `${formatDecoratedGenreLabelHtml(partial.genre)} (${Number(partial.confidence).toFixed(2)})${warmup}`;
                                            }
                                            if (partialTopK) {
                                                partialTopK.textContent = formatSecondaryTopK(partial.top_k);
                                            }
                                            if (partialTimestamp) {
                                                partialTimestamp.textContent = `Captured time: ${formatLocalTimestamp(capturedTimeValue, true)}`;
                                            }
                                            partialPredictionUpdatedAtMs = parseTimestampMs(capturedTimeValue) || Date.now();
                                            applyPartialPredictionFade();
                                            ensurePartialPredictionFadeTimer();
                                            updatePartialPredictionAgeLabel();
                                            ensurePartialPredictionAgeTimer();
                                        }

                                        const final = data.last_final;
                                        if (final) {
                                            if (finalPrediction) {
                                                finalPrediction.textContent = `${mapGenreLabel(final.genre)} (${(final.confidence * 100).toFixed(1)}%)`;
                                            }
                                            if (finalTopK) {
                                                finalTopK.textContent = formatTopK(final.top_k);
                                            }
                                            if (finalTimestamp) {
                                                finalTimestamp.textContent = `Captured time: ${formatLocalTimestamp(final.captured_time_ms || final.timestamp || null, true)}`;
                                            }
                                        }

                                        if (data.last_error) {
                                            serviceStatus.textContent = `Error: ${data.last_error}`;
                                        } else if (data.last_server_event) {
                                            const backoff = data.last_backoff_seconds ? ` | backoff ${Number(data.last_backoff_seconds).toFixed(2)}s` : '';
                                            serviceStatus.textContent = `Last event: ${data.last_server_event}` + (data.last_detail ? ` | ${data.last_detail}` : '') + backoff;
                                        } else {
                                            serviceStatus.textContent = 'Waiting for inference events.';
                                        }

                                        renderGenreNotesMap(data.genre_result_notes || {});
                                        renderTrendPlot(data.prediction_trend_history || [], Number(data.trend_window_seconds || 180));
                                    }

                                    function resetMeter() {
                                        levelSmoothed = 0;
                                        levelBar.style.width = '0%';
                                    }

                                    function setLevel(level01) {
                                        const target = Math.max(0, Math.min(1, level01 || 0));
                                        // Mode-aware smoothing:
                                        // - capture: responsive
                                        // - playback: smooth
                                        const attack = meterMode === 'capture' ? 0.75 : 0.35;
                                        const release = meterMode === 'capture' ? 0.30 : 0.08;
                                        const k = target > levelSmoothed ? attack : release;
                                        levelSmoothed = levelSmoothed + (target - levelSmoothed) * k;
                                        levelBar.style.width = Math.round(levelSmoothed * 100) + '%';
                                    }

                                    function setCapturingUI(isCapturing) {
                                        capturing = isCapturing;
                                        document.body.classList.toggle('capture-active', capturing);
                                        retryBtn.disabled = !capturing;
                                        if (capturing) {
                                            // Disable playback while capturing
                                            player.pause();
                                            player.style.pointerEvents = 'none';
                                            player.src = '';
                                            player.load();
                                            captureBtn.textContent = 'Stop';
                                            statusBadge.textContent = 'Capturing';
                                            modeLabel.textContent = 'CAPTURE';
                                            meterMode = 'capture';
                                            levelBar.style.transition = 'width 0ms linear';
                                            resetMeter();
                                            startLevelPolling();
                                            startInferencePolling();
                                        } else {
                                            player.style.pointerEvents = 'auto';
                                            captureBtn.textContent = 'Capture';
                                            statusBadge.textContent = 'Idle';
                                            modeLabel.textContent = '---';
                                            meterMode = 'idle';
                                            levelBar.style.transition = 'width 30ms linear';
                                            resetMeter();
                                            latest_predicted_genre = null;
                                            previous_predicted_genre = null;
                                            restoreDefaultFallingAssets();
                                            stopLevelPolling();
                                            startInferencePolling();
                                        }
                                    }

                                    async function startLevelPolling() {
                                        stopLevelPolling();
                                        levelPoll = setInterval(async () => {
                                            try {
                                                const res = await fetch('/level', { cache: 'no-store' });
                                                if (!res.ok) return;
                                                const data = await res.json();
                                                setLevel(data.level);
                                            } catch (e) {}
                                        }, 100);
                                    }

                                    function stopLevelPolling() {
                                        if (levelPoll) {
                                            clearInterval(levelPoll);
                                            levelPoll = null;
                                        }
                                    }

                                    async function startInferencePolling() {
                                        stopInferencePolling();
                                        const poll = async () => {
                                            try {
                                                const res = await fetch('/inference/status', { cache: 'no-store' });
                                                if (!res.ok) return;
                                                const data = await res.json();
                                                updateInferenceUI(data);
                                            } catch (e) {
                                                console.error('Inference status poll failed', e);
                                            }
                                        };
                                        await poll();
                                        inferencePoll = setInterval(poll, 250);
                                    }

                                    function stopInferencePolling() {
                                        if (inferencePoll) {
                                            clearInterval(inferencePoll);
                                            inferencePoll = null;
                                        }
                                    }

                                    async function postJson(url) {
                                        const res = await fetch(url, { method: 'POST' });
                                        if (!res.ok) throw new Error(await res.text());
                                        return await res.json();
                                    }

                                    async function preloadLatestClip(retries = 8, delayMs = 200) {
                                        for (let i = 0; i < retries; i++) {
                                            const url = '/clip.wav?t=' + Date.now();
                                            try {
                                                const res = await fetch(url, { method: 'GET', cache: 'no-store' });
                                                if (res.ok) {
                                                    player.src = url;
                                                    player.load();
                                                    return true;
                                                }
                                            } catch (e) {
                                                // ignore
                                            }
                                            await new Promise(r => setTimeout(r, delayMs));
                                        }
                                        return false;
                                    }

                                    captureBtn.addEventListener('click', async () => {
                                        try {
                                            if (!capturing) {
                                                setCapturingUI(true);
                                                await postJson('/capture/start');
                                            } else {
                                                await postJson('/capture/stop');
                                                setCapturingUI(false);
                                                await preloadLatestClip();
                                            }
                                        } catch (e) {
                                            setCapturingUI(false);
                                            alert('Capture error: ' + (e && e.message ? e.message : e));
                                        }
                                    });

                                    retryBtn.addEventListener('click', async () => {
                                        try {
                                            const data = await postJson('/inference/retry');
                                            updateInferenceUI(data);
                                        } catch (e) {
                                            alert('Inference retry error: ' + (e && e.message ? e.message : e));
                                        }
                                    });

                                    // While playing, show playback level using WebAudio analyser.
                                    async function startPlaybackAnalyser() {
                                        if (analyserLoopRunning) return;
                                        analyserLoopRunning = true;

                                        let audioCtx;
                                        try {
                                            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                                            const source = audioCtx.createMediaElementSource(player);
                                            const analyser = audioCtx.createAnalyser();
                                            analyser.fftSize = 256;
                                            analyser.smoothingTimeConstant = 0.6;
                                            source.connect(analyser);
                                            analyser.connect(audioCtx.destination);
                                            const buf = new Uint8Array(analyser.frequencyBinCount);

                                            const tick = () => {
                                                if (player.paused || player.ended) {
                                                    analyserLoopRunning = false;
                                                    try { audioCtx.close(); } catch (e) {}
                                                    return;
                                                }
                                                analyser.getByteTimeDomainData(buf);
                                                let sum = 0;
                                                for (let i = 0; i < buf.length; i++) {
                                                    const x = (buf[i] - 128) / 128;
                                                    sum += x * x;
                                                }
                                                const rms = Math.sqrt(sum / buf.length);
                                                setLevel(rms);
                                                requestAnimationFrame(tick);
                                            };
                                            requestAnimationFrame(tick);
                                        } catch (e) {
                                            analyserLoopRunning = false;
                                            if (audioCtx) {
                                                try { audioCtx.close(); } catch (e2) {}
                                            }
                                        }
                                    }

                                    player.addEventListener('playing', () => {
                                        if (!capturing) startPlaybackAnalyser();
                                        statusBadge.textContent = 'Playing';
                                        modeLabel.textContent = 'PLAYBACK';
                                        meterMode = 'playback';
                                        levelBar.style.transition = 'width 90ms linear';
                                    });

                                    player.addEventListener('pause', () => {
                                        // stop displaying level when playback stops
                                        if (!capturing) {
                                            statusBadge.textContent = 'Idle';
                                            modeLabel.textContent = '---';
                                            meterMode = 'idle';
                                            resetMeter();
                                        }
                                    });

                                    player.addEventListener('ended', () => {
                                        // stop displaying level when playback ends
                                        if (!capturing) {
                                            statusBadge.textContent = 'Idle';
                                            modeLabel.textContent = '---';
                                            meterMode = 'idle';
                                            resetMeter();
                                        }
                                    });

                                    // Initial state: not capturing; playback enabled.
                                    captureDefaultFallingAssetSets();
                                    primeFallingAssetLibrary();
                                    restoreDefaultFallingAssets();
                                    setCapturingUI(false);
                                    applyPartialPredictionFade();
                                    ensurePartialPredictionFadeTimer();
                                    startInferencePolling();
                </script>
            </body>
        </html>
    ''', app_icon_url=_demo_asset_url('app_icon.png'), hero_logo_url=_demo_asset_url('headphone1.png'), hero_bg_url=_demo_asset_url('hero_background.png'), falling_asset_sets=falling_asset_sets, latest_prediction_label=f"Latest {_format_seconds_label(MIN_INFERENCE_SECONDS)}-Seconds Audio Prediction")

if __name__ == "__main__":
    # Threaded=True is required so the web server can serve the UI and the stream simultaneously
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("werkzeug").addFilter(_SuppressLevelEndpointFilter())
    app.run(host="0.0.0.0", port=5000, threaded=True)