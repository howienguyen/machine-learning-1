# Run guide:
# 1. Start the inference service first, for example:
#    python MelCNN-MGR/inference_web_service/app.py --model-dir MelCNN-MGR/models/logmel-cnn-v2_1-YYYYMMDD-HHMMSS
# 2. Then start this Flask capture app, optionally overriding the WebSocket target or batching interval:
#    MELCNN_INFERENCE_WS_URL=ws://127.0.0.1:8000/ws/stream MELCNN_INFERENCE_SEND_INTERVAL_SEC=3 python MelCNN-MGR/demo-app/web_audio_capture_v1.py
# 3. Open http://127.0.0.1:5000 in a browser and use Capture/Stop to stream system audio to the inference service.


import struct
import threading
import time
from collections import deque
import json
import os
import array
import math
import sys
import logging
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from flask import Flask, Response, render_template_string
try:
    import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]
    PYAUDIO_IMPORT_ERROR = None
except Exception as exc:
    pyaudio = None
    PYAUDIO_IMPORT_ERROR = exc
import websocket

import audio_backend

# Local capture chunk size in frames returned by the WASAPI loopback stream.
FRAMES_PER_BUFFER = 4096
# Rolling local clip length kept for playback/reconnect replay purposes.
CLIP_SECONDS = 18
# Preferred capture sample rate; stream opening falls back to device-supported rates if needed.
TARGET_SAMPLE_RATE = 22050
# Inference service WebSocket endpoint consumed by this capture client.
INFERENCE_WS_URL = os.environ.get("MELCNN_INFERENCE_WS_URL", "ws://127.0.0.1:8000/ws/stream")
# Inference mode requested from the server for each stream session.
INFERENCE_MODE = os.environ.get("MELCNN_INFERENCE_MODE", "three_crop")
# Server-side partial-result cadence once audio has arrived at the backend.
EMIT_INTERVAL_SEC = float(os.environ.get("MELCNN_INFERENCE_EMIT_INTERVAL_SEC", "3.0"))
# Client-side batching cadence before buffered PCM is flushed to the WebSocket.
SEND_INTERVAL_SEC = float(os.environ.get("MELCNN_INFERENCE_SEND_INTERVAL_SEC", "3.0"))
# Background retry cadence used when the inference service is unavailable.
RECONNECT_RETRY_INTERVAL_SEC = float(os.environ.get("MELCNN_INFERENCE_RECONNECT_INTERVAL_SEC", "10.0"))
# Maximum number of consecutive reconnect attempts before inference is disabled for the current capture session.
RECONNECT_MAX_ATTEMPTS = int(os.environ.get("MELCNN_INFERENCE_RECONNECT_MAX_ATTEMPTS", "6"))
# Amount of recent PCM to replay after reconnect so the server can rebuild context.
RECONNECT_REPLAY_SECONDS = 12.0

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

class _RollingAudioBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._chunks = deque()
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

    def clear(self):
        with self._lock:
            self._chunks.clear()

    def push(self, pcm_bytes):
        with self._lock:
            self._chunks.append(pcm_bytes)

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


rolling_buffer = _RollingAudioBuffer()


class _InferenceState:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.connected = False
            self.streaming = False
            self.reconnecting = False
            self.ws_url = INFERENCE_WS_URL
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
            self.last_detail = 'WebSocket stream reconnected and session restarted.'

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
                self.last_partial = payload
            elif event == 'final_result':
                self.last_final = payload
                self.streaming = False
            elif event == 'started':
                self.connected = True
                self.streaming = True
                self.reconnecting = False
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
            return {
                'connected': self.connected,
                'streaming': self.streaming,
                'reconnecting': self.reconnecting,
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
            }


inference_state = _InferenceState()


class _StreamingInferenceClient:
    def __init__(self, ws_url, mode, emit_interval_sec):
        self.ws_url = ws_url
        self.mode = mode
        self.emit_interval_sec = float(emit_interval_sec)
        self.sample_rate = None
        self.channels = None
        self.frames_per_buffer = FRAMES_PER_BUFFER
        self.stream_id_base = f'web-audio-capture-{int(time.time())}'
        self.stream_generation = 0
        self._ws = None
        self._recv_thread = None
        self._reconnect_thread = None
        self._recv_stop = threading.Event()
        self._retry_wakeup = threading.Event()
        self._lock = threading.Lock()
        self._reconnect_lock = threading.Lock()
        self._replay_chunks_provider = None
        self._ever_connected = False
        self._next_retry_monotonic = 0.0
        self._reconnect_attempts = 0
        self._inference_disabled = False

    def configure_stream(self, sample_rate, channels, frames_per_buffer):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.frames_per_buffer = int(frames_per_buffer)

    def connect(self, sample_rate, channels, frames_per_buffer):
        self.configure_stream(sample_rate, channels, frames_per_buffer)
        self._ensure_reconnect_loop("Initial connection scheduled.")

    def _start_recv_thread(self):
        self._recv_stop.clear()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def _open_session(self):
        ws = websocket.create_connection(self.ws_url, timeout=5)
        ws.settimeout(0.5)

        try:
            hello = ws.recv()
            if isinstance(hello, str):
                inference_state.on_server_message(json.loads(hello))
        except Exception:
            pass

        stream_id = f'{self.stream_id_base}-r{self.stream_generation}'
        self.stream_generation += 1
        start_payload = {
            'type': 'start',
            'stream_id': stream_id,
            'encoding': 'pcm_s16le',
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'frames_per_buffer': self.frames_per_buffer,
            'mode': self.mode,
            'emit_interval_sec': self.emit_interval_sec,
        }
        ws.send(json.dumps(start_payload))

        with self._lock:
            self._ws = ws
        self._start_recv_thread()

    def _clear_connection(self):
        with self._lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _sleep_until_or_stop(self, deadline_monotonic):
        while not self._recv_stop.is_set() and time.monotonic() < deadline_monotonic:
            if self._retry_wakeup.is_set():
                self._retry_wakeup.clear()
                return
            time.sleep(0.1)

    def _ensure_reconnect_loop(self, reason, replay_chunks_provider=None):
        if replay_chunks_provider is not None:
            self._replay_chunks_provider = replay_chunks_provider
        if self._recv_stop.is_set() or self._inference_disabled:
            return
        with self._reconnect_lock:
            if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
                return
            self._reconnect_thread = threading.Thread(
                target=self._reconnect_loop,
                args=(reason,),
                daemon=True,
            )
            self._reconnect_thread.start()

    def request_manual_retry(self):
        self._inference_disabled = False
        self._reconnect_attempts = 0
        self._next_retry_monotonic = 0.0
        self._retry_wakeup.set()
        inference_state.on_reconnect_scheduled(0.0, 'Manual retry requested. Reconnecting now.')
        self._ensure_reconnect_loop('Manual retry requested.')

    def _reconnect_loop(self, reason):
        while not self._recv_stop.is_set():
            if self._inference_disabled:
                return
            ws = self._ws
            if ws is not None:
                return

            now = time.monotonic()
            deadline = max(now, self._next_retry_monotonic)
            delay = max(0.0, deadline - now)
            inference_state.on_reconnect_scheduled(
                delay,
                f'{reason} Retrying websocket connection in {max(delay, 0.0):.1f}s.',
            )
            if delay > 0:
                self._sleep_until_or_stop(deadline)
                if self._recv_stop.is_set():
                    return

            try:
                self._open_session()
                ws = self._ws
                if ws is None:
                    raise RuntimeError('Reconnect opened no websocket session.')
                replay_chunks = self._replay_chunks_provider() if self._replay_chunks_provider is not None else []
                for chunk in replay_chunks:
                    ws.send_binary(chunk)
                self._next_retry_monotonic = 0.0
                if self._ever_connected:
                    inference_state.on_reconnected()
                else:
                    inference_state.on_session_started(self.sample_rate, self.channels)
                    self._ever_connected = True
                self._reconnect_attempts = 0
                return
            except Exception as exc:
                self._clear_connection()
                self._reconnect_attempts += 1
                if self._reconnect_attempts >= RECONNECT_MAX_ATTEMPTS:
                    self._inference_disabled = True
                    inference_state.on_inference_disabled(
                        'Inference service could not be reached after '
                        f'{RECONNECT_MAX_ATTEMPTS} retries. Capture will continue without inference '
                        'for the rest of this session.'
                    )
                    return
                self._next_retry_monotonic = time.monotonic() + RECONNECT_RETRY_INTERVAL_SEC
                inference_state.on_transport_lost(
                    f'WebSocket connect failed: {exc}. Retry {self._reconnect_attempts}/{RECONNECT_MAX_ATTEMPTS} '
                    f'in {RECONNECT_RETRY_INTERVAL_SEC:.1f}s.'
                )

    def _recv_loop(self):
        while not self._recv_stop.is_set():
            if self._ws is None:
                return
            try:
                message = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                if not self._recv_stop.is_set():
                    inference_state.on_transport_lost(f'WebSocket receive failed: {exc}')
                    self._clear_connection()
                    self._ensure_reconnect_loop(
                        'Receive path lost websocket transport.',
                        replay_chunks_provider=self._replay_chunks_provider,
                    )
                return

            if not message:
                continue
            if isinstance(message, bytes):
                continue
            try:
                payload = json.loads(message)
            except Exception:
                continue
            inference_state.on_server_message(payload)

    def send_chunk(self, pcm_bytes, replay_chunks_provider=None):
        if self._inference_disabled:
            return False
        ws = self._ws
        if ws is None:
            self._ensure_reconnect_loop(
                'Inference service is unavailable.',
                replay_chunks_provider=replay_chunks_provider,
            )
            return False
        try:
            ws.send_binary(pcm_bytes)
            payload_seconds = 0.0
            if self.sample_rate and self.channels:
                payload_seconds = len(pcm_bytes) / (self.sample_rate * self.channels * 2)
            inference_state.on_payload_sent(payload_seconds)
            return True
        except Exception as exc:
            inference_state.on_transport_lost(f'WebSocket send failed: {exc}')
            self._clear_connection()
            self._ensure_reconnect_loop(
                'Send path lost websocket transport.',
                replay_chunks_provider=replay_chunks_provider,
            )
            return False

    def finalize(self, replay_chunks_provider=None):
        if self._ws is None:
            return
        try:
            self._ws.send(json.dumps({'type': 'stop'}))
            deadline = time.time() + 2.0
            while time.time() < deadline:
                snap = inference_state.snapshot()
                if snap.get('last_final') or snap.get('last_error'):
                    break
                time.sleep(0.05)
        except Exception as exc:
            inference_state.on_error(exc)

    def close(self):
        self._recv_stop.set()
        self._retry_wakeup.set()
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=1.0)
        if self._reconnect_thread is not None:
            self._reconnect_thread.join(timeout=1.0)
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
            return

        p = pyaudio.PyAudio()
        stream = None
        inference_client = None
        pending_chunks = []
        pending_started_at = None
        try:
            stream, sample_rate, channels = _open_wasapi_loopback_stream(p)
            rolling_buffer.configure(sample_rate=sample_rate, channels=channels)
            rolling_buffer.clear()
            audio_backend.on_capture_started(sample_rate, channels)
            inference_client = _StreamingInferenceClient(INFERENCE_WS_URL, INFERENCE_MODE, EMIT_INTERVAL_SEC)
            inference_client.connect(sample_rate=sample_rate, channels=channels, frames_per_buffer=FRAMES_PER_BUFFER)
            with self._lock:
                self._inference_client = inference_client

            while not stop_event.is_set():
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                rolling_buffer.push(data)
                audio_backend.on_chunk(data, sample_rate, channels)
                pending_chunks.append(data)
                if pending_started_at is None:
                    pending_started_at = time.monotonic()

                should_flush = (time.monotonic() - pending_started_at) >= SEND_INTERVAL_SEC
                if should_flush:
                    payload = b''.join(pending_chunks)
                    inference_client.send_chunk(
                        payload,
                        replay_chunks_provider=lambda: rolling_buffer.snapshot_chunks(RECONNECT_REPLAY_SECONDS),
                    )
                    pending_chunks = []
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
        finally:
            if inference_client is not None:
                try:
                    if pending_chunks:
                        inference_client.send_chunk(
                            b''.join(pending_chunks),
                            replay_chunks_provider=lambda: rolling_buffer.snapshot_chunks(RECONNECT_REPLAY_SECONDS),
                        )
                    inference_client.finalize(
                        replay_chunks_provider=lambda: rolling_buffer.snapshot_chunks(RECONNECT_REPLAY_SECONDS)
                    )
                finally:
                    inference_client.close()
            with self._lock:
                self._inference_client = None
            audio_backend.on_capture_stopped()
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            finally:
                p.terminate()


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
            2,
            1,
            loopback_device.get("maxInputChannels"),
            default_speakers.get("maxOutputChannels"),
        ]
    )
    candidate_rates = _unique_positive_ints(
        [
            TARGET_SAMPLE_RATE,
            loopback_device.get("defaultSampleRate"),
            default_speakers.get("defaultSampleRate"),
            48000,
            44100,
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
    
@app.route('/clip.wav')
def clip_wav():
    wav_bytes = rolling_buffer.snapshot_wav()
    if wav_bytes is None:
        msg = "Audio buffer not ready. Click Capture, wait ~1-2s, then Stop."
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
        _wait_for_buffer_ready(timeout_s=3.0)
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

@app.route('/')
def index():
    # Minimal UI: user starts/stops capture; playback plays the last 18 seconds captured.
    return render_template_string('''
        <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Music Genre Prediction</title>
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
                        line-height: 1.4;
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
                        left: 20%;
                        animation: leaf-fall 15s linear infinite;
                        animation-delay: -7s;
                    }

                    .leaf-set div:nth-child(7) {
                        left: 0%;
                        animation: leaf-fall 12s linear infinite;
                    }

                    .leaf-set div:nth-child(8) {
                        left: 60%;
                        animation: leaf-fall 15s linear infinite;
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
                        max-width: 760px;
                        margin: 0 auto;
                        padding: 40px 16px;
                    }

                    .card {
                        background: linear-gradient(180deg, rgba(40, 44, 51, 0.96), rgba(40, 42, 46, 0.97));
                        border: 1px solid var(--border);
                        border-radius: 9px;
                        padding: 20px 18px;
                        box-shadow: 0 18px 40px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.03);
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
                        background: linear-gradient(135deg, rgba(93, 132, 182, 0.18), rgba(245, 143, 115, 0.12) 42%, rgba(143, 216, 139, 0.10));
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
                        gap: 12px;
                        align-items: center;
                        justify-content: center;
                        flex-wrap: wrap;
                        margin-top: 14px;
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
                        border: 1px solid var(--border);
                        background: linear-gradient(180deg, rgba(50, 70, 95, 0.34), rgba(20, 28, 40, 0.55));
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

                    .result-box {
                        margin-top: 16px;
                        padding: 14px;
                        border-radius: 12px;
                        border: 1px solid var(--border);
                        background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.022));
                        text-align: left;
                    }

                    .mono {
                        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                        font-size: 12px;
                    }
                </style>
            </head>
            <body style="font-family: sans-serif; text-align: center; margin-top: 0px;">
                <div class="leaf-stage" aria-hidden="true">
                    <div class="leaf-set">
                        <div><img src="https://i.ibb.co/M59443B/leaves1.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/v1WGv6b/leaves2.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/V3KSBdV/leaves3.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/jkGMYLM/leaves4.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/M59443B/leaves1.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/v1WGv6b/leaves2.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/V3KSBdV/leaves3.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/jkGMYLM/leaves4.png" alt="" /></div>
                    </div>
                    <div class="leaf-set alt-a">
                        <div><img src="https://i.ibb.co/M59443B/leaves1.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/v1WGv6b/leaves2.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/V3KSBdV/leaves3.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/jkGMYLM/leaves4.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/M59443B/leaves1.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/v1WGv6b/leaves2.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/V3KSBdV/leaves3.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/jkGMYLM/leaves4.png" alt="" /></div>
                    </div>
                    <div class="leaf-set alt-b">
                        <div><img src="https://i.ibb.co/M59443B/leaves1.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/v1WGv6b/leaves2.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/V3KSBdV/leaves3.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/jkGMYLM/leaves4.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/M59443B/leaves1.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/v1WGv6b/leaves2.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/V3KSBdV/leaves3.png" alt="" /></div>
                        <div><img src="https://i.ibb.co/jkGMYLM/leaves4.png" alt="" /></div>
                    </div>
                </div>
                <div class="wrap">
                    <div class="card hero-card" style="margin-bottom: 18px; text-align: center;">
                        <div style="font-size: 15px; font-weight: 700; letter-spacing: 0.1px;">🎧𝄞💿✮˚.⋆ Machine Learning 1 – Final Project: Music Genre Prediction ♬⋆.˚</div>
                        <div style="color: var(--muted); font-size: 13px; margin-top: 4px;">By Nguyen Sy Hung, 2026</div>
                    </div>
                    <div class="card">
                        <h2>System Audio Capture + Streaming Inference</h2>
                        <p class="sub">Capture system audio via WASAPI loopback, stream raw PCM16 little-endian chunks over WebSocket, and show live genre predictions from the inference web service.</p>

                        <div class="row">
                            <button id="captureBtn">Capture</button>
                            <button id="retryBtn">Retry Inference</button>
                            <span class="badge" id="statusBadge">Idle</span>
                            <span class="badge" id="wsBadge">WS Disconnected</span>
                        </div>

                        <div class="row" style="margin-top: 16px;">
                            <audio id="player" controls></audio>
                        </div>

                        <div class="meter">
                            <div class="meter-label">
                                <span>Audio Level</span>
                                <span id="modeLabel">---</span>
                            </div>
                            <div class="bar">
                                <div id="levelBar"></div>
                            </div>
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
                                <div class="label">Received Chunks</div>
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
                            <div class="stat">
                                <div class="label">Reconnects</div>
                                <div class="value" id="reconnectCount">0</div>
                            </div>
                            <div class="stat">
                                <div class="label">Sent Payloads</div>
                                <div class="value" id="sentPayloads">0</div>
                            </div>
                        </div>

                        <div class="result-box">
                            <div class="label">Latest Partial Prediction</div>
                            <div class="value" id="partialPrediction">Waiting for stream...</div>
                            <div class="mono" id="partialTopK"></div>
                        </div>

                        <div class="result-box">
                            <div class="label">Latest Final Prediction</div>
                            <div class="value" id="finalPrediction">---</div>
                            <div class="mono" id="finalTopK"></div>
                        </div>

                        <div class="result-box">
                            <div class="label">Service / Error Status</div>
                            <div class="mono" id="serviceStatus">No messages yet.</div>
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
                                    const partialTopK = document.getElementById('partialTopK');
                                    const finalPrediction = document.getElementById('finalPrediction');
                                    const finalTopK = document.getElementById('finalTopK');
                                    const serviceStatus = document.getElementById('serviceStatus');
                                    let capturing = false;
                                    let levelPoll = null;
                                    let inferencePoll = null;
                                    let analyserLoopRunning = false;
                                    let levelSmoothed = 0;
                                    let meterMode = 'idle'; // 'capture' | 'playback' | 'idle'

                                    function formatTopK(items) {
                                        if (!items || !items.length) return '---';
                                        return items.map(item => `${item.genre} (${(item.probability * 100).toFixed(1)}%)`).join(' | ');
                                    }

                                    function updateInferenceUI(data) {
                                        wsBadge.textContent = data.connected ? 'WS Connected' : 'WS Disconnected';
                                        if (data.reconnecting) {
                                            wsBadge.textContent = `WS Reconnecting (${data.reconnect_attempt || 0})`;
                                        }
                                        if (data.inference_disabled) {
                                            wsBadge.textContent = 'Inference Disabled';
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
                                        reconnectCount.textContent = String(data.reconnect_count || 0);
                                        retryBtn.disabled = !data.capturing || data.connected || data.reconnecting;

                                        const partial = data.last_partial;
                                        if (partial) {
                                            const warmup = partial.is_warmup ? ' [warmup]' : '';
                                            partialPrediction.textContent = `${partial.genre} (${(partial.confidence * 100).toFixed(1)}%)${warmup}`;
                                            partialTopK.textContent = formatTopK(partial.top_k);
                                        }

                                        const final = data.last_final;
                                        if (final) {
                                            finalPrediction.textContent = `${final.genre} (${(final.confidence * 100).toFixed(1)}%)`;
                                            finalTopK.textContent = formatTopK(final.top_k);
                                        }

                                        if (data.last_error) {
                                            serviceStatus.textContent = `Error: ${data.last_error}`;
                                        } else if (data.last_server_event) {
                                            const backoff = data.last_backoff_seconds ? ` | backoff ${Number(data.last_backoff_seconds).toFixed(2)}s` : '';
                                            serviceStatus.textContent = `Last event: ${data.last_server_event}` + (data.last_detail ? ` | ${data.last_detail}` : '') + backoff;
                                        } else {
                                            serviceStatus.textContent = 'Waiting for websocket events.';
                                        }
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
                                            } catch (e) {}
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
                                    setCapturingUI(false);
                                    startInferencePolling();
                </script>
            </body>
        </html>
    ''')

if __name__ == "__main__":
    # Threaded=True is required so the web server can serve the UI and the stream simultaneously
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("werkzeug").addFilter(_SuppressLevelEndpointFilter())
    app.run(host="0.0.0.0", port=5000, threaded=True)