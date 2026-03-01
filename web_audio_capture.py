import struct
import threading
import time
from collections import deque
import json
import array
import math
import sys
import logging

from flask import Flask, Response, render_template_string
import pyaudiowpatch as pyaudio

FRAMES_PER_BUFFER = 4096
CLIP_SECONDS = 18

app = Flask(__name__)


class _SuppressLevelEndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        # Werkzeug logs look like: "127.0.0.1 - - [..] \"GET /level HTTP/1.1\" 200 -"
        return "GET /level" not in msg


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


rolling_buffer = _RollingAudioBuffer()


class _CaptureManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = None
        self._is_capturing = False
        self.last_error = None
        self._level = 0.0

    def is_capturing(self):
        with self._lock:
            return self._is_capturing

    def start(self):
        with self._lock:
            if self._is_capturing:
                return
            self.last_error = None
            self._level = 0.0
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

    def _capture_loop(self, stop_event: threading.Event):
        p = pyaudio.PyAudio()
        stream = None
        try:
            stream, sample_rate, channels = _open_wasapi_loopback_stream(p)
            rolling_buffer.configure(sample_rate=sample_rate, channels=channels)
            rolling_buffer.clear()

            while not stop_event.is_set():
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                rolling_buffer.push(data)

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
        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            finally:
                p.terminate()


capture_manager = _CaptureManager()


def _open_wasapi_loopback_stream(p: pyaudio.PyAudio):
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
            except ValueError as e:
                last_error = e
        if stream is not None:
            break

    if stream is None:
        attempted_ch = ", ".join(map(str, candidate_channels)) if candidate_channels else "(none)"
        attempted_rates = ", ".join(map(str, candidate_rates)) if candidate_rates else "(none)"
        raise Exception(
            "Failed to open WASAPI loopback stream. "
            f"Attempted rates: {attempted_rates}. Attempted channels: {attempted_ch}. Last error: {last_error}"
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

@app.route('/')
def index():
    # Minimal UI: user starts/stops capture; playback plays the last 18 seconds captured.
    return render_template_string('''
        <html>
            <head>
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1" />
                <title>Last 18 Seconds Recorder</title>
                <style>
                    :root {
                        --bg: #0B0F14;
                        --panel: #101823;
                        --panel-2: #0E1620;
                        --text: #E7EEF6;
                        --muted: #9BB0C2;
                        --border: #213041;
                        --teal: #006A80;
                        --orange: #FF8A00;
                    }

                    * { box-sizing: border-box; }
                    html, body { height: 100%; }
                    body {
                        margin: 0;
                        background: radial-gradient(1200px 600px at 20% 10%, rgba(0, 106, 128, 0.22), transparent 55%),
                                    radial-gradient(900px 500px at 80% 20%, rgba(255, 138, 0, 0.10), transparent 60%),
                                    var(--bg);
                        color: var(--text);
                        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
                        line-height: 1.4;
                    }

                    .wrap {
                        max-width: 760px;
                        margin: 0 auto;
                        padding: 40px 16px;
                    }

                    .card {
                        background: linear-gradient(180deg, rgba(16, 24, 35, 0.92), rgba(14, 22, 32, 0.92));
                        border: 1px solid var(--border);
                        border-radius: 16px;
                        padding: 20px 18px;
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
                        border: 1px solid rgba(0, 106, 128, 0.45);
                        background: linear-gradient(180deg, rgba(0, 106, 128, 1), rgba(0, 86, 104, 1));
                        color: #FFFFFF;
                        font-weight: 600;
                        border-radius: 10px;
                        padding: 10px 14px;
                        cursor: pointer;
                    }

                    button:hover {
                        border-color: rgba(255, 138, 0, 0.55);
                    }

                    button:disabled {
                        cursor: not-allowed;
                        opacity: 0.7;
                    }

                    audio {
                        width: min(560px, 100%);
                        border-radius: 10px;
                        border: 1px solid var(--border);
                        background: rgba(0, 0, 0, 0.25);
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
                        background: rgba(231, 238, 246, 0.06);
                        border: 1px solid var(--border);
                        border-radius: 999px;
                        overflow: hidden;
                    }

                    .bar > div {
                        height: 12px;
                        width: 0%;
                        background: linear-gradient(90deg, var(--teal), var(--orange));
                        transition: width 90ms linear;
                    }

                    .badge {
                        display: inline-block;
                        padding: 3px 8px;
                        border-radius: 999px;
                        font-size: 11px;
                        border: 1px solid rgba(0, 106, 128, 0.45);
                        color: var(--text);
                        background: rgba(0, 106, 128, 0.12);
                    }
                </style>
            </head>
            <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
                <div class="wrap">
                    <div class="card">
                        <h2>Last 18 Seconds Recorder</h2>
                        <p class="sub">Capture system audio silently, then play back the most recent 18 seconds.</p>

                        <div class="row">
                            <button id="captureBtn">Capture</button>
                            <span class="badge" id="statusBadge">Idle</span>
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
                    </div>
                </div>

                <script>
                                    const player = document.getElementById('player');
                                    const captureBtn = document.getElementById('captureBtn');
                                    const levelBar = document.getElementById('levelBar');
                                    const statusBadge = document.getElementById('statusBadge');
                                    const modeLabel = document.getElementById('modeLabel');
                                    let capturing = false;
                                    let levelPoll = null;
                                    let analyserLoopRunning = false;
                                    let levelSmoothed = 0;
                                    let meterMode = 'idle'; // 'capture' | 'playback' | 'idle'

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
                                        } else {
                                            player.style.pointerEvents = 'auto';
                                            captureBtn.textContent = 'Capture';
                                            statusBadge.textContent = 'Idle';
                                            modeLabel.textContent = '---';
                                            meterMode = 'idle';
                                            levelBar.style.transition = 'width 30ms linear';
                                            resetMeter();
                                            stopLevelPolling();
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
                </script>
            </body>
        </html>
    ''')

if __name__ == "__main__":
    # Threaded=True is required so the web server can serve the UI and the stream simultaneously
    logging.getLogger("werkzeug").addFilter(_SuppressLevelEndpointFilter())
    app.run(host="0.0.0.0", port=5000, threaded=True)