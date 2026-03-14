# Dev Log — 2026-03-11 — `MelCNN-MGR/demo-app/web_audio_capture_v1.py` WebSocket Streaming Client

## Scope

This change updates `MelCNN-MGR/demo-app/web_audio_capture_v1.py` so it no longer acts only as a
local loopback recorder with a delayed WAV snapshot. It now also acts as a Python
WebSocket client for the Log-Mel CNN inference web service.

Updated artifacts:

1. `MelCNN-MGR/demo-app/web_audio_capture_v1.py`
2. `MelCNN-MGR/inference_web_service/app.py`
3. `requirements.txt`

## Summary

The capture app now streams raw PCM16 little-endian chunks directly from the WASAPI
loopback capture thread into the inference service over WebSocket binary frames.

The transport model is:

1. one JSON control message at session start
2. raw PCM binary frames for chunk payloads
3. one JSON stop/finalize control message at session end

The Flask UI was also extended so it can surface live websocket and inference status.

## Main Changes

### 1. Added Python WebSocket client integration

`MelCNN-MGR/demo-app/web_audio_capture_v1.py` now includes a `_StreamingInferenceClient` that:

1. connects to `ws://127.0.0.1:8000/ws/stream` by default
2. sends a `start` control message with encoding, sample rate, channels, and frames per buffer
3. sends each captured audio chunk as a raw binary WebSocket frame
4. sends a stop/finalize message when capture ends
5. receives partial and final inference results asynchronously on a background receive thread

### 2. Added shared inference state for the Flask UI

The capture app now keeps a thread-safe `_InferenceState` with:

1. websocket connection state
2. latest partial result
3. latest final result
4. buffer duration and chunk counts
5. latest server event or error

New endpoint:

1. `/inference/status`

This lets the browser UI poll the current streaming inference status while capture is active.

### 3. Preferred 22050 Hz capture rate

The loopback stream negotiation now prefers `22050` first, then falls back to device
defaults and other common rates.

That keeps the client aligned with the inference model's default sample rate while still
allowing fallback when a device cannot open at 22050 Hz.

### 4. Updated the capture UI

The page now shows:

1. websocket connection state
2. stream format summary
3. received chunk count
4. buffered seconds
5. latest partial prediction
6. latest final prediction
7. latest service/error status

The old last-18-seconds playback behavior was preserved, so the app still doubles as a
local capture-and-review tool.

### 5. Added protocol aliases on the inference service

The websocket service now also accepts:

1. `type` as an alias for `event`
2. `encoding` as an alias for `sample_format`
3. `stop` as an alias for `finalize`

This keeps the service compatible with the cleaner streaming control shape used by the
capture client.

### 6. Made the capture module import-safe outside Windows capture environments

`pyaudiowpatch` is Windows/WASAPI-specific. The module now imports cleanly even when
that package is unavailable, and only raises a clear error when capture is actually started.

### 7. Added reconnect/backoff and replay after service restarts

The streaming client now tolerates short inference-service restarts during capture.

The reconnect behavior now prioritizes keeping capture alive even when the inference service is down:
1. detect transport loss on either `send_binary(...)`, receive-loop failure, or initial websocket connection failure
2. keep local capture and playback running normally instead of blocking the capture thread
3. retry websocket connection in the background every 10 seconds
4. stop trying after 6 failed retries in the same capture session
5. if a reconnect succeeds before that limit, restart the websocket session with a new `start` control message
6. replay a recent rolling PCM window (up to 12 seconds) from the local capture buffer

That replay step matters because the service maintains stream-local audio state. If the
service restarts, simply reconnecting without replay would leave it with no acoustic
context until enough new live audio arrives.

If all 6 retries fail, inference is disabled for the rest of the current capture session,
but the app still continues functioning as a local capture-and-review tool.

The UI now also provides a manual retry button so the user can explicitly re-arm inference
attempts during an active capture session without restarting capture itself.

The local Flask UI now also exposes reconnect state, reconnect count, retry limits,
disabled-inference state, and the latest backoff/detail message through `/inference/status`.

### 8. Added configurable client-side send frequency

The capture client no longer has to forward every raw capture callback directly to the
inference websocket.

New setting:
- `MELCNN_INFERENCE_SEND_INTERVAL_SEC` (default: 3 seconds)

Behavior:
1. PCM chunks are still captured continuously from WASAPI loopback (e.g., every 4096 frames)
2. chunks are accumulated locally in a pending buffer inside the capture thread
3. the client flushes one larger binary websocket payload whenever the configured send interval (3s) elapses
4. any remaining buffered PCM is flushed before the final stop/finalize control message

This makes it possible to trade off latency against network and protocol churn. For example,
setting the interval to `3` sends roughly one payload every 3 seconds instead of one payload
per capture callback (which could be dozens of times per second).

This send interval is separate from the inference service's own `emit_interval_sec`, which
controls how often the server produces partial inference outputs once audio has arrived.

### 9. Relocated Demo App and Updated Workspace Paths

To keep the project root clean, the capture app was moved:
- From: `web_audio_capture_v1.py` (root)
- To: `MelCNN-MGR/demo-app/web_audio_capture_v1.py`

The script was updated to dynamically resolve the `WORKSPACE_ROOT` so it can still import modules like `MelCNN-MGR.inference_logmel_cnn_v1_1`.

## Codebase Alignment and Documentation

We also performed a cleanup and alignment pass:

1. **`MelCNN-MGR/model_inference/inference_logmel_cnn_v2_x.py`**: Acts as the current config-driven inference entry point used by the web service.
2. **Settings Header**: Updated the settings block in the demo app with clear descriptions for `RECONNECT_MAX_ATTEMPTS`, `RECONNECT_BACKOFF_FACTOR`, `REPLAY_BUFFER_SEC`, and `SEND_INTERVAL_SEC`.
3. **Execution Guides**: Added a "How to run" comment block at the top of the demo app to guide users on environment variables and dependencies.
4. **`MelCNN-MGR/README-DEMO.md`**: Created/Updated this central manifest to document the full production-like pipeline and all command-line entry points.

## Validation Status

Validated in this session:

1. static error checks passed for `MelCNN-MGR/demo-app/web_audio_capture_v1.py`
2. static error checks passed for `MelCNN-MGR/inference_web_service/app.py`
3. `MelCNN-MGR/demo-app/web_audio_capture_v1.py` imports successfully in the current environment after making the PyAudio dependency lazy
4. reconnect/backoff code paths passed static validation after the client update
5. configurable send-interval batching passed static validation after the client update
6. capped retry / inference-disable flow passed static validation after the follow-up reconnect policy update
7. manual retry button / retry endpoint flow passed static validation after the follow-up UI update

Not yet done:

1. live end-to-end streaming verification on a Windows host with WASAPI loopback available
2. browser/UI validation against a running inference service instance