# Dev Log — 2026-03-11 — Inference Web Service WebSocket Streaming

## Scope

This change extends the Log-Mel CNN v1.1 inference web service so it can support
both ordinary HTTP inference and near-real-time chunked audio streaming over WebSockets.

Updated artifacts:

1. `MelCNN-MGR/inference_web_service/app.py`
2. `MelCNN-MGR/inference_web_service/README.md`
3. `MelCNN-MGR/inference_logmel_cnn_v1_1.py`
4. `requirements.txt`

## Summary

The service was originally built as a small Flask REST wrapper around
`inference_logmel_cnn_v1_1.py`.

That was sufficient for one-shot request/response inference, but it was not an
appropriate base for persistent bidirectional audio streaming. The service has now
been moved to FastAPI so HTTP endpoints and a WebSocket stream endpoint can live in
the same process.

## Main Changes

### 1. Service stack moved from Flask to FastAPI

`MelCNN-MGR/inference_web_service/app.py` now uses:

1. `FastAPI`
2. `uvicorn`
3. FastAPI's built-in WebSocket support

This keeps the existing REST functionality while enabling long-lived streaming sessions.

### 2. Existing HTTP inference behavior was preserved

The service still supports:

1. health checks
2. model metadata inspection
3. upload-based prediction
4. JSON path-based prediction
5. batch prediction from local file paths

The path-based HTTP endpoint was split into `POST /predict_json` so multipart upload
and JSON request handling stay unambiguous under FastAPI.

### 3. Added WebSocket streaming endpoint

New endpoint:

1. `WS /ws/stream`

Protocol design:

1. client sends a `start` JSON event with stream metadata
2. client sends audio chunks as either binary frames or base64-encoded JSON chunk events
3. server acknowledges chunks with `chunk_received`
4. server emits `partial_result` messages while audio is still arriving
5. client sends `finalize`
6. server returns `final_result`

### 4. Added streaming session state and PCM decoding

The service now includes a `StreamSession` helper that manages:

1. chunk decoding from `pcm_s16le` or `pcm_f32le`
2. channel downmixing to mono
3. optional resampling to the model sample rate
4. rolling audio buffer management
5. timed partial-result emission

### 5. Added waveform-level inference to the core inference module

`MelCNN-MGR/inference_logmel_cnn_v1_1.py` now exposes:

1. `predict_waveform(...)`

This allows the service to run inference directly on streamed PCM buffers without
writing temporary audio files.

### 6. Added protocol aliases for simpler streaming clients

The websocket layer now also accepts:

1. `type` as an alias for `event`
2. `encoding` as an alias for `sample_format`
3. `stop` as an alias for `finalize`

This keeps the service compatible with simple chunk-streaming clients that send one
small JSON control message up front and then switch to raw binary PCM frames.

## Why This Matters

This gives the project a more realistic transport model for interactive or live-ish
applications:

1. repeated POST requests are no longer required for every short audio chunk
2. the backend can return partial predictions before a stream is finalized
3. the same loaded model instance can serve both one-shot HTTP requests and stream sessions
4. the transport is now closer to how a browser or Python web app would integrate with a live microphone or rolling audio feed
5. the service is easier to pair with reconnecting clients because the control-message shape is more forgiving

## Dependency Changes

Added:

1. `fastapi`
2. `uvicorn`
3. `python-multipart`

These were needed for FastAPI routing, ASGI serving, file uploads, and WebSocket support.

## Validation Status

Validated in this session:

1. static file error checks passed for the updated service and inference module
2. the FastAPI module imports successfully in the configured Python environment

Not yet done:

1. live end-to-end testing of the WebSocket protocol against a real streaming client
2. throughput or latency measurement under sustained chunk traffic