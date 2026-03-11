# Log-Mel CNN v1.1 Inference Web Service

This service exposes `MelCNN-MGR/inference_logmel_cnn_v1_1.py` over both HTTP and WebSocket transports.

## Endpoints

1. `GET /health`
2. `GET /model`
3. `POST /predict` for multipart upload inference
4. `POST /predict_json` for local-path JSON inference
5. `POST /predict_batch`
6. `WS /ws/stream`

## Run

```bash
python MelCNN-MGR/inference_web_service/app.py \
  --run-dir MelCNN-MGR/models/logmel-cnn-v1_1-20260311-025046 \
  --host 127.0.0.1 \
  --port 8000
```

You can also set the run directory via `LOGMEL_CNN_V11_RUN_DIR`.

## Predict by local path

```bash
curl -X POST http://127.0.0.1:8000/predict_json \
  -H 'Content-Type: application/json' \
  -d '{
    "audio_path": "audio_demo/Metal-Black Sabbath-Black Sabbath.mp3",
    "mode": "three_crop"
  }'
```

## Predict by upload

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F mode=three_crop \
  -F file=@audio_demo/Blues-Chris\ Stapleton-Tennessee\ Whiskey.mp3
```

## Batch predict by local path

```bash
curl -X POST http://127.0.0.1:8000/predict_batch \
  -H 'Content-Type: application/json' \
  -d '{
    "mode": "single_crop",
    "audio_paths": [
      "audio_demo/Blues-Chris Stapleton-Tennessee Whiskey.mp3",
      "audio_demo/Metal-Black Sabbath-Black Sabbath.mp3"
    ]
  }'
```

## Notes

1. The model is loaded once at service startup and reused across HTTP and WebSocket requests.
2. `POST /predict` accepts multipart uploads only.
3. `POST /predict_json` and `POST /predict_batch` use trusted local file paths.

## WebSocket Streaming

The WebSocket endpoint is intended for near-real-time chunked PCM streaming from a Python web app or browser client.

Endpoint:

```text
ws://127.0.0.1:8000/ws/stream
```

Protocol overview:

1. Connect to the socket.
2. Send a JSON `start` event describing the stream format.
3. Send audio chunks either as binary WebSocket frames or as JSON `chunk` events with base64 payloads.
4. Receive `chunk_received` acknowledgements and `partial_result` messages while audio is still arriving.
5. Send a JSON `finalize` event to receive the final prediction.

Compatibility aliases accepted by the service:

1. `type` may be used instead of `event`
2. `encoding` may be used instead of `sample_format`
3. `stop` may be used instead of `finalize`

### Start message

```json
{
  "event": "start",
  "stream_id": "demo-001",
  "sample_rate": 22050,
  "channels": 1,
  "sample_format": "pcm_s16le",
  "mode": "three_crop",
  "emit_interval_sec": 1.0
}
```

Supported `sample_format` values:

1. `pcm_s16le`
2. `pcm_f32le`

### Chunk message as JSON

```json
{
  "event": "chunk",
  "audio_base64": "<base64-encoded PCM bytes>"
}
```

### Finalize message

```json
{
  "event": "finalize"
}
```

### Typical server messages

1. `hello`
2. `started`
3. `chunk_received`
4. `partial_result`
5. `final_result`
6. `error`

`partial_result` messages begin as soon as enough streamed audio has accumulated according to `emit_interval_sec`. Early outputs may be marked with `is_warmup: true` when the buffered audio is still shorter than the model's nominal 10-second clip length.

Client-side send frequency is independent from `emit_interval_sec`.

For example, a capture client may buffer local PCM and send one larger binary frame every 3 seconds while still asking the server to emit partial inference results on its own schedule once enough audio has accumulated.

## Restart Tolerance

The current Python capture client in `MelCNN-MGR/demo-app/web_audio_capture_v1.py` implements a small reconnect/backoff policy for service restarts during capture.

Behavior:

1. if a WebSocket send or receive fails, the client marks the transport as lost
2. it retries with a short bounded backoff sequence
3. after reconnect, it starts a fresh stream session
4. it replays a recent rolling window of buffered PCM chunks so the server can recover context and continue emitting useful partial results

This keeps the client usable during short service restarts without requiring the whole capture session to be stopped manually.

## Capture Client Batching

The Python capture client `MelCNN-MGR/demo-app/web_audio_capture_v1.py` now supports a configurable local send interval through:

1. `MELCNN_INFERENCE_SEND_INTERVAL_SEC`

Example:

```bash
MELCNN_INFERENCE_SEND_INTERVAL_SEC=3 python MelCNN-MGR/demo-app/web_audio_capture_v1.py
```

With that setting, the client still captures continuously, but it batches raw PCM16 chunks locally and flushes them to the websocket roughly every 3 seconds instead of sending every capture callback.