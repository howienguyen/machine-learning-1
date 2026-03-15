"""HTTP and WebSocket inference service for the Log-Mel CNN v2.x family.

WebSocket streaming behavior:
    - waits until at least one full model clip is available before inferring
    - always infers on the latest clip-duration window only
    - can treat each incoming chunk as a full latest-window snapshot when the
        client enables replace_buffer_on_chunk
    - emits the newest partial result for each eligible update instead of sending
        both a status message and a result for the same chunk

python MelCNN-MGR/inference_web_service/app.py \
  --model-dir MelCNN-MGR/models/logmel-cnn-demo \
    --host 0.0.0.0 \
  --port 8000

curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/model

curl -X POST http://127.0.0.1:8000/predict_json \
  -H "Content-Type: application/json" \
  -d '{
    "audio_path": "audio_demo/Mưa Chiều-Metal Rock.mp3",
    "mode": "three_crop"
  }'

"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse


SERVICE_DIR = Path(__file__).resolve().parent
MELCNN_DIR = SERVICE_DIR.parent
if str(MELCNN_DIR) not in sys.path:
    sys.path.insert(0, str(MELCNN_DIR))

from model_inference.inference_logmel_cnn_v2_x import (
    AUDIO_BACKEND,
    LogMelCNNV2XInference,
    PredictionResult,
)


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MODEL_DIR = (MELCNN_DIR / "demo-models" / "logmel-cnn-v2_1-20260313-081401").resolve()
STREAM_PATH = "/ws/stream"
LOGGER = logging.getLogger("melcnn.inference_web_service")
DEFAULT_WS_PING_INTERVAL = 20.0
DEFAULT_WS_PING_TIMEOUT = 20.0


def _ws_payload_summary(payload: dict[str, Any]) -> str:
    summary_keys = (
        "event",
        "type",
        "stream_id",
        "sample_rate",
        "channels",
        "sample_format",
        "encoding",
        "mode",
        "emit_interval_sec",
        "received_chunks",
        "buffer_seconds",
        "ready",
        "genre",
        "confidence",
        "detail",
    )
    parts: list[str] = []
    for key in summary_keys:
        if key in payload and payload.get(key) is not None:
            parts.append(f"{key}={payload.get(key)!r}")
    return ", ".join(parts) if parts else "(no summary fields)"


def _log_ws_inbound_message(client: Any, payload: dict[str, Any]) -> None:
    LOGGER.info("WS recv client=%s %s", client, _ws_payload_summary(payload))


def _log_ws_outbound_message(client: Any, payload: dict[str, Any]) -> None:
    LOGGER.info("WS send client=%s %s", client, _ws_payload_summary(payload))


def _ensure_websocket_backend_available() -> None:
    if importlib.util.find_spec("websockets") is not None:
        return
    if importlib.util.find_spec("wsproto") is not None:
        return
    raise RuntimeError(
        "WebSocket support requires an ASGI WebSocket backend. "
        "Install one of: 'websockets', 'wsproto', or 'uvicorn[standard]'."
    )

class StreamSession:
    def __init__(
        self,
        engine: LogMelCNNV2XInference,
        sample_rate: int,
        channels: int,
        sample_format: str,
        mode: str,
        emit_interval_sec: float,
        stream_id: str | None,
        replace_buffer_on_chunk: bool = False,
        max_buffer_sec: float = 30.0,
    ) -> None:
        # When replace_buffer_on_chunk is true, each client payload is treated
        # as the newest complete audio snapshot for inference rather than as a
        # delta to append to the rolling stream buffer.
        self.engine = engine
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_format = sample_format
        self.mode = mode
        self.emit_interval_sec = float(emit_interval_sec)
        self.stream_id = stream_id or "stream"
        self.replace_buffer_on_chunk = bool(replace_buffer_on_chunk)
        self.target_sample_rate = int(engine.sample_rate)
        self.target_clip_duration = float(engine.clip_duration)
        self.target_clip_samples = int(round(self.target_clip_duration * self.target_sample_rate))
        self.max_buffer_samples = int(round(max_buffer_sec * self.target_sample_rate))
        self.emit_interval_samples = max(1, int(round(self.emit_interval_sec * self.target_sample_rate)))

        self.buffer = np.zeros((0,), dtype=np.float32)
        self.total_resampled_samples = 0
        self.received_chunks = 0
        self.last_emit_total_samples = 0

    def _decode_chunk(self, payload: bytes) -> np.ndarray:
        if self.sample_format == "pcm_s16le":
            audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
        elif self.sample_format == "pcm_f32le":
            audio = np.frombuffer(payload, dtype="<f4").astype(np.float32, copy=False)
        else:
            raise ValueError("Unsupported sample_format. Use pcm_s16le or pcm_f32le.")

        if self.channels > 1:
            usable = (audio.size // self.channels) * self.channels
            if usable == 0:
                return np.zeros((0,), dtype=np.float32)
            audio = audio[:usable].reshape(-1, self.channels).mean(axis=1)
        return audio.astype(np.float32, copy=False)

    def _append_audio(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        if self.sample_rate != self.target_sample_rate:
            audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=self.target_sample_rate).astype(np.float32, copy=False)

        if self.replace_buffer_on_chunk:
            self.buffer = audio[-self.max_buffer_samples :]
        else:
            self.buffer = np.concatenate([self.buffer, audio])
            if self.buffer.size > self.max_buffer_samples:
                self.buffer = self.buffer[-self.max_buffer_samples :]
        self.total_resampled_samples += int(audio.size)
        self.received_chunks += 1

    def _result_payload(self, result: PredictionResult, event: str, is_warmup: bool) -> dict[str, Any]:
        payload = _prediction_to_dict(result)
        payload.update(
            {
                "event": event,
                "stream_id": self.stream_id,
                "received_chunks": self.received_chunks,
                "buffer_seconds": round(self.buffer.size / self.target_sample_rate, 3),
                "is_warmup": is_warmup,
            }
        )
        return payload

    def _latest_inference_window(self) -> np.ndarray:
        # Streaming inference is aligned to the training contract: use one
        # full latest clip-duration window and reject shorter buffers.
        if self.buffer.size < self.target_clip_samples:
            raise ValueError(
                f"Need at least {self.target_clip_duration:.1f}s of audio before inference. "
                f"Current buffered audio: {self.buffer.size / self.target_sample_rate:.3f}s."
            )
        return self.buffer[-self.target_clip_samples :]

    def ingest_chunk(self, payload: bytes) -> tuple[dict[str, Any], dict[str, Any] | None]:
        audio = self._decode_chunk(payload)
        self._append_audio(audio)

        status = {
            "event": "chunk_received",
            "stream_id": self.stream_id,
            "received_chunks": self.received_chunks,
            "buffer_seconds": round(self.buffer.size / self.target_sample_rate, 3),
            "ready": self.buffer.size >= self.target_clip_samples,
        }

        should_emit = (
            self.total_resampled_samples - self.last_emit_total_samples >= self.emit_interval_samples
        )
        if not should_emit or self.buffer.size < self.target_clip_samples:
            return status, None

        LOGGER.info(
            "WS infer partial stream_id=%s chunks=%s buffer_seconds=%.3f mode=%s replace_buffer_on_chunk=%s",
            self.stream_id,
            self.received_chunks,
            self.buffer.size / self.target_sample_rate,
            self.mode,
            self.replace_buffer_on_chunk,
        )
        result = self.engine.predict_waveform(
            self._latest_inference_window(),
            sr=self.target_sample_rate,
            mode=self.mode,
            source_name=f"{self.stream_id}:live",
        )
        self.last_emit_total_samples = self.total_resampled_samples
        LOGGER.info(
            "WS infer partial result stream_id=%s genre=%s confidence=%.4f",
            self.stream_id,
            result.genre,
            result.confidence,
        )
        return status, self._result_payload(result, event="partial_result", is_warmup=False)

    def finalize(self) -> dict[str, Any]:
        if self.buffer.size < self.target_clip_samples:
            raise ValueError(
                f"Need at least {self.target_clip_duration:.1f}s of audio before final inference. "
                f"Current buffered audio: {self.buffer.size / self.target_sample_rate:.3f}s."
            )
        LOGGER.info(
            "WS infer final stream_id=%s chunks=%s buffer_seconds=%.3f mode=%s",
            self.stream_id,
            self.received_chunks,
            self.buffer.size / self.target_sample_rate,
            self.mode,
        )
        result = self.engine.predict_waveform(
            self._latest_inference_window(),
            sr=self.target_sample_rate,
            mode=self.mode,
            source_name=f"{self.stream_id}:final",
        )
        LOGGER.info(
            "WS infer final result stream_id=%s genre=%s confidence=%.4f",
            self.stream_id,
            result.genre,
            result.confidence,
        )
        return self._result_payload(result, event="final_result", is_warmup=False)


def _prediction_to_dict(result: PredictionResult) -> dict[str, Any]:
    return {
        "file": result.file,
        "genre": result.genre,
        "confidence": result.confidence,
        "mode": result.mode,
        "genre_classes": result.genre_classes,
        "top_k": [
            {"genre": genre, "probability": probability}
            for genre, probability in result.top_k(3)
        ],
        "crops": [
            {
                "genre": crop.genre,
                "confidence": crop.confidence,
                "probs": crop.probs,
            }
            for crop in result.crops
        ],
        "probs": result.probs,
    }


def _resolve_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or "three_crop").strip()
    if mode not in {"single_crop", "three_crop"}:
        raise ValueError("mode must be one of: single_crop, three_crop")
    return mode


def _resolve_model_dir(model_dir: str | Path | None) -> Path:
    candidate = model_dir or DEFAULT_MODEL_DIR
    if not candidate:
        raise ValueError("Model directory is required.")
    return Path(candidate).expanduser().resolve()


def create_app(
    model_dir: str | Path | None = None,
    prefer_macro_f1: bool = True,
) -> FastAPI:
    resolved_model_dir = _resolve_model_dir(model_dir)
    LOGGER.info(
        "Initializing inference service model_dir=%s prefer_macro_f1=%s",
        resolved_model_dir,
        prefer_macro_f1,
    )
    engine = LogMelCNNV2XInference(
        resolved_model_dir,
        prefer_macro_f1=prefer_macro_f1,
    )
    LOGGER.info(
        "Loaded inference model model_file=%s sample_rate=%s clip_duration=%.3f logmel_shape=%s classes=%s",
        engine.model_path.name,
        engine.sample_rate,
        engine.clip_duration,
        engine.logmel_shape,
        engine.n_classes,
    )

    app = FastAPI(title="Log-Mel CNN v2.x Family Inference Service")
    app.state.engine = engine

    @app.get("/health")
    async def health() -> Any:
        return {
            "status": "ok",
            "service": "logmel-cnn-v2_x-family-inference",
            "transport": ["http", "websocket"],
            "audio_backend": AUDIO_BACKEND,
            "model_dir": str(engine.model_dir),
            "model_file": engine.model_path.name,
            "sample_rate": engine.sample_rate,
            "clip_duration": engine.clip_duration,
            "logmel_shape": list(engine.logmel_shape),
            "n_classes": engine.n_classes,
        }

    @app.get("/model")
    async def model_info() -> Any:
        return {
            "model_dir": str(engine.model_dir),
            "model_file": engine.model_path.name,
            "audio_backend": AUDIO_BACKEND,
            "sample_rate": engine.sample_rate,
            "clip_duration": engine.clip_duration,
            "logmel_shape": list(engine.logmel_shape),
            "genre_classes": engine.genre_classes,
            "n_classes": engine.n_classes,
            "run_report": engine.run_report,
        }

    @app.post("/predict")
    async def predict(
        audio_path: str | None = Form(default=None),
        mode_form: str | None = Form(default=None),
        file: UploadFile | None = File(default=None),
    ) -> Any:
        mode = mode_form
        if file is None:
            raise HTTPException(status_code=400, detail="Use JSON with /predict_json or multipart upload with /predict.")

        try:
            resolved_mode = _resolve_mode(mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        suffix = Path(file.filename or "upload.audio").suffix or ".audio"
        temp_path: Path | None = None
        try:
            LOGGER.info(
                "HTTP /predict request original_filename=%s mode=%s content_type=%s",
                file.filename,
                resolved_mode,
                getattr(file, "content_type", None),
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_path = Path(tmp.name)
            result = engine.predict(temp_path, mode=resolved_mode)
            LOGGER.info(
                "HTTP /predict result original_filename=%s genre=%s confidence=%.4f",
                file.filename,
                result.genre,
                result.confidence,
            )
            payload = _prediction_to_dict(result)
            payload["request_source"] = "upload"
            payload["original_filename"] = file.filename
            return payload
        except Exception as exc:
            LOGGER.exception("HTTP /predict failed original_filename=%s", file.filename)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    @app.post("/predict_json")
    async def predict_json(body: dict[str, Any]) -> Any:
        audio_path = body.get("audio_path")
        if not audio_path:
            raise HTTPException(status_code=400, detail="Provide JSON body with 'audio_path'.")

        try:
            mode = _resolve_mode(body.get("mode"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        path = Path(audio_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail=f"Audio path not found: {path}")

        try:
            LOGGER.info("HTTP /predict_json request audio_path=%s mode=%s", path, mode)
            result = engine.predict(path, mode=mode)
            LOGGER.info(
                "HTTP /predict_json result audio_path=%s genre=%s confidence=%.4f",
                path,
                result.genre,
                result.confidence,
            )
        except Exception as exc:
            LOGGER.exception("HTTP /predict_json failed audio_path=%s", path)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = _prediction_to_dict(result)
        payload["request_source"] = "audio_path"
        return payload

    @app.post("/predict_batch")
    async def predict_batch(body: dict[str, Any]) -> Any:
        audio_paths = body.get("audio_paths")
        if not isinstance(audio_paths, list) or not audio_paths:
            raise HTTPException(status_code=400, detail="Provide JSON body with non-empty 'audio_paths' list.")

        try:
            mode = _resolve_mode(body.get("mode"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        missing_paths = []
        resolved_paths: list[Path] = []
        for raw_path in audio_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists() or not path.is_file():
                missing_paths.append(str(path))
            else:
                resolved_paths.append(path)

        if missing_paths:
            return JSONResponse(
                status_code=404,
                content={"error": "Some audio paths were not found.", "missing_paths": missing_paths},
            )

        LOGGER.info("HTTP /predict_batch request count=%s mode=%s", len(resolved_paths), mode)
        results = engine.predict_batch(resolved_paths, mode=mode)
        LOGGER.info("HTTP /predict_batch result count=%s", len(results))
        return {
            "mode": mode,
            "count": len(results),
            "results": [_prediction_to_dict(result) for result in results],
        }

    @app.websocket(STREAM_PATH)
    async def stream_audio(websocket: WebSocket) -> None:
        await websocket.accept()
        session: StreamSession | None = None
        client = getattr(websocket, "client", None)
        LOGGER.info("WS accepted client=%s", client)
        hello_payload = {
            "event": "hello",
            "service": "logmel-cnn-v2_x-family-inference",
            "sample_rate": engine.sample_rate,
            "clip_duration": engine.clip_duration,
            "supported_sample_formats": ["pcm_s16le", "pcm_f32le"],
            "supported_modes": ["single_crop", "three_crop"],
        }
        _log_ws_outbound_message(client, hello_payload)
        await websocket.send_json(hello_payload)

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    _log_ws_inbound_message(client, payload)
                    event = payload.get("event") or payload.get("type")

                    if event == "start":
                        try:
                            mode = _resolve_mode(payload.get("mode"))
                            session = StreamSession(
                                engine=engine,
                                sample_rate=int(payload.get("sample_rate", engine.sample_rate)),
                                channels=int(payload.get("channels", 1)),
                                sample_format=str(payload.get("sample_format") or payload.get("encoding") or "pcm_s16le"),
                                mode=mode,
                                emit_interval_sec=float(payload.get("emit_interval_sec", 1.0)),
                                stream_id=payload.get("stream_id"),
                                replace_buffer_on_chunk=bool(payload.get("replace_buffer_on_chunk", False)),
                            )
                            LOGGER.info(
                                "WS start stream_id=%s client=%s sample_rate=%s channels=%s sample_format=%s mode=%s emit_interval_sec=%.3f replace_buffer_on_chunk=%s",
                                session.stream_id,
                                client,
                                session.sample_rate,
                                session.channels,
                                session.sample_format,
                                session.mode,
                                session.emit_interval_sec,
                                session.replace_buffer_on_chunk,
                            )
                        except Exception as exc:
                            LOGGER.exception("WS start failed client=%s payload=%s", client, payload)
                            error_payload = {"event": "error", "detail": str(exc)}
                            _log_ws_outbound_message(client, error_payload)
                            await websocket.send_json(error_payload)
                            continue

                        started_payload = {
                            "event": "started",
                            "stream_id": session.stream_id,
                            "sample_rate": session.sample_rate,
                            "channels": session.channels,
                            "sample_format": session.sample_format,
                            "mode": session.mode,
                            "emit_interval_sec": session.emit_interval_sec,
                            "replace_buffer_on_chunk": session.replace_buffer_on_chunk,
                        }
                        _log_ws_outbound_message(client, started_payload)
                        await websocket.send_json(started_payload)
                        continue

                    if event == "chunk":
                        if session is None:
                            error_payload = {"event": "error", "detail": "Send a start event before chunk events."}
                            _log_ws_outbound_message(client, error_payload)
                            await websocket.send_json(error_payload)
                            continue
                        encoded = payload.get("audio_base64")
                        if not encoded:
                            error_payload = {"event": "error", "detail": "chunk event requires audio_base64."}
                            _log_ws_outbound_message(client, error_payload)
                            await websocket.send_json(error_payload)
                            continue
                        chunk_bytes = base64.b64decode(encoded)
                        LOGGER.debug(
                            "WS chunk(json) stream_id=%s bytes=%s",
                            session.stream_id,
                            len(chunk_bytes),
                        )
                        status, partial = session.ingest_chunk(chunk_bytes)
                        if partial is not None:
                            # When a fresh inference exists, send only the
                            # newest result for this update path.
                            _log_ws_outbound_message(client, partial)
                            await websocket.send_json(partial)
                        else:
                            _log_ws_outbound_message(client, status)
                            await websocket.send_json(status)
                        continue

                    if event in {"finalize", "stop"}:
                        if session is None:
                            error_payload = {"event": "error", "detail": "No active stream session to finalize."}
                            _log_ws_outbound_message(client, error_payload)
                            await websocket.send_json(error_payload)
                            continue
                        try:
                            LOGGER.info("WS finalize stream_id=%s client=%s", session.stream_id, client)
                            final_payload = session.finalize()
                            _log_ws_outbound_message(client, final_payload)
                            await websocket.send_json(final_payload)
                        except Exception as exc:
                            LOGGER.exception("WS finalize failed stream_id=%s", session.stream_id)
                            error_payload = {"event": "error", "detail": str(exc)}
                            _log_ws_outbound_message(client, error_payload)
                            await websocket.send_json(error_payload)
                        session = None
                        continue

                    if event == "ping":
                        pong_payload = {"event": "pong"}
                        _log_ws_outbound_message(client, pong_payload)
                        await websocket.send_json(pong_payload)
                        continue

                    error_payload = {"event": "error", "detail": f"Unsupported event: {event}"}
                    _log_ws_outbound_message(client, error_payload)
                    await websocket.send_json(error_payload)
                    continue

                if message.get("bytes") is not None:
                    if session is None:
                        error_payload = {"event": "error", "detail": "Send a start event before binary audio frames."}
                        _log_ws_outbound_message(client, error_payload)
                        await websocket.send_json(error_payload)
                        continue
                    LOGGER.debug(
                        "WS chunk(binary) stream_id=%s bytes=%s",
                        session.stream_id,
                        len(message["bytes"]),
                    )
                    status, partial = session.ingest_chunk(message["bytes"])
                    if partial is not None:
                        # When a fresh inference exists, send only the newest
                        # result for this update path.
                        _log_ws_outbound_message(client, partial)
                        await websocket.send_json(partial)
                    else:
                        _log_ws_outbound_message(client, status)
                        await websocket.send_json(status)
        except WebSocketDisconnect:
            LOGGER.info("WS disconnected client=%s stream_id=%s", client, getattr(session, "stream_id", None))
            return

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Log-Mel CNN v2.x-family HTTP/WebSocket inference service.",
    )
    parser.add_argument(
        "--model-dir",
        "--run-dir",
        dest="model_dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Model directory. The --run-dir alias is deprecated.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host interface to bind.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port to bind.",
    )
    parser.add_argument(
        "--final-model",
        action="store_true",
        help="Prefer the final saved model over the best Macro-F1 checkpoint.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    _ensure_websocket_backend_available()

    app = create_app(
        model_dir=args.model_dir,
        prefer_macro_f1=not args.final_model,
    )
    LOGGER.info(
        "Starting uvicorn host=%s port=%s ws_ping_interval=%.1f ws_ping_timeout=%.1f",
        args.host,
        args.port,
        DEFAULT_WS_PING_INTERVAL,
        DEFAULT_WS_PING_TIMEOUT,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ws_ping_interval=DEFAULT_WS_PING_INTERVAL,
        ws_ping_timeout=DEFAULT_WS_PING_TIMEOUT,
    )


if __name__ == "__main__":
    main()