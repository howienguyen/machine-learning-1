"""HTTP and WebSocket inference service for Log-Mel CNN v1.1.

python MelCNN-MGR/inference_web_service/app.py \
  --run-dir MelCNN-MGR/models/logmel-cnn-demo \
  --host 127.0.0.1 \
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
import json
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

from inference_logmel_cnn_v1_1 import (
    AUDIO_BACKEND,
    CLIP_DURATION,
    SAMPLE_RATE,
    LogMelCNNV11Inference,
    PredictionResult,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
ENV_RUN_DIR = "LOGMEL_CNN_V11_RUN_DIR"
STREAM_PATH = "/ws/stream"


class StreamSession:
    def __init__(
        self,
        engine: LogMelCNNV11Inference,
        sample_rate: int,
        channels: int,
        sample_format: str,
        mode: str,
        emit_interval_sec: float,
        stream_id: str | None,
        max_buffer_sec: float = 30.0,
    ) -> None:
        self.engine = engine
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_format = sample_format
        self.mode = mode
        self.emit_interval_sec = float(emit_interval_sec)
        self.stream_id = stream_id or "stream"
        self.max_buffer_samples = int(round(max_buffer_sec * SAMPLE_RATE))
        self.emit_interval_samples = max(1, int(round(self.emit_interval_sec * SAMPLE_RATE)))

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
        if self.sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=self.sample_rate, target_sr=SAMPLE_RATE).astype(np.float32, copy=False)

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
                "buffer_seconds": round(self.buffer.size / SAMPLE_RATE, 3),
                "is_warmup": is_warmup,
            }
        )
        return payload

    def ingest_chunk(self, payload: bytes) -> tuple[dict[str, Any], dict[str, Any] | None]:
        audio = self._decode_chunk(payload)
        self._append_audio(audio)

        status = {
            "event": "chunk_received",
            "stream_id": self.stream_id,
            "received_chunks": self.received_chunks,
            "buffer_seconds": round(self.buffer.size / SAMPLE_RATE, 3),
            "ready": self.buffer.size >= int(round(CLIP_DURATION * SAMPLE_RATE)),
        }

        should_emit = (
            self.total_resampled_samples - self.last_emit_total_samples >= self.emit_interval_samples
        )
        if not should_emit or self.buffer.size == 0:
            return status, None

        result = self.engine.predict_waveform(
            self.buffer,
            sr=SAMPLE_RATE,
            mode=self.mode,
            source_name=f"{self.stream_id}:live",
        )
        self.last_emit_total_samples = self.total_resampled_samples
        is_warmup = self.buffer.size < int(round(CLIP_DURATION * SAMPLE_RATE))
        return status, self._result_payload(result, event="partial_result", is_warmup=is_warmup)

    def finalize(self) -> dict[str, Any]:
        if self.buffer.size == 0:
            raise ValueError("No audio received for this stream.")
        result = self.engine.predict_waveform(
            self.buffer,
            sr=SAMPLE_RATE,
            mode=self.mode,
            source_name=f"{self.stream_id}:final",
        )
        is_warmup = self.buffer.size < int(round(CLIP_DURATION * SAMPLE_RATE))
        return self._result_payload(result, event="final_result", is_warmup=is_warmup)


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


def _resolve_run_dir(run_dir: str | Path | None) -> Path:
    candidate = run_dir or os.environ.get(ENV_RUN_DIR)
    if not candidate:
        raise ValueError(
            f"Run directory is required. Pass --run-dir or set {ENV_RUN_DIR}."
        )
    return Path(candidate).expanduser().resolve()


def create_app(
    run_dir: str | Path | None = None,
    prefer_macro_f1: bool = True,
) -> FastAPI:
    resolved_run_dir = _resolve_run_dir(run_dir)
    engine = LogMelCNNV11Inference(
        resolved_run_dir,
        prefer_macro_f1=prefer_macro_f1,
    )

    app = FastAPI(title="Log-Mel CNN v1.1 Inference Service")
    app.state.engine = engine

    @app.get("/health")
    async def health() -> Any:
        return {
            "status": "ok",
            "service": "logmel-cnn-v1_1-inference",
            "transport": ["http", "websocket"],
            "audio_backend": AUDIO_BACKEND,
            "run_dir": str(engine.run_dir),
            "model_file": engine.model_path.name,
            "n_classes": engine.n_classes,
        }

    @app.get("/model")
    async def model_info() -> Any:
        return {
            "run_dir": str(engine.run_dir),
            "model_file": engine.model_path.name,
            "audio_backend": AUDIO_BACKEND,
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                temp_path = Path(tmp.name)
            result = engine.predict(temp_path, mode=resolved_mode)
            payload = _prediction_to_dict(result)
            payload["request_source"] = "upload"
            payload["original_filename"] = file.filename
            return payload
        except Exception as exc:
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
            result = engine.predict(path, mode=mode)
        except Exception as exc:
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

        results = engine.predict_batch(resolved_paths, mode=mode)
        return {
            "mode": mode,
            "count": len(results),
            "results": [_prediction_to_dict(result) for result in results],
        }

    @app.websocket(STREAM_PATH)
    async def stream_audio(websocket: WebSocket) -> None:
        await websocket.accept()
        session: StreamSession | None = None
        await websocket.send_json(
            {
                "event": "hello",
                "service": "logmel-cnn-v1_1-inference",
                "sample_rate": SAMPLE_RATE,
                "clip_duration": CLIP_DURATION,
                "supported_sample_formats": ["pcm_s16le", "pcm_f32le"],
                "supported_modes": ["single_crop", "three_crop"],
            }
        )

        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if message.get("text") is not None:
                    payload = json.loads(message["text"])
                    event = payload.get("event") or payload.get("type")

                    if event == "start":
                        try:
                            mode = _resolve_mode(payload.get("mode"))
                            session = StreamSession(
                                engine=engine,
                                sample_rate=int(payload.get("sample_rate", SAMPLE_RATE)),
                                channels=int(payload.get("channels", 1)),
                                sample_format=str(payload.get("sample_format") or payload.get("encoding") or "pcm_s16le"),
                                mode=mode,
                                emit_interval_sec=float(payload.get("emit_interval_sec", 1.0)),
                                stream_id=payload.get("stream_id"),
                            )
                        except Exception as exc:
                            await websocket.send_json({"event": "error", "detail": str(exc)})
                            continue

                        await websocket.send_json(
                            {
                                "event": "started",
                                "stream_id": session.stream_id,
                                "sample_rate": session.sample_rate,
                                "channels": session.channels,
                                "sample_format": session.sample_format,
                                "mode": session.mode,
                                "emit_interval_sec": session.emit_interval_sec,
                            }
                        )
                        continue

                    if event == "chunk":
                        if session is None:
                            await websocket.send_json({"event": "error", "detail": "Send a start event before chunk events."})
                            continue
                        encoded = payload.get("audio_base64")
                        if not encoded:
                            await websocket.send_json({"event": "error", "detail": "chunk event requires audio_base64."})
                            continue
                        chunk_bytes = base64.b64decode(encoded)
                        status, partial = session.ingest_chunk(chunk_bytes)
                        await websocket.send_json(status)
                        if partial is not None:
                            await websocket.send_json(partial)
                        continue

                    if event in {"finalize", "stop"}:
                        if session is None:
                            await websocket.send_json({"event": "error", "detail": "No active stream session to finalize."})
                            continue
                        try:
                            await websocket.send_json(session.finalize())
                        except Exception as exc:
                            await websocket.send_json({"event": "error", "detail": str(exc)})
                        session = None
                        continue

                    if event == "ping":
                        await websocket.send_json({"event": "pong"})
                        continue

                    await websocket.send_json({"event": "error", "detail": f"Unsupported event: {event}"})
                    continue

                if message.get("bytes") is not None:
                    if session is None:
                        await websocket.send_json({"event": "error", "detail": "Send a start event before binary audio frames."})
                        continue
                    status, partial = session.ingest_chunk(message["bytes"])
                    await websocket.send_json(status)
                    if partial is not None:
                        await websocket.send_json(partial)
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Log-Mel CNN v1.1 HTTP/WebSocket inference service.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("MelCNN-MGR/models/logmel-cnn-demo"),
        help=f"Training run directory. Can also be provided via {ENV_RUN_DIR}.",
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

    app = create_app(
        run_dir=args.run_dir,
        prefer_macro_f1=not args.final_model,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()