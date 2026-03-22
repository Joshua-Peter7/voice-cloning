"""
api/app.py
──────────
FastAPI application — WebSocket voice endpoint + REST endpoints.

WebSocket protocol:
  Client → Server messages:
    {"type": "audio_chunk",  "audio": "<base64 float32 PCM>", "language": "ta"}
    {"type": "text_query",   "text": "...", "language": "ta"}
    {"type": "reset"}
    {"type": "update_voice", "path": "/path/to/ref.wav"}

  Server → Client messages:
    {"type": "status",        "msg": "asr_start"}
    {"type": "asr_result",    "text": "...", "language": "ta", "confidence": 0.99}
    {"type": "audio_chunk",   "audio": "<base64 WAV>", "chunk_index": 0}
    {"type": "ttfb",          "ms": 432.1}
    {"type": "turn_complete", "session_id": "..."}
    {"type": "error",         "stage": "asr", "msg": "..."}
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import Response, JSONResponse

from core.pipeline import PipelineOrchestrator, build_pipeline
from core.logger import get_logger, setup_logger
from config.settings import Settings, get_settings

log = get_logger("api.app")

# ── App lifecycle ─────────────────────────────────────────────────────────────

_pipeline: Optional[PipelineOrchestrator] = None
_settings: Optional[Settings] = None
_active_sessions: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _settings
    _settings = get_settings()
    setup_logger(
        level=_settings.app.log_level,
        env=_settings.app.env,
        log_dir=_settings.infra.log_dir,
    )
    log.info("Starting Indic Voice Cloner API", version=_settings.app.version)
    _pipeline = build_pipeline(_settings)
    log.info("Pipeline ready")
    yield
    log.info("Shutting down Indic Voice Cloner API")


class TTSRequest(BaseModel):
    text: str
    language: str = "ta"
    session_id: Optional[str] = None
    voice_description: Optional[str] = None


class CloneVoiceRequest(BaseModel):
    reference_wav_b64: str   # base64-encoded WAV file
    text: str
    language: str = "ta"


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create the FastAPI application.
    Pass settings to override defaults (useful in tests).
    """
    cfg = settings or get_settings()

    app = FastAPI(
        title="Indic Voice Cloner API",
        description="Enterprise-grade voice cloning for South Indian languages",
        version=cfg.app.version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health / Readiness ────────────────────────────────────────────

    @app.get("/health", tags=["ops"])
    async def health():
        return {
            "status": "ok",
            "service": "indic-voice-cloner",
            "version": cfg.app.version,
        }

    @app.get("/ready", tags=["ops"])
    async def ready():
        if _pipeline is None:
            raise HTTPException(503, "Pipeline not initialized")
        return {"status": "ready"}

    # ── Metrics ───────────────────────────────────────────────────────

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        if _pipeline and hasattr(_pipeline.monitor, "get_metrics_text"):
            return Response(
                _pipeline.monitor.get_metrics_text(),
                media_type=_pipeline.monitor.content_type,
            )
        return Response("# no metrics", media_type="text/plain")

    # ── REST: Text-to-Speech ──────────────────────────────────────────

    @app.post("/tts", tags=["voice"])
    async def synthesize_text(req: TTSRequest):
        """
        Synthesize speech from text (non-streaming).
        Returns base64-encoded WAV audio.
        """
        if _pipeline is None:
            raise HTTPException(503, "Pipeline not ready")

        session_id = req.session_id or str(uuid.uuid4())
        audio_parts = []

        async for chunk in _pipeline.process_text(
            user_text=req.text,
            session_id=session_id,
            language=req.language,
            voice_description=req.voice_description,
        ):
            audio_parts.append(chunk.audio)

        if not audio_parts:
            raise HTTPException(500, "No audio generated")

        combined = np.concatenate(audio_parts)
        buf = io.BytesIO()
        sf.write(buf, combined, samplerate=_pipeline.settings.tts.sample_rate, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()

        return {"audio_base64": audio_b64, "session_id": session_id}

    # ── REST: Clone Voice ─────────────────────────────────────────────

    @app.post("/clone", tags=["voice"])
    async def clone_voice(req: CloneVoiceRequest):
        """
        Clone a voice and synthesize text.
        Upload a 6–30s reference WAV as base64.
        """
        import tempfile
        import os

        if _pipeline is None:
            raise HTTPException(503, "Pipeline not ready")

        # Save reference audio to a temp file
        ref_bytes = base64.b64decode(req.reference_wav_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            tmp_path = tmp.name

        try:
            _pipeline.update_reference_voice(tmp_path)
            session_id = str(uuid.uuid4())
            audio_parts = []

            async for chunk in _pipeline.process_text(
                user_text=req.text,
                session_id=session_id,
                language=req.language,
            ):
                audio_parts.append(chunk.audio)

            combined = np.concatenate(audio_parts)
            buf = io.BytesIO()
            sf.write(buf, combined, samplerate=_pipeline.settings.tts.sample_rate, format="WAV")
            return {"audio_base64": base64.b64encode(buf.getvalue()).decode()}
        finally:
            os.unlink(tmp_path)

    # ── WebSocket: Full-Duplex Voice ──────────────────────────────────

    @app.websocket("/ws/voice")
    async def voice_ws(ws: WebSocket):
        """
        Full-duplex voice endpoint.

        Accepts audio chunks (base64 float32 PCM @ 16kHz) or text queries.
        Streams back audio chunks as base64 WAV.
        """
        global _active_sessions

        if _pipeline is None:
            await ws.close(code=1013, reason="Server not ready")
            return

        await ws.accept()
        session_id = str(uuid.uuid4())
        _active_sessions += 1
        _pipeline.monitor.set_active_sessions(_active_sessions)
        _pipeline.monitor.increment_error  # touch to ensure counter exists

        log.info("WebSocket connected", session_id=session_id)

        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")

                # ── audio_chunk ───────────────────────────────────────
                if msg_type == "audio_chunk":
                    t_request = time.perf_counter()
                    language = data.get("language", "auto")

                    # Decode base64 float32 PCM
                    audio_bytes = base64.b64decode(data["audio"])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

                    await ws.send_json({"type": "status", "msg": "asr_start"})

                    first_audio = True
                    async for chunk in _pipeline.process_audio(
                        audio_array, session_id, language
                    ):
                        if first_audio:
                            ttfb = round((time.perf_counter() - t_request) * 1000, 1)
                            await ws.send_json({"type": "ttfb", "ms": ttfb})
                            first_audio = False

                        buf = io.BytesIO()
                        sf.write(
                            buf, chunk.audio,
                            samplerate=chunk.sample_rate,
                            format="WAV",
                        )
                        await ws.send_json({
                            "type": "audio_chunk",
                            "audio": base64.b64encode(buf.getvalue()).decode(),
                            "chunk_index": chunk.chunk_index,
                            "text_segment": chunk.text_segment,
                        })

                    await ws.send_json({
                        "type": "turn_complete",
                        "session_id": session_id,
                    })

                # ── text_query ────────────────────────────────────────
                elif msg_type == "text_query":
                    language = data.get("language", "ta")
                    text = data.get("text", "")

                    async for chunk in _pipeline.process_text(
                        text, session_id, language,
                        voice_description=data.get("voice_description"),
                    ):
                        buf = io.BytesIO()
                        sf.write(
                            buf, chunk.audio,
                            samplerate=chunk.sample_rate,
                            format="WAV",
                        )
                        await ws.send_json({
                            "type": "audio_chunk",
                            "audio": base64.b64encode(buf.getvalue()).decode(),
                            "chunk_index": chunk.chunk_index,
                        })

                    await ws.send_json({
                        "type": "turn_complete",
                        "session_id": session_id,
                    })

                # ── reset ─────────────────────────────────────────────
                elif msg_type == "reset":
                    _pipeline.reset_session(session_id)
                    await ws.send_json({"type": "status", "msg": "session_reset"})

                # ── update_voice ──────────────────────────────────────
                elif msg_type == "update_voice":
                    wav_path = data.get("path", "")
                    try:
                        _pipeline.update_reference_voice(wav_path)
                        await ws.send_json({"type": "status", "msg": "voice_updated"})
                    except FileNotFoundError as exc:
                        await ws.send_json({"type": "error", "stage": "voice", "msg": str(exc)})

                else:
                    await ws.send_json({
                        "type": "error",
                        "stage": "protocol",
                        "msg": f"Unknown message type: {msg_type}",
                    })

        except WebSocketDisconnect:
            log.info("WebSocket disconnected", session_id=session_id)
        except Exception as exc:
            _pipeline.monitor.increment_error("orchestrator")
            log.error("WebSocket error", error=str(exc), session_id=session_id)
            try:
                await ws.send_json({"type": "error", "msg": str(exc)})
            except Exception:
                pass
        finally:
            _active_sessions -= 1
            _pipeline.monitor.set_active_sessions(_active_sessions)

    return app


# ── Entrypoint ─────────────────────────────────────────────────────────────────

app = create_app()
