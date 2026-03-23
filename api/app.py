"""
api/app.py
----------
FastAPI application with REST and WebSocket endpoints for voice cloning.

WebSocket protocol:
  Client -> Server:
    {"type": "audio_chunk", "audio": "<base64 float32 PCM>", "language": "ta"}
    {"type": "text_query", "text": "...", "language": "ta", "skip_slm": true}
    {"type": "reset"}
    {"type": "update_voice", "path": "/path/to/ref.wav"}
    {"type": "update_voice_b64", "reference_wav_b64": "<base64 wav>"}
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import RedirectResponse, Response

from config.settings import Settings, get_settings
from core.logger import get_logger, setup_logger
from core.pipeline import PipelineOrchestrator, build_pipeline

log = get_logger("api.app")

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
    skip_slm: bool = True
    emotion: Optional[str] = "neutral"
    voice: Optional[str] = None
    reference_wav_b64: Optional[str] = None
    reference_filename: Optional[str] = None
    reference_content_type: Optional[str] = None


class CloneVoiceRequest(BaseModel):
    reference_wav_b64: str
    reference_filename: Optional[str] = None
    reference_content_type: Optional[str] = None
    text: str
    language: str = "ta"
    skip_slm: bool = True
    emotion: Optional[str] = "neutral"
    voice: Optional[str] = None


def _guess_audio_suffix(
    reference_filename: Optional[str] = None,
    reference_content_type: Optional[str] = None,
) -> str:
    if reference_filename:
        _, ext = os.path.splitext(reference_filename)
        if ext:
            return ext
    if reference_content_type:
        if "mpeg" in reference_content_type or "mp3" in reference_content_type:
            return ".mp3"
        if "wav" in reference_content_type or "wave" in reference_content_type:
            return ".wav"
        if "ogg" in reference_content_type:
            return ".ogg"
        if "flac" in reference_content_type:
            return ".flac"
        if "m4a" in reference_content_type or "mp4" in reference_content_type:
            return ".m4a"
    return ".wav"


def _decode_b64_wav_to_temp(
    reference_wav_b64: str,
    reference_filename: Optional[str] = None,
    reference_content_type: Optional[str] = None,
) -> str:
    ref_bytes = base64.b64decode(reference_wav_b64)
    with tempfile.NamedTemporaryFile(
        suffix=_guess_audio_suffix(reference_filename, reference_content_type),
        delete=False,
    ) as tmp:
        tmp.write(ref_bytes)
        return tmp.name


def _transcribe_reference_audio(
    pipeline: PipelineOrchestrator,
    wav_path: str,
    language: str,
) -> Optional[str]:
    transcribe_file = getattr(pipeline.asr, "transcribe_file", None)
    if not callable(transcribe_file):
        return None
    try:
        result = transcribe_file(wav_path, language=language)
    except Exception as exc:
        log.warning("Reference transcription failed", path=wav_path, error=str(exc))
        return None
    text = (result.text or "").strip()
    return text or None


def _is_memory_error_message(msg: str) -> bool:
    msg_l = (msg or "").lower()
    return any(
        marker in msg_l
        for marker in (
            "pagefile",
            "paging file",
            "virtual memory",
            "os error 1455",
            "insufficient storage",
        )
    )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """
    Create FastAPI app. Settings can be injected in tests.
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

    static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
    if os.path.exists(static_dir):
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="static")

    @app.get("/", include_in_schema=False)
    async def docs_redirect():
        return RedirectResponse(url="/ui/")

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

    @app.get("/metrics", tags=["ops"])
    async def metrics():
        if _pipeline and hasattr(_pipeline.monitor, "get_metrics_text"):
            return Response(
                _pipeline.monitor.get_metrics_text(),
                media_type=_pipeline.monitor.content_type,
            )
        return Response("# no metrics", media_type="text/plain")

    @app.post("/tts", tags=["voice"])
    async def synthesize_text(req: TTSRequest):
        """
        Synthesize speech from text (non-streaming), returns base64 WAV.
        """
        if _pipeline is None:
            raise HTTPException(503, "Pipeline not ready")

        session_id = req.session_id or str(uuid.uuid4())
        sample_rate = _pipeline.settings.tts.sample_rate
        audio_parts = []
        temp_reference_path: Optional[str] = None
        reference_text: Optional[str] = None

        if req.reference_wav_b64:
            temp_reference_path = _decode_b64_wav_to_temp(
                req.reference_wav_b64,
                req.reference_filename,
                req.reference_content_type,
            )
            reference_text = _transcribe_reference_audio(_pipeline, temp_reference_path, req.language)

        try:
            async for chunk in _pipeline.process_text(
                user_text=req.text,
                session_id=session_id,
                language=req.language,
                voice_description=req.voice_description,
                skip_slm=req.skip_slm,
                reference_wav=temp_reference_path,
                emotion=req.emotion,
                voice=req.voice,
                reference_text=reference_text,
            ):
                audio_parts.append(chunk.audio)
                sample_rate = chunk.sample_rate
        except RuntimeError as exc:
            msg = str(exc)
            if _is_memory_error_message(msg):
                raise HTTPException(507, msg)
            raise
        finally:
            if temp_reference_path and os.path.exists(temp_reference_path):
                os.unlink(temp_reference_path)

        if not audio_parts:
            raise HTTPException(500, "No audio generated")

        combined = np.concatenate(audio_parts)
        buf = io.BytesIO()
        sf.write(buf, combined, samplerate=sample_rate, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode()
        return {"audio_base64": audio_b64, "session_id": session_id}

    @app.post("/clone", tags=["voice"])
    async def clone_voice(req: CloneVoiceRequest):
        """
        Clone a voice from a base64 wav and synthesize text.
        """
        if _pipeline is None:
            raise HTTPException(503, "Pipeline not ready")

        tmp_path = _decode_b64_wav_to_temp(
            req.reference_wav_b64,
            req.reference_filename,
            req.reference_content_type,
        )
        reference_text = _transcribe_reference_audio(_pipeline, tmp_path, req.language)
        try:
            session_id = str(uuid.uuid4())
            sample_rate = _pipeline.settings.tts.sample_rate
            audio_parts = []

            async for chunk in _pipeline.process_text(
                user_text=req.text,
                session_id=session_id,
                language=req.language,
                skip_slm=req.skip_slm,
                reference_wav=tmp_path,
                emotion=req.emotion,
                voice=req.voice,
                reference_text=reference_text,
            ):
                audio_parts.append(chunk.audio)
                sample_rate = chunk.sample_rate

            if not audio_parts:
                raise HTTPException(500, "No audio generated")

            combined = np.concatenate(audio_parts)
            buf = io.BytesIO()
            sf.write(buf, combined, samplerate=sample_rate, format="WAV")
            return {"audio_base64": base64.b64encode(buf.getvalue()).decode()}
        except RuntimeError as exc:
            msg = str(exc)
            if _is_memory_error_message(msg):
                raise HTTPException(507, msg)
            raise
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @app.websocket("/ws/voice")
    async def voice_ws(ws: WebSocket):
        """
        Full-duplex voice endpoint.

        Real-time path:
          - Client streams float32 PCM chunks.
          - Server splits into VAD-sized frames.
          - ASR/TTS runs only after speech-end is detected.
        """
        global _active_sessions

        if _pipeline is None:
            await ws.close(code=1013, reason="Server not ready")
            return

        await ws.accept()
        session_id = str(uuid.uuid4())
        _active_sessions += 1
        _pipeline.monitor.set_active_sessions(_active_sessions)

        session_reference_wav: Optional[str] = None
        session_reference_temp: Optional[str] = None
        session_reference_text: Optional[str] = None
        log.info("WebSocket connected", session_id=session_id)

        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")

                if msg_type == "audio_chunk":
                    language = data.get("language", "auto")
                    skip_slm = bool(data.get("skip_slm", False))
                    emotion = data.get("emotion", "neutral")
                    voice = data.get("voice")

                    audio_bytes = base64.b64decode(data["audio"])
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    if audio_array.size == 0:
                        continue

                    chunk_samples = _pipeline.settings.vad.chunk_samples
                    remainder = audio_array.size % chunk_samples
                    if remainder:
                        audio_array = np.pad(audio_array, (0, chunk_samples - remainder))

                    for i in range(0, audio_array.size, chunk_samples):
                        frame = np.ascontiguousarray(
                            audio_array[i : i + chunk_samples], dtype=np.float32
                        )
                        vad_result = _pipeline.vad.process_chunk(frame)
                        if not vad_result.is_speech_end or vad_result.utterance is None:
                            continue

                        t_request = time.perf_counter()
                        await ws.send_json({"type": "status", "msg": "asr_start"})

                        first_audio = True
                        async for chunk in _pipeline.process_audio(
                            vad_result.utterance,
                            session_id,
                            language,
                            skip_slm=skip_slm,
                            reference_wav=session_reference_wav,
                            emotion=emotion,
                            voice=voice,
                            reference_text=session_reference_text,
                        ):
                            if first_audio:
                                ttfb = round((time.perf_counter() - t_request) * 1000, 1)
                                await ws.send_json({"type": "ttfb", "ms": ttfb})
                                first_audio = False

                            buf = io.BytesIO()
                            sf.write(
                                buf,
                                chunk.audio,
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

                elif msg_type == "text_query":
                    language = data.get("language", "ta")
                    text = data.get("text", "")
                    skip_slm = bool(data.get("skip_slm", True))

                    async for chunk in _pipeline.process_text(
                        text,
                        session_id,
                        language,
                        voice_description=data.get("voice_description"),
                        skip_slm=skip_slm,
                        reference_wav=session_reference_wav,
                        emotion=data.get("emotion", "neutral"),
                        voice=data.get("voice"),
                        reference_text=session_reference_text,
                    ):
                        buf = io.BytesIO()
                        sf.write(
                            buf,
                            chunk.audio,
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

                elif msg_type == "reset":
                    _pipeline.reset_session(session_id)
                    await ws.send_json({"type": "status", "msg": "session_reset"})

                elif msg_type == "update_voice":
                    wav_path = data.get("path", "")
                    reference_language = data.get("language", "ta")
                    if not os.path.exists(wav_path):
                        await ws.send_json({
                            "type": "error",
                            "stage": "voice",
                            "msg": f"Reference wav not found: {wav_path}",
                        })
                        continue
                    session_reference_wav = wav_path
                    session_reference_text = _transcribe_reference_audio(
                        _pipeline,
                        wav_path,
                        reference_language,
                    )
                    await ws.send_json({"type": "status", "msg": "voice_updated"})

                elif msg_type == "update_voice_b64":
                    b64_audio = data.get("reference_wav_b64", "")
                    reference_language = data.get("language", "ta")
                    if not b64_audio:
                        await ws.send_json({
                            "type": "error",
                            "stage": "voice",
                            "msg": "Missing reference_wav_b64",
                        })
                        continue

                    try:
                        tmp_path = _decode_b64_wav_to_temp(
                            b64_audio,
                            data.get("reference_filename"),
                            data.get("reference_content_type"),
                        )
                    except Exception as exc:
                        await ws.send_json({
                            "type": "error",
                            "stage": "voice",
                            "msg": f"Invalid base64 audio: {exc}",
                        })
                        continue

                    if session_reference_temp and os.path.exists(session_reference_temp):
                        os.unlink(session_reference_temp)

                    session_reference_temp = tmp_path
                    session_reference_wav = tmp_path
                    session_reference_text = _transcribe_reference_audio(
                        _pipeline,
                        tmp_path,
                        reference_language,
                    )
                    await ws.send_json({"type": "status", "msg": "voice_updated"})

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
            if session_reference_temp and os.path.exists(session_reference_temp):
                os.unlink(session_reference_temp)
            _active_sessions -= 1
            _pipeline.monitor.set_active_sessions(_active_sessions)

    return app


app = create_app()
