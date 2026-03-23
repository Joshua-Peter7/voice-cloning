"""
core/pipeline.py
────────────────
The PipelineOrchestrator wires all components together via dependency injection.
Components are swappable — pass any object implementing the right interface.

Dependency Graph:
    Settings → VAD → ASR → SLM → TTS → Response
                                     ↓
                              SessionStore
                                     ↓
                               Monitor
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import numpy as np

from core.interfaces import (
    BaseASR,
    BaseMonitor,
    BaseSLM,
    BaseSessionStore,
    BaseVAD,
    LatencyProfile,
    TTSChunk,
)
from core.logger import get_logger
from tts.engine import StreamingTTSEngine
from config.settings import Settings

log = get_logger("core.pipeline")


class PipelineOrchestrator:
    """
    Full pipeline: audio_in → VAD → ASR → SLM → TTS → audio_out

    All components are injected, making the pipeline fully testable
    and modifiable without touching orchestration logic.

    Usage:
        pipeline = PipelineOrchestrator(settings, vad, asr, slm, tts, sessions, monitor)
        async for chunk in pipeline.process_audio(audio_array, session_id):
            send_to_client(chunk)
    """

    def __init__(
        self,
        settings: Settings,
        vad: BaseVAD,
        asr: BaseASR,
        slm: BaseSLM,
        tts: StreamingTTSEngine,
        sessions: BaseSessionStore,
        monitor: BaseMonitor,
        reference_wav: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.vad = vad
        self.asr = asr
        self.slm = slm
        self.tts = tts
        self.sessions = sessions
        self.monitor = monitor
        self.reference_wav = reference_wav
        log.info("PipelineOrchestrator initialized")

    def _resolve_reference_wav(self, candidate: Optional[str]) -> Optional[str]:
        """Return a usable reference wav path or None."""
        ref = candidate or self.reference_wav
        if not ref:
            return None
        if not os.path.exists(ref):
            log.warning("Reference wav missing, continuing without cloning reference", path=ref)
            return None
        return ref

    async def process_audio(
        self,
        audio_array: np.ndarray,
        session_id: str,
        language: str = "auto",
        skip_slm: bool = False,
        reference_wav: Optional[str] = None,
        emotion: Optional[str] = None,
        voice: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Full pipeline from raw audio to TTS audio chunks.

        Args:
            audio_array: float32, 16kHz mono numpy array (complete utterance).
            session_id:  Unique session identifier.
            language:    Language hint; "auto" = detect from ASR.
            skip_slm:    If True, speak ASR text directly without LLM rewrite.
            reference_wav: Optional per-request reference voice override.

        Yields:
            TTSChunk objects (audio + metadata) as they're synthesized.
        """
        t_start = time.perf_counter()
        profile = LatencyProfile(session_id=session_id)

        # ── Stage 1: ASR ──────────────────────────────────────────────
        log.debug("ASR start", session_id=session_id)
        try:
            asr_result = self.asr.transcribe(audio_array, language=language)
        except Exception as exc:
            self.monitor.increment_error("asr")
            log.error("ASR failed", error=str(exc), session_id=session_id)
            raise

        profile.asr_ms = asr_result.latency_ms
        profile.language = asr_result.language
        profile.text_length = len(asr_result.text)

        log.info(
            "ASR complete",
            text=asr_result.text[:80],
            lang=asr_result.language,
            latency_ms=asr_result.latency_ms,
            session_id=session_id,
        )

        if not asr_result.text.strip():
            log.debug("Empty ASR result, skipping response", session_id=session_id)
            return

        # ── Stage 2: Session history ───────────────────────────────────
        history = self.sessions.get_history(session_id)

        # ── Stage 3: SLM → TTS streaming pipeline ─────────────────────
        full_response_parts: List[str] = []
        first_chunk_sent = False

        async def _token_gen():
            """Bridge: SLMToken stream → string token stream for TTS."""
            if skip_slm:
                full_response_parts.append(asr_result.text)
                yield asr_result.text
            else:
                async for tok in self.slm.generate_stream(
                    asr_result.text, history, asr_result.language
                ):
                    full_response_parts.append(tok.text)
                    yield tok.text

        t_slm_start = time.perf_counter()

        ref_wav = self._resolve_reference_wav(reference_wav)

        async for chunk in self.tts.stream_from_tokens(
            _token_gen(),
            language=asr_result.language,
            reference_wav=ref_wav,
            emotion=emotion,
            voice=voice,
            reference_text=reference_text,
        ):
            if not first_chunk_sent:
                profile.tts_first_chunk_ms = chunk.latency_ms
                profile.slm_ttft_ms = round(
                    (time.perf_counter() - t_slm_start) * 1000 - chunk.latency_ms, 2
                )
                profile.total_ttfb_ms = round(
                    (time.perf_counter() - t_start) * 1000, 2
                )
                first_chunk_sent = True
                log.info(
                    "First audio chunk ready",
                    ttfb_ms=profile.total_ttfb_ms,
                    session_id=session_id,
                )
            yield chunk

        # ── Stage 4: Post-turn bookkeeping ────────────────────────────
        full_response = "".join(full_response_parts)
        self.sessions.append_turn(session_id, asr_result.text, full_response)

        self.monitor.record_request(profile, status="success")
        log.info(
            "Turn complete",
            total_ttfb_ms=profile.total_ttfb_ms,
            session_id=session_id,
        )

    async def process_text(
        self,
        user_text: str,
        session_id: str,
        language: str = "ta",
        voice_description: Optional[str] = None,
        skip_slm: bool = False,
        reference_wav: Optional[str] = None,
        emotion: Optional[str] = None,
        voice: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Text-in → TTS streaming (skips VAD + ASR).
        Useful for text-based API endpoints and testing.
        """
        t_start = time.perf_counter()
        profile = LatencyProfile(session_id=session_id, language=language)
        history = self.sessions.get_history(session_id)
        full_response_parts: List[str] = []
        first_chunk_sent = False

        async def _token_gen():
            if skip_slm:
                # Bypass SLM: yield the full text AT ONCE so utterance-level
                # models (F5, Svara) get the complete text, not word fragments.
                full_response_parts.append(user_text)
                yield user_text
            else:
                async for tok in self.slm.generate_stream(user_text, history, language):
                    full_response_parts.append(tok.text)
                    yield tok.text

        ref_wav = self._resolve_reference_wav(reference_wav)

        async for chunk in self.tts.stream_from_tokens(
            _token_gen(),
            language=language,
            reference_wav=ref_wav,
            voice_description=voice_description,
            emotion=emotion,
            voice=voice,
            reference_text=reference_text,
        ):
            if not first_chunk_sent:
                profile.total_ttfb_ms = round(
                    (time.perf_counter() - t_start) * 1000, 2
                )
                first_chunk_sent = True
            yield chunk

        full_response = "".join(full_response_parts)
        self.sessions.append_turn(session_id, user_text, full_response)
        self.monitor.record_request(profile)

    def reset_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.sessions.clear(session_id)
        self.vad.reset()
        log.info("Session reset", session_id=session_id)

    def update_reference_voice(self, wav_path: str) -> None:
        """
        Hot-swap the reference voice for XTTS cloning.
        No restart required.
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Reference wav not found: {wav_path}")
        self.reference_wav = wav_path
        log.info("Reference voice updated", path=wav_path)


def build_pipeline(
    settings: Optional[Settings] = None,
    reference_wav: Optional[str] = None,
) -> PipelineOrchestrator:
    """
    Convenience factory: build a fully-wired pipeline from settings.
    For production use. Each component can be replaced individually.
    """
    from config.settings import get_settings
    from vad.silero_vad import create_vad
    from asr.whisper_asr import create_asr
    from slm.vllm_engine import create_slm
    from tts.engine import create_tts_engine
    from session.manager import create_session_store
    from monitoring.metrics import create_monitor

    if settings is None:
        settings = get_settings()

    return PipelineOrchestrator(
        settings=settings,
        vad=create_vad(settings.vad),
        asr=create_asr(settings.asr),
        slm=create_slm(settings.slm),
        tts=create_tts_engine(settings.tts),
        sessions=create_session_store(settings.session),
        monitor=create_monitor(settings.monitoring),
        reference_wav=(
            reference_wav
            or (
                str(Path(settings.infra.audio_samples_dir) / "reference_voice.wav")
                if (Path(settings.infra.audio_samples_dir) / "reference_voice.wav").exists()
                else None
            )
        ),
    )
