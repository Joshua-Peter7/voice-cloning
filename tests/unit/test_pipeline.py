"""
tests/unit/test_pipeline.py
────────────────────────────
Unit tests for pipeline components using mocks.
Run with:  pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from core.interfaces import (
    ASRResult, BaseASR, BaseSLM, BaseVAD, SLMToken, TTSChunk, VADResult,
)
from core.pipeline import PipelineOrchestrator
from core.profiler import simulate_benchmark
from config.settings import load_settings
from session.manager import InMemorySessionStore
from monitoring.metrics import NullMonitor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def mock_vad():
    vad = MagicMock(spec=BaseVAD)
    vad.process_chunk.return_value = VADResult(
        is_speech_end=True,
        utterance=np.zeros(16000, dtype=np.float32),
        speech_probability=0.9,
    )
    return vad


@pytest.fixture
def mock_asr():
    asr = MagicMock(spec=BaseASR)
    asr.transcribe.return_value = ASRResult(
        text="வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        language="ta",
        language_name="Tamil",
        confidence=0.98,
        latency_ms=145.0,
    )
    return asr


@pytest.fixture
def mock_slm():
    class MockSLM(BaseSLM):
        async def generate_stream(self, user_text, history, language="auto"):
            for word in ["நான்", " நன்றாக", " இருக்கிறேன்", "."]:
                yield SLMToken(text=word)
            yield SLMToken(text="", is_final=True)

        async def generate(self, user_text, history, language="auto"):
            return "நான் நன்றாக இருக்கிறேன்."

    return MockSLM()


@pytest.fixture
def mock_tts():
    class MockTTSEngine:
        async def stream_from_tokens(
            self,
            token_gen,
            language,
            reference_wav=None,
            voice_description=None,
            emotion=None,
            voice=None,
            reference_text=None,
        ):
            # consume all tokens
            async for _ in token_gen:
                pass
            yield TTSChunk(
                audio=np.zeros(24000, dtype=np.float32),
                sample_rate=24000,
                text_segment="நான் நன்றாக இருக்கிறேன்.",
                latency_ms=210.0,
                chunk_index=0,
            )

    return MockTTSEngine()


@pytest.fixture
def session_store():
    from config.settings import SessionConfig
    return InMemorySessionStore(SessionConfig())


@pytest.fixture
def monitor():
    return NullMonitor()


@pytest.fixture
def pipeline(settings, mock_vad, mock_asr, mock_slm, mock_tts, session_store, monitor):
    return PipelineOrchestrator(
        settings=settings,
        vad=mock_vad,
        asr=mock_asr,
        slm=mock_slm,
        tts=mock_tts,
        sessions=session_store,
        monitor=monitor,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSettings:
    def test_load_default_settings(self, settings):
        assert settings.app.name == "indic-voice-cloner"
        assert "ta" in settings.languages.primary
        assert "te" in settings.languages.primary
        assert "kn" in settings.languages.primary
        assert "ml" in settings.languages.primary

    def test_south_indian_languages_in_primary(self, settings):
        primary = settings.languages.primary
        assert set(["ta", "te", "kn", "ml"]).issubset(set(primary))

    def test_parler_voice_descriptions(self, settings):
        desc = settings.tts.parler.voice_for("ta")
        assert "Tamil" in desc
        desc_kn = settings.tts.parler.voice_for("kn")
        assert "Kannada" in desc_kn

    def test_unknown_language_uses_default(self, settings):
        desc = settings.tts.parler.voice_for("xx")
        assert desc == settings.tts.parler.voice_descriptions["default"]


class TestVAD:
    def test_process_chunk_returns_vad_result(self, mock_vad):
        chunk = np.zeros(512, dtype=np.float32)
        result = mock_vad.process_chunk(chunk)
        assert isinstance(result, VADResult)
        assert result.is_speech_end is True

    def test_reset_called(self, mock_vad):
        mock_vad.reset()
        mock_vad.reset.assert_called_once()


class TestASR:
    def test_transcribe_returns_result(self, mock_asr):
        audio = np.zeros(16000 * 3, dtype=np.float32)
        result = mock_asr.transcribe(audio, language="ta")
        assert isinstance(result, ASRResult)
        assert result.language == "ta"
        assert len(result.text) > 0
        assert result.confidence > 0

    def test_transcribe_tamil(self, mock_asr):
        result = mock_asr.transcribe(np.zeros(16000, dtype=np.float32), language="ta")
        assert result.language_name == "Tamil"


class TestSessionStore:
    def test_empty_history(self, session_store):
        history = session_store.get_history("test-001")
        assert history == []

    def test_append_and_retrieve(self, session_store):
        session_store.append_turn("test-002", "வணக்கம்", "வணக்கம்!")
        history = session_store.get_history("test-002")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_max_history_truncation(self, session_store):
        sid = "test-003"
        for i in range(20):
            session_store.append_turn(sid, f"msg_{i}", f"resp_{i}")
        history = session_store.get_history(sid)
        # max_history_turns=6 → max 12 messages
        assert len(history) <= 12

    def test_clear_session(self, session_store):
        session_store.append_turn("test-004", "hello", "hi")
        session_store.clear("test-004")
        assert session_store.get_history("test-004") == []


class TestPipeline:
    def test_pipeline_initialization(self, pipeline):
        assert pipeline.vad is not None
        assert pipeline.asr is not None
        assert pipeline.slm is not None
        assert pipeline.tts is not None
        assert pipeline.sessions is not None
        assert pipeline.monitor is not None

    @pytest.mark.asyncio
    async def test_process_audio_yields_chunks(self, pipeline):
        audio = np.zeros(16000 * 3, dtype=np.float32)
        chunks = []
        async for chunk in pipeline.process_audio(audio, "session-001", "ta"):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert all(isinstance(c, TTSChunk) for c in chunks)

    @pytest.mark.asyncio
    async def test_process_text_yields_chunks(self, pipeline):
        chunks = []
        async for chunk in pipeline.process_text("வணக்கம்", "session-002", "ta"):
            chunks.append(chunk)
        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_session_history_saved(self, pipeline):
        audio = np.zeros(16000 * 3, dtype=np.float32)
        async for _ in pipeline.process_audio(audio, "session-003", "ta"):
            pass
        history = pipeline.sessions.get_history("session-003")
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_process_audio_skip_slm(self, pipeline):
        async def _should_not_run(*args, **kwargs):
            raise AssertionError("SLM should be skipped when skip_slm=True")
            yield

        pipeline.slm.generate_stream = _should_not_run
        chunks = []
        audio = np.zeros(16000 * 3, dtype=np.float32)
        async for chunk in pipeline.process_audio(
            audio,
            "session-004b",
            "ta",
            skip_slm=True,
        ):
            chunks.append(chunk)
        assert len(chunks) > 0

    def test_reset_session(self, pipeline, session_store):
        session_store.append_turn("session-004", "hi", "hello")
        pipeline.reset_session("session-004")
        assert session_store.get_history("session-004") == []


class TestProfiler:
    def test_simulate_benchmark(self):
        profiler = simulate_benchmark(n=20)
        assert len(profiler.profiles) == 20

    def test_report_structure(self):
        profiler = simulate_benchmark(n=10)
        report = profiler.report()
        assert "samples" in report
        assert "total_ttfb" in report
        assert "sla_pass_rate" in report
        assert "by_language" in report

    def test_sla_pass_rate_range(self):
        profiler = simulate_benchmark(n=50)
        report = profiler.report()
        rate = report["sla_pass_rate"]
        assert 0.0 <= rate <= 100.0

    def test_print_report_does_not_crash(self, capsys):
        profiler = simulate_benchmark(n=10)
        profiler.print_report()
        captured = capsys.readouterr()
        assert "LATENCY BENCHMARK" in captured.out
        assert "Tamil" in captured.out or "ta" in captured.out.lower()
