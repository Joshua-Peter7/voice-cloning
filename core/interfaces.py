"""
core/interfaces.py
──────────────────
Abstract base classes for all pipeline components.
Swap any component by implementing its interface — no changes elsewhere.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np


# ── Data Transfer Objects ─────────────────────────────────────────────────────

@dataclass
class ASRResult:
    text: str
    language: str
    language_name: str
    confidence: float
    latency_ms: float
    word_timestamps: Optional[List[Dict]] = None


@dataclass
class TTSChunk:
    audio: np.ndarray          # float32 PCM
    sample_rate: int
    text_segment: str
    latency_ms: float
    chunk_index: int = 0


@dataclass
class SLMToken:
    text: str
    is_final: bool = False
    latency_ms: float = 0.0


@dataclass
class VADResult:
    is_speech_end: bool
    utterance: Optional[np.ndarray]  # Full utterance if speech ended
    speech_probability: float = 0.0


@dataclass
class LatencyProfile:
    vad_ms: float = 0.0
    asr_ms: float = 0.0
    slm_ttft_ms: float = 0.0        # Time to first token
    tts_first_chunk_ms: float = 0.0
    total_ttfb_ms: float = 0.0      # Time to first audio byte
    language: str = "unknown"
    text_length: int = 0
    session_id: str = ""


# ── Abstract Interfaces ───────────────────────────────────────────────────────

class BaseVAD(abc.ABC):
    """Voice Activity Detection interface."""

    @abc.abstractmethod
    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        """
        Process a fixed-size audio chunk.
        Must be called with exactly cfg.vad.chunk_samples samples.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state between sessions."""


class BaseASR(abc.ABC):
    """Automatic Speech Recognition interface."""

    @abc.abstractmethod
    def transcribe(
        self,
        audio_array: np.ndarray,
        language: str = "auto",
    ) -> ASRResult:
        """
        Transcribe a complete utterance.
        audio_array: float32, 16kHz mono.
        """

    def transcribe_stream(
        self,
        audio_array: np.ndarray,
    ) -> AsyncGenerator[ASRResult, None]:
        """Optional: yield partial results as segments complete."""
        raise NotImplementedError


class BaseTTSBackend(abc.ABC):
    """Single TTS backend interface (Parler, XTTS, etc.)."""

    @abc.abstractmethod
    def synthesize(
        self,
        text: str,
        language: str,
        reference_wav: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Synthesize speech. Returns float32 PCM array."""

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""

    @property
    @abc.abstractmethod
    def supported_languages(self) -> List[str]:
        """Language codes this backend supports."""


class BaseSLM(abc.ABC):
    """Small Language Model interface."""

    @abc.abstractmethod
    async def generate_stream(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        language: str = "auto",
    ) -> AsyncGenerator[SLMToken, None]:
        """Stream tokens from the model."""

    @abc.abstractmethod
    async def generate(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        language: str = "auto",
    ) -> str:
        """Generate full response (non-streaming)."""


class BaseSessionStore(abc.ABC):
    """Conversation history store interface."""

    @abc.abstractmethod
    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Return conversation history for a session."""

    @abc.abstractmethod
    def append_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        """Append a user/assistant turn to history."""

    @abc.abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all history for a session."""


class BaseMonitor(abc.ABC):
    """Metrics / observability interface."""

    @abc.abstractmethod
    def record_request(self, profile: LatencyProfile, status: str = "success") -> None:
        """Record a completed request's latency profile."""

    @abc.abstractmethod
    def increment_error(self, stage: str) -> None:
        """Increment error counter for a pipeline stage."""

    @abc.abstractmethod
    def set_active_sessions(self, count: int) -> None:
        """Update active WebSocket session gauge."""
