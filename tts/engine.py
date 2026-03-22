"""
tts/engine.py
─────────────
Dual-backend TTS engine with streaming support.

Backend 1 — Indic-Parler-TTS
  • Best quality for Tamil, Telugu, Kannada, Malayalam + other Indic langs
  • Prompt-driven voice description (language-specific voices configurable)
  • No reference audio needed

Backend 2 — XTTS v2
  • Best for voice cloning from a 6–30s reference .wav
  • Supports all 10 Indic languages + English
  • Slightly higher latency than Parler

Streaming strategy:
  • Buffer N words from SLM token stream
  • Fire TTS on sentence end OR word-count threshold
  • Yield audio chunks as they're ready → low TTFB
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, List, Optional, Set

import numpy as np
import torch

from core.interfaces import BaseTTSBackend, TTSChunk
from core.device import resolve_device, get_dtype
from core.logger import get_logger
from config.settings import TTSConfig, ParlerConfig, XTTSConfig

log = get_logger("tts.engine")


# ── Backend: Indic-Parler-TTS ─────────────────────────────────────────────────

class ParlerTTSBackend(BaseTTSBackend):
    """
    Indic-Parler-TTS backend.

    Excels at: Tamil, Telugu, Kannada, Malayalam, Hindi.
    Voice quality controlled via text description prompt.
    """

    # All languages supported by indic-parler-tts
    _SUPPORTED: List[str] = ["ta", "te", "kn", "ml", "hi", "bn", "mr", "gu", "pa", "en"]

    def __init__(self, cfg: ParlerConfig) -> None:
        self.cfg = cfg
        self._device = resolve_device(cfg.device)
        self._dtype = get_dtype(self._device, cfg.dtype)
        self._model = None
        self._tokenizer = None
        self._sample_rate_val: int = 44100  # Parler outputs 44.1kHz
        log.info("ParlerTTS backend created (lazy load)")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        log.info("Loading Indic-Parler-TTS", device=self._device)
        self._model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.cfg.model_id,
            torch_dtype=self._dtype,
        ).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        self._sample_rate_val = self._model.config.sampling_rate
        log.info("Indic-Parler-TTS ready", sample_rate=self._sample_rate_val)

    def synthesize(
        self,
        text: str,
        language: str = "ta",
        reference_wav: Optional[str] = None,  # unused; kept for interface compat
        **kwargs,
    ) -> np.ndarray:
        """
        Synthesize speech.

        Args:
            text: Input text (in target language script).
            language: ISO 639-1 language code.
            kwargs:
                voice_description (str): override the default voice description prompt.
        Returns:
            float32 PCM array at self.sample_rate.
        """
        self._ensure_loaded()
        description = kwargs.get("voice_description") or self.cfg.voice_for(language)

        desc_inputs = self._tokenizer(description, return_tensors="pt").to(self._device)
        prompt_ids = self._tokenizer(text, return_tensors="pt").input_ids.to(self._device)

        with torch.no_grad():
            generation = self._model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_ids,
            )

        audio = generation.cpu().numpy().squeeze().astype(np.float32)
        log.debug("Parler synthesis done", text_len=len(text), lang=language)
        return audio

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_val

    @property
    def supported_languages(self) -> List[str]:
        return self._SUPPORTED


# ── Backend: XTTS v2 (Voice Cloning) ─────────────────────────────────────────

class XTTSBackend(BaseTTSBackend):
    """
    XTTS v2 backend — voice cloning from a reference audio file.

    Best for: high-quality voice cloning when speaker identity matters.
    Reference audio: 6–30 seconds, clear speech, single speaker.
    """

    _SUPPORTED: List[str] = ["ta", "te", "kn", "ml", "hi", "bn", "mr", "gu", "pa", "en",
                              "zh-cn", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
                              "cs", "ar", "hu", "ko", "ja"]

    def __init__(self, cfg: XTTSConfig) -> None:
        self.cfg = cfg
        self._device = resolve_device(cfg.device)
        self._model = None
        self._sample_rate_val: int = 24000
        log.info("XTTS backend created (lazy load)")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from TTS.api import TTS

        log.info("Loading XTTS v2", device=self._device)
        self._model = TTS(self.cfg.model_id).to(self._device)
        self._sample_rate_val = 24000
        log.info("XTTS v2 ready")

    def synthesize(
        self,
        text: str,
        language: str = "ta",
        reference_wav: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Clone voice from reference and synthesize text.

        Args:
            text: Input text.
            language: ISO 639-1 code.
            reference_wav: Path to 6–30s reference .wav file.
        """
        self._ensure_loaded()
        if not reference_wav:
            raise ValueError("XTTS requires a reference_wav for voice cloning.")

        audio = self._model.tts(
            text=text,
            speaker_wav=reference_wav,
            language=language,
            split_sentences=self.cfg.split_sentences,
        )
        arr = np.array(audio, dtype=np.float32)
        log.debug("XTTS synthesis done", text_len=len(text), lang=language)
        return arr

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_val

    @property
    def supported_languages(self) -> List[str]:
        return self._SUPPORTED


# ── Streaming TTS Engine ──────────────────────────────────────────────────────

class StreamingTTSEngine:
    """
    Manages backend selection and the streaming synthesis loop.

    The streaming loop:
      1. Receives tokens from the SLM token generator
      2. Accumulates text in a buffer
      3. Fires TTS when a sentence boundary or word-count threshold is hit
      4. Yields TTSChunk objects as audio becomes available

    This achieves low TTFB (~200ms for first audio) without waiting
    for the full LLM response.
    """

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._backends: dict[str, BaseTTSBackend] = {}
        self._sentence_ends: Set[str] = set(cfg.sentence_ends)
        self._load_default_backend()

    def _load_default_backend(self) -> None:
        """Pre-load the configured default backend."""
        if self.cfg.backend == "parler":
            self._backends["parler"] = ParlerTTSBackend(self.cfg.parler)
        elif self.cfg.backend == "xtts":
            self._backends["xtts"] = XTTSBackend(self.cfg.xtts)
        elif self.cfg.backend == "auto":
            # Load both; auto-select at synthesis time
            self._backends["parler"] = ParlerTTSBackend(self.cfg.parler)
            self._backends["xtts"] = XTTSBackend(self.cfg.xtts)

    def _select_backend(
        self,
        language: str,
        reference_wav: Optional[str],
    ) -> BaseTTSBackend:
        """
        Auto-select backend:
          - XTTS when reference audio is provided
          - Parler when language is a primary Indic lang
          - XTTS as fallback
        """
        if reference_wav:
            return self._backends.get("xtts") or self._backends.get("parler")
        if language in ParlerTTSBackend._SUPPORTED:
            return self._backends.get("parler") or self._backends.get("xtts")
        return self._backends.get("xtts") or list(self._backends.values())[0]

    def synthesize_once(
        self,
        text: str,
        language: str = "ta",
        reference_wav: Optional[str] = None,
        voice_description: Optional[str] = None,
    ) -> TTSChunk:
        """Synthesize a complete text segment (non-streaming)."""
        t0 = time.perf_counter()
        backend = self._select_backend(language, reference_wav)
        audio = backend.synthesize(
            text,
            language=language,
            reference_wav=reference_wav,
            voice_description=voice_description,
        )
        return TTSChunk(
            audio=audio,
            sample_rate=backend.sample_rate,
            text_segment=text,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    async def stream_from_tokens(
        self,
        token_generator: AsyncGenerator[str, None],
        language: str = "ta",
        reference_wav: Optional[str] = None,
        voice_description: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Core streaming loop: buffer tokens → fire TTS → yield audio.

        Trigger conditions (whichever comes first):
          1. Sentence-ending punctuation + ≥3 words in buffer
          2. Word count reaches cfg.stream_chunk_words

        This ensures first audio arrives ~200ms after first SLM token.
        """
        buffer = ""
        chunk_index = 0
        backend = self._select_backend(language, reference_wav)

        async for token in token_generator:
            buffer += token
            word_count = len(buffer.split())

            has_sentence_end = any(buffer.rstrip().endswith(p) for p in self._sentence_ends)
            should_fire = (
                (has_sentence_end and word_count >= 3)
                or word_count >= self.cfg.stream_chunk_words
            )

            if should_fire and buffer.strip():
                segment = buffer.strip()
                buffer = ""
                t0 = time.perf_counter()

                try:
                    # Offload CPU-heavy synthesis to thread pool to keep event loop free
                    audio = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: backend.synthesize(
                            segment,
                            language=language,
                            reference_wav=reference_wav,
                            voice_description=voice_description,
                        ),
                    )
                    latency = round((time.perf_counter() - t0) * 1000, 2)
                    log.debug(
                        "TTS chunk ready",
                        chunk=chunk_index,
                        text=segment[:40],
                        latency_ms=latency,
                    )
                    yield TTSChunk(
                        audio=audio,
                        sample_rate=backend.sample_rate,
                        text_segment=segment,
                        latency_ms=latency,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
                except Exception as exc:
                    log.error("TTS synthesis failed", segment=segment, error=str(exc))
                    raise

        # Flush remaining buffer
        if buffer.strip():
            segment = buffer.strip()
            t0 = time.perf_counter()
            audio = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: backend.synthesize(
                    segment,
                    language=language,
                    reference_wav=reference_wav,
                    voice_description=voice_description,
                ),
            )
            yield TTSChunk(
                audio=audio,
                sample_rate=backend.sample_rate,
                text_segment=segment,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                chunk_index=chunk_index,
            )

    def add_backend(self, name: str, backend: BaseTTSBackend) -> None:
        """
        Register a custom TTS backend.
        Allows swapping in proprietary or fine-tuned models without
        changing any other code.

        Example:
            engine.add_backend("my_model", MyCustomTTSBackend(cfg))
        """
        self._backends[name] = backend
        log.info("Custom TTS backend registered", name=name)

    def list_backends(self) -> List[str]:
        return list(self._backends.keys())


def create_tts_engine(cfg: TTSConfig) -> StreamingTTSEngine:
    """Factory: create the configured TTS engine."""
    return StreamingTTSEngine(cfg)
