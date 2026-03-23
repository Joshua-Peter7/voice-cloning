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
import gc
import time
from typing import AsyncGenerator, List, Optional, Set, Tuple

import numpy as np
import torch

from core.interfaces import BaseTTSBackend, TTSChunk
from core.device import resolve_device, get_dtype
from core.logger import get_logger
from config.settings import TTSConfig, ParlerConfig, XTTSConfig, F5Config
from tts.svara_backend import SvaraTTSBackend

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


# ── Backend: F5-TTS (Voice Cloning) ────────────────────────────────────────
class F5TTSBackend(BaseTTSBackend):
    """
    F5-TTS backend for pure-PyTorch high quality voice cloning.
    Excellent for Windows machines without C++ build tools.
    """
    _SUPPORTED: List[str] = ["en", "ta", "hi", "te", "ml", "kn", "bn", "mr", "gu", "pa", "zh-cn"]

    def __init__(self, cfg: F5Config) -> None:
        self.cfg = cfg
        self._device = resolve_device(cfg.device)
        self._model = None
        self._sample_rate_val: int = 24000
        log.info("F5-TTS backend created (lazy load)")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from f5_tts.api import F5TTS
        log.info("Loading F5-TTS Model", device=self._device)
        self._model = F5TTS(device=self._device)
        log.info("F5-TTS ready")

    def synthesize(
        self,
        text: str,
        language: str = "en",
        reference_wav: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        self._ensure_loaded()
        reference_text = (kwargs.get("reference_text") or "").strip()
        temp_reference_path: Optional[str] = None
        if not reference_wav:
            # Fast fallback if no reference.
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, np.zeros(24000 * 2, dtype=np.float32), 24000)
                reference_wav = tmp.name
                temp_reference_path = tmp.name

        try:
            wav, _, _ = self._model.infer(
                ref_file=reference_wav,
                ref_text=reference_text,
                gen_text=text,
            )
        finally:
            if temp_reference_path:
                import os

                try:
                    os.unlink(temp_reference_path)
                except OSError:
                    pass
        # F5-TTS infer returns float array, we just need to ensure precision
        arr = np.array(wav, dtype=np.float32)
        log.debug("F5-TTS synthesis done", text_len=len(text))
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

    def _create_svara_backend(self) -> SvaraTTSBackend:
        return SvaraTTSBackend(
            model_id=self.cfg.svara.model_id,
            device=self.cfg.svara.device,
            min_reference_seconds=self.cfg.svara.min_reference_seconds,
            max_reference_seconds=self.cfg.svara.max_reference_seconds,
            reference_target_seconds=self.cfg.svara.reference_target_seconds,
            max_new_tokens=self.cfg.svara.max_new_tokens,
            temperature=self.cfg.svara.temperature,
            top_p=self.cfg.svara.top_p,
        )

    def _ensure_backend_registered(self, name: str) -> None:
        if name in self._backends:
            return
        if name == "svara":
            self._backends["svara"] = self._create_svara_backend()
        elif name == "parler":
            self._backends["parler"] = ParlerTTSBackend(self.cfg.parler)
        elif name == "xtts":
            self._backends["xtts"] = XTTSBackend(self.cfg.xtts)
        elif name == "f5":
            self._backends["f5"] = F5TTSBackend(self.cfg.f5)

    def _load_default_backend(self) -> None:
        """
        Register lazy backends.

        - `auto`: make all built-ins available for robust runtime fallback.
        - explicit backend: register that backend; additional fallbacks are
          registered lazily in `_ordered_backend_names`.
        """
        if self.cfg.backend == "auto":
            for name in ("f5", "svara", "xtts", "parler"):
                self._ensure_backend_registered(name)
            return

        if self.cfg.backend in ("f5", "svara", "xtts", "parler"):
            self._ensure_backend_registered(self.cfg.backend)
        else:
            # Safe default if config has unexpected value.
            self._ensure_backend_registered("f5")

    def _ordered_backend_names(self, reference_wav: Optional[str]) -> List[str]:
        clone_order = ["f5", "svara", "xtts", "parler"]
        non_clone_order = ["parler", "svara", "f5", "xtts"]
        base_order = clone_order if reference_wav else non_clone_order

        # Ensure all candidates in this path are available for fallback.
        for name in base_order:
            self._ensure_backend_registered(name)
        if self.cfg.backend in ("f5", "svara", "xtts", "parler"):
            self._ensure_backend_registered(self.cfg.backend)

        ordered: List[str] = []
        preferred = self.cfg.backend if self.cfg.backend in self._backends else None
        if preferred:
            ordered.append(preferred)

        for name in base_order:
            if name in self._backends and name not in ordered:
                ordered.append(name)
        return ordered

    @staticmethod
    def _is_recoverable_backend_error(exc: Exception) -> bool:
        if isinstance(exc, ValueError):
            msg = str(exc).lower()
            if "requires a reference_wav" in msg or "unsupported" in msg:
                return True
        if isinstance(exc, (ModuleNotFoundError, ImportError)):
            return True
        if isinstance(exc, OSError):
            msg = str(exc).lower()
            if "os error 1455" in msg or "paging file is too small" in msg:
                return True
        msg = str(exc).lower()
        return (
            "no module named" in msg
            or "cuda out of memory" in msg
            or "outofmemoryerror" in msg
            or "pagefile" in msg
            or "virtual memory" in msg
            or "paging file" in msg
        )

    @staticmethod
    def _cleanup_after_backend_failure(backend: BaseTTSBackend) -> None:
        # Best-effort cleanup so a failed backend does not block the next fallback.
        if hasattr(backend, "_model"):
            try:
                setattr(backend, "_model", None)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _synthesize_with_fallback(
        self,
        text: str,
        language: str,
        reference_wav: Optional[str],
        voice_description: Optional[str],
        emotion: Optional[str],
        voice: Optional[str],
        reference_text: Optional[str],
    ) -> Tuple[np.ndarray, BaseTTSBackend]:
        ordered_names = self._ordered_backend_names(reference_wav)
        failures: List[str] = []

        for idx, name in enumerate(ordered_names):
            backend = self._backends[name]
            try:
                audio = backend.synthesize(
                    text,
                    language=language,
                    reference_wav=reference_wav,
                    voice_description=voice_description,
                    emotion=emotion,
                    voice=voice,
                    reference_text=reference_text,
                )
                if idx > 0:
                    log.warning(
                        "Using fallback TTS backend",
                        fallback_backend=name,
                        preferred_backend=ordered_names[0],
                    )
                return audio, backend
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                if not self._is_recoverable_backend_error(exc):
                    log.error(
                        "TTS backend failed with non-recoverable error",
                        backend=name,
                        error=str(exc),
                    )
                    raise
                log.warning(
                    "TTS backend unavailable, trying next fallback",
                    backend=name,
                    error=str(exc),
                )
                self._cleanup_after_backend_failure(backend)
                continue

        raise RuntimeError("All TTS backends failed: " + " | ".join(failures))

    def synthesize_once(
        self,
        text: str,
        language: str = "ta",
        reference_wav: Optional[str] = None,
        voice_description: Optional[str] = None,
        emotion: Optional[str] = None,
        voice: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> TTSChunk:
        """Synthesize a complete text segment (non-streaming)."""
        t0 = time.perf_counter()
        audio, backend = self._synthesize_with_fallback(
            text=text,
            language=language,
            reference_wav=reference_wav,
            voice_description=voice_description,
            emotion=emotion,
            voice=voice,
            reference_text=reference_text,
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
        emotion: Optional[str] = None,
        voice: Optional[str] = None,
        reference_text: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Core streaming loop: buffer tokens → fire TTS → yield audio.

        - F5-TTS / XTTS: Accumulate the FULL response, then synthesize once.
          (These are utterance-level models — chunking degrades quality severely.)
        - Parler: Stream in sentence chunks for low TTFB.
        """
        preferred_names = self._ordered_backend_names(reference_wav)
        preferred_backend = self._backends[preferred_names[0]]
        is_utterance_level = isinstance(
            preferred_backend,
            (F5TTSBackend, XTTSBackend, SvaraTTSBackend),
        )

        if is_utterance_level:
            # Collect the full text response before synthesizing
            full_text_parts: List[str] = []
            async for token in token_generator:
                full_text_parts.append(token)

            full_text = "".join(full_text_parts).strip()
            if not full_text:
                return

            t0 = time.perf_counter()
            try:
                audio, used_backend = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._synthesize_with_fallback(
                        text=full_text,
                        language=language,
                        reference_wav=reference_wav,
                        voice_description=voice_description,
                        emotion=emotion,
                        voice=voice,
                        reference_text=reference_text,
                    ),
                )
                latency = round((time.perf_counter() - t0) * 1000, 2)
                log.debug("TTS full utterance ready", text=full_text[:60], latency_ms=latency)
                yield TTSChunk(
                    audio=audio,
                    sample_rate=used_backend.sample_rate,
                    text_segment=full_text,
                    latency_ms=latency,
                    chunk_index=0,
                )
            except Exception as exc:
                log.error("TTS synthesis failed", segment=full_text, error=str(exc))
                raise
            return

        # Parler: sentence-chunked streaming for low TTFB
        buffer = ""
        chunk_index = 0

        async for token in token_generator:
            buffer += token
            word_count = len(buffer.split())

            has_sentence_end = any(buffer.rstrip().endswith(p) for p in self._sentence_ends)
            should_fire = (
                (has_sentence_end and word_count >= 8)
                or word_count >= self.cfg.stream_chunk_words
            )

            if should_fire and buffer.strip():
                segment = buffer.strip()
                buffer = ""
                t0 = time.perf_counter()

                try:
                    audio, used_backend = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda s=segment: self._synthesize_with_fallback(
                            text=s,
                            language=language,
                            reference_wav=reference_wav,
                            voice_description=voice_description,
                            emotion=emotion,
                            voice=voice,
                            reference_text=reference_text,
                        ),
                    )
                    latency = round((time.perf_counter() - t0) * 1000, 2)
                    log.debug("TTS chunk ready", chunk=chunk_index, text=segment[:40], latency_ms=latency)
                    yield TTSChunk(
                        audio=audio,
                        sample_rate=used_backend.sample_rate,
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
            audio, used_backend = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda s=segment: self._synthesize_with_fallback(
                    text=s,
                    language=language,
                    reference_wav=reference_wav,
                    voice_description=voice_description,
                    emotion=emotion,
                    voice=voice,
                    reference_text=reference_text,
                ),
            )
            yield TTSChunk(
                audio=audio,
                sample_rate=used_backend.sample_rate,
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
