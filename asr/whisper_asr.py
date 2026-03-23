"""
asr/whisper_asr.py
──────────────────
faster-whisper ASR — optimized for South Indian languages.
Latency target: <150ms for a 5-second utterance on GPU.

South Indian language accuracy notes:
  • Tamil (ta): whisper large-v3 works well; use language="ta" explicitly for best WER
  • Telugu (te): similarly well supported in large-v3
  • Kannada (kn): moderate accuracy; large-v3 recommended over smaller models
  • Malayalam (ml): moderate accuracy; large-v3 recommended
  • Auto-detect works but adds ~10ms; pass explicit lang when known from VAD context
"""

from __future__ import annotations

import time
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np

from core.interfaces import ASRResult, BaseASR
from core.device import resolve_device
from core.logger import get_logger
from config.settings import ASRConfig

log = get_logger("asr.whisper")

# Human-readable language names for all supported Indic langs
LANG_NAMES: Dict[str, str] = {
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "hi": "Hindi",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "en": "English",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese",
}

class FasterWhisperASR(BaseASR):
    """
    Faster-Whisper ASR using CTranslate2 backend.
    4–8x faster than vanilla Whisper with same accuracy.

    Usage:
        asr = FasterWhisperASR(cfg)
        result = asr.transcribe(audio_array, language="ta")
    """

    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        device = resolve_device(cfg.device)
        compute = cfg.compute_type if device != "cpu" else "int8"

        log.info(
            "Loading Whisper ASR",
            model=cfg.model,
            device=device,
            compute=compute,
        )
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            cfg.model.replace("openai/", ""),
            device=device,
            compute_type=compute,
            download_root=cfg.download_root,
        )
        self._device = device
        log.info("Whisper ASR ready", model=cfg.model)

    def transcribe(
        self,
        audio_array: np.ndarray,
        language: str = "auto",
    ) -> ASRResult:
        """
        Transcribe a complete audio utterance.

        Args:
            audio_array: float32, 16kHz mono numpy array.
            language: ISO 639-1 code or "auto" for detection.

        Returns:
            ASRResult with text, language, confidence, latency.
        """
        t0 = time.perf_counter()

        # Explicit language for South Indian langs improves accuracy
        lang: Optional[str] = None
        if language != "auto":
            lang = language
        elif audio_array.shape[0] < 16000 * 2:
            # Short clips: auto-detect is unreliable; don't specify
            lang = None

        segments, info = self._model.transcribe(
            audio_array,
            language=lang,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
            word_timestamps=self.cfg.word_timestamps,
            condition_on_previous_text=self.cfg.condition_on_previous,
            # Suppress hallucination tokens common in silence
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
        )

        text_parts: List[str] = []
        word_ts: List[Dict] = []

        for seg in segments:
            text_parts.append(seg.text.strip())
            if self.cfg.word_timestamps and seg.words:
                for w in seg.words:
                    word_ts.append({
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    })

        latency_ms = (time.perf_counter() - t0) * 1000
        detected_lang = info.language

        result = ASRResult(
            text=" ".join(text_parts),
            language=detected_lang,
            language_name=LANG_NAMES.get(detected_lang, detected_lang),
            confidence=round(info.language_probability, 3),
            latency_ms=round(latency_ms, 2),
            word_timestamps=word_ts if word_ts else None,
        )

        log.debug(
            "ASR result",
            lang=detected_lang,
            text_preview=result.text[:60],
            latency_ms=round(latency_ms, 1),
            confidence=result.confidence,
        )
        return result

    def transcribe_stream(
        self,
        audio_array: np.ndarray,
    ) -> AsyncGenerator[ASRResult, None]:
        """
        Yield partial transcription results as segments complete.
        Useful for long audio where partial results improve UX.
        """
        segments, info = self._model.transcribe(
            audio_array,
            beam_size=1,
            word_timestamps=True,
        )
        for seg in segments:
            yield ASRResult(
                text=seg.text.strip(),
                language=info.language,
                language_name=LANG_NAMES.get(info.language, info.language),
                confidence=info.language_probability,
                latency_ms=0.0,
            )

    def transcribe_file(self, wav_path: str, language: str = "auto") -> ASRResult:
        """Convenience method to transcribe a wav file directly."""
        import soundfile as sf
        try:
            audio, sr = sf.read(wav_path, dtype="float32")
        except Exception:
            import librosa
            audio, sr = librosa.load(wav_path, sr=None, mono=False)
        if audio.ndim > 1 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # stereo → mono
        return self.transcribe(audio, language=language)


def create_asr(cfg: ASRConfig) -> BaseASR:
    """Factory: create the configured ASR backend."""
    return FasterWhisperASR(cfg)
