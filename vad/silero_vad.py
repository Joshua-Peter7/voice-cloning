"""
vad/silero_vad.py
──────────────────
Silero VAD — real-time voice activity detection.
Latency target: ~20ms per 32ms chunk.

Key constraints:
  • Chunk size MUST be exactly 512 samples at 16kHz (32ms window).
  • Input must be float32, mono, range [-1, 1].
  • ONNX mode is used for faster CPU inference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from core.interfaces import BaseVAD, VADResult
from core.logger import get_logger
from config.settings import VADConfig

log = get_logger("vad.silero")


class SileroVAD(BaseVAD):
    """
    Silero VAD wrapper.

    Usage:
        vad = SileroVAD(cfg)
        result = vad.process_chunk(chunk)   # chunk: np.ndarray (512,)
        if result.is_speech_end:
            full_utterance = result.utterance
    """

    #: Hard requirement: exactly 512 samples @ 16kHz for VADIterator
    REQUIRED_CHUNK_SAMPLES: int = 512

    def __init__(self, cfg: VADConfig) -> None:
        self.cfg = cfg
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking: bool = False

        log.info("Loading Silero VAD", onnx=cfg.onnx_mode)
        self._model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=cfg.onnx_mode,
        )
        (
            self._get_speech_timestamps,
            self._save_audio,
            self._read_audio,
            self._VADIterator,
            self._collect_chunks,
        ) = utils

        self._iterator = self._make_iterator()
        log.info(
            "Silero VAD ready",
            threshold=cfg.threshold,
            min_silence_ms=cfg.min_silence_ms,
            chunk_ms=self.REQUIRED_CHUNK_SAMPLES / cfg.sample_rate * 1000,
        )

    def _make_iterator(self):
        """Create a fresh VADIterator (called on init and reset)."""
        return self._VADIterator(
            self._model,
            threshold=self.cfg.threshold,
            sampling_rate=self.cfg.sample_rate,
            min_silence_duration_ms=self.cfg.min_silence_ms,
            speech_pad_ms=self.cfg.speech_pad_ms,
        )

    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        """
        Process one 32ms audio chunk.

        Args:
            audio_chunk: float32 numpy array of exactly 512 samples @ 16kHz.

        Returns:
            VADResult with utterance populated when speech segment ends.

        Raises:
            ValueError: if chunk length != REQUIRED_CHUNK_SAMPLES.
        """
        if len(audio_chunk) != self.REQUIRED_CHUNK_SAMPLES:
            raise ValueError(
                f"Chunk must be {self.REQUIRED_CHUNK_SAMPLES} samples, "
                f"got {len(audio_chunk)}"
            )

        chunk_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        speech_dict = self._iterator(chunk_tensor, return_seconds=False)

        if speech_dict is not None:
            if "start" in speech_dict:
                log.debug("Speech start detected")
                self._is_speaking = True
                self._speech_buffer = []

            if self._is_speaking:
                self._speech_buffer.append(audio_chunk.copy())

            if "end" in speech_dict and self._is_speaking:
                log.debug(
                    "Speech end detected",
                    duration_s=round(
                        len(np.concatenate(self._speech_buffer)) / self.cfg.sample_rate, 2
                    ),
                )
                self._is_speaking = False
                utterance = np.concatenate(self._speech_buffer)
                self._speech_buffer = []
                return VADResult(
                    is_speech_end=True,
                    utterance=utterance,
                    speech_probability=1.0,
                )

        elif self._is_speaking:
            # Still inside speech window — keep buffering
            self._speech_buffer.append(audio_chunk.copy())

        return VADResult(
            is_speech_end=False,
            utterance=None,
            speech_probability=0.0,
        )

    def process_audio_file(self, audio_array: np.ndarray) -> list[np.ndarray]:
        """
        Process a full audio file and return all detected utterances.
        Useful for batch processing / testing.
        """
        self.reset()
        utterances = []

        # Pad to exact multiple of chunk size
        pad_len = (self.REQUIRED_CHUNK_SAMPLES - len(audio_array) % self.REQUIRED_CHUNK_SAMPLES) % self.REQUIRED_CHUNK_SAMPLES
        padded = np.pad(audio_array, (0, pad_len))

        for i in range(0, len(padded), self.REQUIRED_CHUNK_SAMPLES):
            chunk = padded[i: i + self.REQUIRED_CHUNK_SAMPLES]
            result = self.process_chunk(chunk)
            if result.is_speech_end and result.utterance is not None:
                utterances.append(result.utterance)

        return utterances

    def reset(self) -> None:
        """Reset for a new session."""
        self._iterator = self._make_iterator()
        self._speech_buffer = []
        self._is_speaking = False
        log.debug("VAD state reset")


class WebRTCVADFallback(BaseVAD):
    """
    WebRTC VAD fallback (CPU-only, no torch dependency).
    Less accurate than Silero but zero GPU memory.
    Use when Silero torch.hub fails in air-gapped environments.
    """

    def __init__(self, cfg: VADConfig, aggressiveness: int = 2) -> None:
        import webrtcvad
        self.cfg = cfg
        self.vad = webrtcvad.Vad(aggressiveness)  # 0–3, higher = more aggressive
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_chunks = 0
        self._silence_threshold = cfg.min_silence_ms // (cfg.chunk_samples * 1000 // cfg.sample_rate)
        log.info("WebRTC VAD fallback initialized", aggressiveness=aggressiveness)

    def process_chunk(self, audio_chunk: np.ndarray) -> VADResult:
        # WebRTC needs int16 PCM bytes
        pcm = (audio_chunk * 32767).astype(np.int16).tobytes()
        # Frame duration must be 10, 20, or 30 ms
        ms = int(len(audio_chunk) / self.cfg.sample_rate * 1000)
        if ms not in (10, 20, 30):
            ms = 30  # default

        is_speech = self.vad.is_speech(pcm, self.cfg.sample_rate)

        if is_speech:
            self._silence_chunks = 0
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_buffer = []
            self._speech_buffer.append(audio_chunk.copy())
        elif self._is_speaking:
            self._silence_chunks += 1
            self._speech_buffer.append(audio_chunk.copy())
            if self._silence_chunks >= self._silence_threshold:
                self._is_speaking = False
                self._silence_chunks = 0
                utterance = np.concatenate(self._speech_buffer)
                self._speech_buffer = []
                return VADResult(is_speech_end=True, utterance=utterance)

        return VADResult(is_speech_end=False, utterance=None)

    def reset(self) -> None:
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_chunks = 0


def create_vad(cfg: VADConfig) -> BaseVAD:
    """
    Factory: create the configured VAD backend.
    Falls back to WebRTC if Silero fails (e.g., no network in air-gapped env).
    """
    if cfg.model == "silero":
        try:
            return SileroVAD(cfg)
        except Exception as exc:
            log.warning(
                "Silero VAD failed to load, falling back to WebRTC VAD",
                error=str(exc),
            )
            return WebRTCVADFallback(cfg)
    elif cfg.model == "webrtc":
        return WebRTCVADFallback(cfg)
    else:
        raise ValueError(f"Unknown VAD model: {cfg.model}")
