"""
tts/svara_backend.py
--------------------
Svara TTS backend using Kenpath's Hugging Face voice-clone model.

Key behavior:
  1. Loads `kenpath/svara-tts-voiceclone-beta` (causal LM) + SNAC codec.
  2. If `reference_wav` is provided, encodes it into SNAC reference tokens.
  3. Prompts the model with reference tokens + target text.
  4. Decodes generated SNAC tokens back to waveform.

This backend is optimized for Indic languages (including Tamil) and
supports low-latency zero-shot cloning from short clean references.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

from core.interfaces import BaseTTSBackend
from core.device import resolve_device
from core.logger import get_logger

log = get_logger("tts.svara")

# Hugging Face model IDs
SVARA_CLONE_MODEL = "kenpath/svara-tts-voiceclone-beta"
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"

# Speaker fallback map (used only when no reference audio is provided)
LANG_TO_VOICE = {
    "ta": "Tamil (Female)",
    "te": "Telugu (Female)",
    "kn": "Kannada (Female)",
    "ml": "Malayalam (Female)",
    "hi": "Hindi (Female)",
    "bn": "Bengali (Female)",
    "mr": "Marathi (Female)",
    "gu": "Gujarati (Female)",
    "pa": "Punjabi (Female)",
    "en": "English (Female)",
    "sa": "Sanskrit (Female)",
}
DEFAULT_VOICE = "Tamil (Female)"

# Model vocabulary / special tokens (from Kenpath's inference constants)
_BOS_TOKEN = 128000
_SOA_TOKEN = 128257  # start of audio
_EOA_TOKEN = 128258  # end of audio
_AUDIO_VOCAB_SIZE = 4096
_AUDIO_TOKEN_OFFSETS = [128266, 132362, 136458, 140554, 144650, 148746, 152842]
_AUDIO_TOKEN_MIN = _AUDIO_TOKEN_OFFSETS[0]
_AUDIO_TOKEN_MAX = _AUDIO_TOKEN_OFFSETS[-1] + _AUDIO_VOCAB_SIZE


class SvaraTTSBackend(BaseTTSBackend):
    """
    Svara voice cloning backend (Hugging Face).

    Notes:
      - Best quality with 6-15s clean single-speaker reference audio.
      - For repeated real-time turns with the same reference file, encoded
        reference tokens are cached for lower latency.
    """

    _SUPPORTED: List[str] = list(LANG_TO_VOICE.keys())

    def __init__(
        self,
        model_id: str = SVARA_CLONE_MODEL,
        device: str = "auto",
        min_reference_seconds: float = 3.0,
        max_reference_seconds: float = 15.0,
        reference_target_seconds: float = 10.0,
        max_new_tokens: int = 1200,
        temperature: float = 0.55,
        top_p: float = 0.90,
    ) -> None:
        self.model_id = model_id
        self._device_str = resolve_device(device)
        self.min_reference_seconds = min_reference_seconds
        self.max_reference_seconds = max_reference_seconds
        self.reference_target_seconds = reference_target_seconds
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self._model = None
        self._tokenizer = None
        self._snac = None
        self._sample_rate_val = 24000

        # Cache keyed by (abs_path, mtime) so updates invalidate correctly.
        self._reference_token_cache: Dict[Tuple[str, float], List[int]] = {}
        log.info("Svara backend created (lazy load)", model=model_id, device=self._device_str)

    # -- Loading -----------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from snac import SNAC

        log.info("Loading Svara model", model=self.model_id, device=self._device_str)
        t0 = time.perf_counter()

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        load_dtype = torch.float16 if self._device_str != "cpu" else torch.float32
        load_kwargs = {
            "torch_dtype": load_dtype,
            "low_cpu_mem_usage": True,
            "offload_state_dict": True,
            "device_map": {"": self._device_str},
        }
        try:
            # device_map + low_cpu_mem_usage avoids CPU->GPU duplicate copies.
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs,
            )
        except (TypeError, ValueError) as exc:
            # Older transformers/accelerate combos may not support these flags.
            log.warning("Memory-efficient load path unavailable; using standard load", error=str(exc))
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=load_dtype,
                ).to(self._device_str)
            except OSError as nested_exc:
                msg = str(nested_exc).lower()
                if "paging file is too small" in msg or "os error 1455" in msg:
                    raise RuntimeError(
                        "Svara model could not load because Windows virtual memory/pagefile "
                        "is too small. Increase pagefile size (recommended 32GB+), then retry."
                    ) from nested_exc
                raise
        except OSError as exc:
            msg = str(exc).lower()
            if "paging file is too small" in msg or "os error 1455" in msg:
                raise RuntimeError(
                    "Svara model could not load because Windows virtual memory/pagefile "
                    "is too small. Increase pagefile size (recommended 32GB+), then retry."
                ) from exc
            raise
        self._model.eval()

        log.info("Loading SNAC codec", model=SNAC_MODEL_ID)
        self._snac = SNAC.from_pretrained(SNAC_MODEL_ID).to(self._device_str)
        self._snac.eval()

        elapsed = round(time.perf_counter() - t0, 1)
        log.info("Svara ready", elapsed_s=elapsed)

    # -- Reference audio ----------------------------------------------------------

    def _load_reference_audio(self, reference_wav: str) -> np.ndarray:
        if not os.path.exists(reference_wav):
            raise FileNotFoundError(f"Reference wav not found: {reference_wav}")

        try:
            audio, sr = sf.read(reference_wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            # Fallback for files like mp3 where libsndfile support may be limited.
            log.warning("soundfile read failed; falling back to librosa loader", path=reference_wav)
            audio, sr = librosa.load(reference_wav, sr=None, mono=True)
            audio = np.asarray(audio, dtype=np.float32)

        if sr != self._sample_rate_val:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self._sample_rate_val)

        duration_s = len(audio) / self._sample_rate_val
        if duration_s < self.min_reference_seconds:
            raise ValueError(
                f"Reference audio is too short ({duration_s:.2f}s). "
                f"Use at least {self.min_reference_seconds:.1f}s."
            )

        if duration_s > self.max_reference_seconds:
            target_s = min(self.reference_target_seconds, self.max_reference_seconds)
            target_samples = int(target_s * self._sample_rate_val)
            audio = audio[:target_samples]

        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1.0:
            audio = audio / peak

        return np.ascontiguousarray(audio, dtype=np.float32)

    def _encode_reference_wav(self, reference_wav: str) -> List[int]:
        self._ensure_loaded()

        abs_path = os.path.abspath(reference_wav)
        mtime = os.path.getmtime(abs_path)
        cache_key = (abs_path, mtime)
        if cache_key in self._reference_token_cache:
            return self._reference_token_cache[cache_key]

        wav = self._load_reference_audio(abs_path)
        wav_tensor = torch.from_numpy(wav).to(self._device_str, dtype=torch.float32)
        wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)

        with torch.inference_mode():
            codes = self._snac.encode(wav_tensor)

        num_frames = int(codes[0].shape[1])
        if num_frames <= 0:
            raise ValueError("Failed to encode reference audio into SNAC tokens.")

        tokens: List[int] = []
        for i in range(num_frames):
            c0 = int(codes[0][0][i].item())
            c1 = int(codes[1][0][2 * i].item())
            c2 = int(codes[2][0][4 * i].item())
            c3 = int(codes[2][0][4 * i + 1].item())
            c4 = int(codes[1][0][2 * i + 1].item())
            c5 = int(codes[2][0][4 * i + 2].item())
            c6 = int(codes[2][0][4 * i + 3].item())

            frame = [c0, c1, c2, c3, c4, c5, c6]
            for level, code in enumerate(frame):
                tokens.append(code + _AUDIO_TOKEN_OFFSETS[level])

        self._reference_token_cache[cache_key] = tokens
        log.info("Encoded reference audio", frames=num_frames, tokens=len(tokens))
        return tokens

    # -- Prompting ----------------------------------------------------------------

    def _build_prompt(
        self,
        text: str,
        language: str = "en",
        emotion: str = "neutral",
        voice: Optional[str] = None,
    ) -> str:
        speaker = voice or LANG_TO_VOICE.get(language, DEFAULT_VOICE)
        return f"{speaker}\n{text.strip()} <{emotion}>"

    def _build_input_ids(
        self,
        text: str,
        language: str,
        emotion: str,
        voice: Optional[str],
        reference_wav: Optional[str],
    ) -> torch.Tensor:
        if reference_wav:
            ref_audio_tokens = self._encode_reference_wav(reference_wav)
            text_with_style = f"{text.strip()} <{emotion}>"
            text_ids = self._tokenizer(text_with_style, add_special_tokens=False).input_ids
            prompt_ids = [_BOS_TOKEN, *ref_audio_tokens, *text_ids, _SOA_TOKEN]
            return torch.tensor([prompt_ids], dtype=torch.long, device=self._device_str)

        prompt = self._build_prompt(text, language=language, emotion=emotion, voice=voice)
        return self._tokenizer(prompt, return_tensors="pt").input_ids.to(self._device_str)

    # -- Audio decode -------------------------------------------------------------

    @staticmethod
    def _token_matches_level(token_id: int, level: int) -> bool:
        start = _AUDIO_TOKEN_OFFSETS[level]
        end = start + _AUDIO_VOCAB_SIZE
        return start <= token_id < end

    def _decode_audio_tokens(self, token_ids: List[int]) -> np.ndarray:
        if not token_ids:
            return np.zeros(self._sample_rate_val // 2, dtype=np.float32)

        layer1: List[int] = []
        layer2: List[int] = []
        layer3: List[int] = []

        frame_tokens: List[int] = []
        expected_level = 0

        for token_id in token_ids:
            if token_id == _EOA_TOKEN:
                break
            if token_id < _AUDIO_TOKEN_MIN or token_id >= _AUDIO_TOKEN_MAX:
                continue

            if self._token_matches_level(token_id, expected_level):
                frame_tokens.append(token_id)
                expected_level += 1

                if expected_level == 7:
                    decoded: List[int] = []
                    valid_frame = True
                    for level, tok in enumerate(frame_tokens):
                        value = tok - _AUDIO_TOKEN_OFFSETS[level]
                        if value < 0 or value >= _AUDIO_VOCAB_SIZE:
                            valid_frame = False
                            break
                        decoded.append(value)

                    if valid_frame:
                        layer1.append(decoded[0])
                        layer2.extend([decoded[1], decoded[4]])
                        layer3.extend([decoded[2], decoded[3], decoded[5], decoded[6]])

                    frame_tokens = []
                    expected_level = 0
            else:
                # Attempt to realign on the start of a new frame.
                if self._token_matches_level(token_id, 0):
                    frame_tokens = [token_id]
                    expected_level = 1
                else:
                    frame_tokens = []
                    expected_level = 0

        if not layer1:
            log.warning("No valid audio frames decoded from model output")
            return np.zeros(self._sample_rate_val // 2, dtype=np.float32)

        with torch.inference_mode():
            t1 = torch.tensor(layer1, dtype=torch.long, device=self._device_str).unsqueeze(0)
            t2 = torch.tensor(layer2, dtype=torch.long, device=self._device_str).unsqueeze(0)
            t3 = torch.tensor(layer3, dtype=torch.long, device=self._device_str).unsqueeze(0)
            decoded = self._snac.decode([t1, t2, t3])

        wav = decoded.squeeze().detach().cpu().float().numpy()
        return wav.astype(np.float32)

    # -- Synthesis ----------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        language: str = "en",
        reference_wav: Optional[str] = None,
        voice: Optional[str] = None,
        emotion: str = "neutral",
        **kwargs,
    ) -> np.ndarray:
        self._ensure_loaded()

        input_ids = self._build_input_ids(
            text=text,
            language=language,
            emotion=emotion,
            voice=voice,
            reference_wav=reference_wav,
        )
        attention_mask = torch.ones_like(input_ids, device=self._device_str)

        max_new_tokens = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        top_p = float(kwargs.get("top_p", self.top_p))

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1,
                eos_token_id=_EOA_TOKEN,
                pad_token_id=self._tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
        log.debug("Svara generated tokens", count=len(generated_ids))
        waveform = self._decode_audio_tokens(generated_ids)
        log.info(
            "Svara synthesis done",
            text_len=len(text),
            samples=len(waveform),
            used_reference=bool(reference_wav),
        )
        return waveform

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_val

    @property
    def supported_languages(self) -> List[str]:
        return self._SUPPORTED
