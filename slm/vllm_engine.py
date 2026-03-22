"""
slm/vllm_engine.py
──────────────────
SLM inference via vLLM with AWQ 4-bit quantization.
Latency target: <80ms time-to-first-token (TTFT).

The system prompt is multilingual-aware:
  • Responds in the SAME language as the user query
  • Plain text only (no markdown — TTS can't render it)
  • Concise conversational responses (1–3 sentences)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

from core.interfaces import BaseSLM, SLMToken
from core.logger import get_logger
from config.settings import SLMConfig

log = get_logger("slm.vllm")

# ── Multilingual system prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a helpful multilingual voice assistant supporting Tamil, Telugu, Kannada, \
Malayalam, Hindi, Bengali, Marathi, Gujarati, Punjabi, and English.

CRITICAL RULES:
1. Always respond in the EXACT SAME language as the user's query.
2. Keep responses SHORT and CONVERSATIONAL — 1 to 3 sentences maximum.
3. Output PLAIN TEXT ONLY — no markdown, no bullet points, no asterisks, \
no special formatting of any kind. The response goes directly to speech synthesis.
4. If you cannot understand the language, respond in English.
5. Be warm, helpful, and natural-sounding when spoken aloud."""


def _build_messages(
    user_text: str,
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Build the chat message list for vLLM."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


# ── vLLM Engine ───────────────────────────────────────────────────────────────

class VLLMSLMEngine(BaseSLM):
    """
    SLM inference using vLLM's AsyncLLMEngine.
    Supports AWQ / GPTQ / bitsandbytes quantization.

    The engine is initialized lazily on first use to avoid
    holding GPU memory before the pipeline is ready.
    """

    def __init__(self, cfg: SLMConfig) -> None:
        self.cfg = cfg
        self._engine = None
        self._tokenizer = None
        log.info("VLLMSLMEngine created (lazy init)")

    def _ensure_loaded(self) -> None:
        """Initialize vLLM engine on first use."""
        if self._engine is not None:
            return

        from vllm import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs
        from transformers import AutoTokenizer

        model_path = (
            self.cfg.awq_save_path
            if self.cfg.quantization == "awq"
            else self.cfg.model_id
        )

        log.info(
            "Initializing vLLM engine",
            model=model_path,
            quantization=self.cfg.quantization,
        )

        engine_args = AsyncEngineArgs(
            model=model_path,
            quantization=self.cfg.quantization if self.cfg.quantization != "none" else None,
            tensor_parallel_size=self.cfg.tensor_parallel,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            max_model_len=4096,
            trust_remote_code=True,
            disable_log_requests=True,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id, trust_remote_code=True
        )
        log.info("vLLM engine ready", model=model_path)

    async def generate_stream(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        language: str = "auto",
    ) -> AsyncGenerator[SLMToken, None]:
        """
        Stream tokens from the SLM.
        Each yielded SLMToken carries the new text delta.
        """
        self._ensure_loaded()

        from vllm import SamplingParams

        messages = _build_messages(user_text, history)
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        params = SamplingParams(
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            stop=["</s>", "<|end|>", "<|eot_id|>"],
        )

        request_id = str(uuid.uuid4())
        t0 = time.perf_counter()
        first_token = True

        async for output in self._engine.generate(prompt, params, request_id):
            if output.outputs:
                delta = output.outputs[0].text
                if delta:
                    if first_token:
                        ttft = round((time.perf_counter() - t0) * 1000, 2)
                        log.debug("SLM first token", ttft_ms=ttft)
                        first_token = False
                    is_final = output.finished
                    yield SLMToken(text=delta, is_final=is_final)

    async def generate(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        language: str = "auto",
    ) -> str:
        """Generate a complete response (non-streaming convenience method)."""
        parts = []
        async for token in self.generate_stream(user_text, history, language):
            parts.append(token.text)
        return "".join(parts)

    def quantize_model(self) -> None:
        """
        One-time AWQ quantization of the base model.
        Run this ONCE before deploying to generate the quantized model.

        Example:
            engine = VLLMSLMEngine(cfg)
            engine.quantize_model()
            # Then set cfg.model_id = cfg.awq_save_path
        """
        import os
        if os.path.exists(self.cfg.awq_save_path):
            log.info("AWQ model already exists", path=self.cfg.awq_save_path)
            return

        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer

        log.info(
            "Starting AWQ quantization (10–30 min)",
            model=self.cfg.model_id,
            save_to=self.cfg.awq_save_path,
        )
        model = AutoAWQForCausalLM.from_pretrained(
            self.cfg.model_id, low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_id)
        model.quantize(tokenizer, quant_config=self.cfg.awq_quant_config.__dict__)
        model.save_quantized(self.cfg.awq_save_path)
        tokenizer.save_pretrained(self.cfg.awq_save_path)
        log.info("AWQ quantization complete", path=self.cfg.awq_save_path)


# ── Fallback: HuggingFace transformers (no vLLM) ─────────────────────────────

class HFSLMEngine(BaseSLM):
    """
    Fallback SLM using HuggingFace transformers + bitsandbytes.
    Use when vLLM is not available (e.g., CPU-only machines).
    Higher latency, but same interface.
    """

    def __init__(self, cfg: SLMConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        log.info("HFSLMEngine created (lazy init)")

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        log.info("Loading HF SLM", model=self.cfg.model_id)

        bnb_config = None
        if self.cfg.quantization == "bitsandbytes":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )
        log.info("HF SLM ready")

    async def generate_stream(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        language: str = "auto",
    ) -> AsyncGenerator[SLMToken, None]:
        """Stream tokens using HF TextIteratorStreamer."""
        self._ensure_loaded()
        from transformers import TextIteratorStreamer
        import threading

        messages = _build_messages(user_text, history)
        input_ids = self._tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            do_sample=True,
            streamer=streamer,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        loop = asyncio.get_event_loop()
        for text in streamer:
            if text:
                yield SLMToken(text=text)
            await asyncio.sleep(0)  # yield control to event loop

        thread.join()
        yield SLMToken(text="", is_final=True)

    async def generate(
        self,
        user_text: str,
        history: List[Dict[str, str]],
        language: str = "auto",
    ) -> str:
        parts = []
        async for token in self.generate_stream(user_text, history, language):
            parts.append(token.text)
        return "".join(parts)


def create_slm(cfg: SLMConfig) -> BaseSLM:
    """
    Factory: create SLM engine.
    Falls back to HF engine if vLLM is not installed.
    """
    try:
        import vllm  # noqa: F401
        return VLLMSLMEngine(cfg)
    except ImportError:
        log.warning("vLLM not available, falling back to HF transformers engine")
        return HFSLMEngine(cfg)
