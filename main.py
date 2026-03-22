#!/usr/bin/env python3
"""
main.py
───────
Application entrypoint.

Usage:
  python main.py                          # Start API server (default)
  python main.py serve                    # Start API server
  python main.py benchmark                # Run latency benchmark
  python main.py quantize                 # AWQ-quantize the SLM
  python main.py tts --text "வணக்கம்" --lang ta   # Quick TTS test
"""

from __future__ import annotations

import argparse
import sys


def serve(args):
    import sys
    import uvicorn
    from config.settings import get_settings
    from core.logger import setup_logger

    cfg = get_settings()
    setup_logger(level=cfg.app.log_level, env=cfg.app.env, log_dir=cfg.infra.log_dir)

    # uvloop is not available on Windows; fall back to default asyncio loop
    loop_policy = "auto"
    if sys.platform != "win32":
        try:
            import uvloop  # noqa: F401
            loop_policy = "uvloop"
        except ImportError:
            pass

    uvicorn.run(
        "api.app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        workers=cfg.api.workers,
        loop=loop_policy,
        log_level=cfg.app.log_level.lower(),
        reload=cfg.app.env == "development",
    )


def benchmark(args):
    from core.profiler import simulate_benchmark
    n = getattr(args, "n", 100)
    print(f"Running simulated benchmark with {n} samples...")
    profiler = simulate_benchmark(n=n)
    profiler.print_report()


def quantize(args):
    from config.settings import get_settings
    from slm.vllm_engine import VLLMSLMEngine

    cfg = get_settings()
    engine = VLLMSLMEngine(cfg.slm)
    engine.quantize_model()
    print(f"✅ Quantized model saved to: {cfg.slm.awq_save_path}")


def tts_test(args):
    """Quick smoke test: synthesize one sentence and save to output.wav."""
    import asyncio
    import soundfile as sf
    import numpy as np
    from config.settings import get_settings
    from tts.engine import create_tts_engine

    cfg = get_settings()
    engine = create_tts_engine(cfg.tts)

    text = args.text
    lang = args.lang

    async def _run():
        async def _token_gen():
            for word in text.split(" "):
                yield word + " "

        chunks = []
        async for chunk in engine.stream_from_tokens(_token_gen(), language=lang):
            chunks.append(chunk.audio)
            print(f"  Chunk {chunk.chunk_index}: {chunk.text_segment!r} — {chunk.latency_ms:.0f}ms")

        combined = np.concatenate(chunks)
        sf.write("output.wav", combined, samplerate=cfg.tts.sample_rate)
        print(f"\n✅ Saved to output.wav ({len(combined)/cfg.tts.sample_rate:.2f}s)")

    asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(
        description="Indic Voice Cloner — Enterprise Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve
    serve_p = subparsers.add_parser("serve", help="Start the API server")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run latency benchmark")
    bench_p.add_argument("--n", type=int, default=100, help="Number of simulated samples")

    # quantize
    quant_p = subparsers.add_parser("quantize", help="AWQ-quantize the SLM")

    # tts
    tts_p = subparsers.add_parser("tts", help="Quick TTS smoke test")
    tts_p.add_argument("--text", required=True, help="Text to synthesize")
    tts_p.add_argument("--lang", default="ta", help="Language code (ta/te/kn/ml/hi/en)")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        serve(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "quantize":
        quantize(args)
    elif args.command == "tts":
        tts_test(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
