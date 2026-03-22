"""
core/profiler.py
────────────────
Pipeline latency profiler and benchmark reporter.
Run the benchmark CLI:  python -m core.profiler
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.interfaces import LatencyProfile
from core.logger import get_logger

log = get_logger("core.profiler")

# South Indian language test phrases
SOUTH_INDIC_TEST_PHRASES = [
    ("ta", "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"),      # Tamil
    ("te", "నమస్కారం, మీరు ఎలా ఉన్నారు?"),                    # Telugu
    ("kn", "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?"),                      # Kannada
    ("ml", "ഹലോ, നിങ്ങൾ എങ്ങനെ ഇരിക്കുന്നു?"),               # Malayalam
    ("hi", "नमस्ते, आप कैसे हैं?"),                            # Hindi
    ("en", "Hello, how can I assist you today?"),               # English
]


class PipelineProfiler:
    """
    Records and analyzes latency profiles across pipeline stages.

    Example usage:
        profiler = PipelineProfiler(sla_ms=1000)
        profiler.add(profile)
        profiler.print_report()
    """

    def __init__(self, sla_ms: int = 1000) -> None:
        self.sla_ms = sla_ms
        self.profiles: List[LatencyProfile] = []

    def add(self, profile: LatencyProfile) -> None:
        self.profiles.append(profile)

    def _stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
        s = sorted(values)
        n = len(s)
        return {
            "mean": round(statistics.mean(values), 1),
            "p50":  round(s[int(n * 0.50)], 1),
            "p95":  round(s[int(n * 0.95)], 1),
            "p99":  round(s[min(int(n * 0.99), n - 1)], 1),
            "max":  round(max(values), 1),
        }

    def report(self) -> Dict:
        if not self.profiles:
            return {"error": "No profiles recorded."}

        return {
            "samples": len(self.profiles),
            "vad":        self._stats([p.vad_ms for p in self.profiles]),
            "asr":        self._stats([p.asr_ms for p in self.profiles]),
            "slm_ttft":   self._stats([p.slm_ttft_ms for p in self.profiles]),
            "tts_first":  self._stats([p.tts_first_chunk_ms for p in self.profiles]),
            "total_ttfb": self._stats([p.total_ttfb_ms for p in self.profiles]),
            "by_language": self._by_language(),
            "sla_pass_rate": round(
                sum(1 for p in self.profiles if p.total_ttfb_ms < self.sla_ms)
                / len(self.profiles) * 100, 1
            ),
        }

    def _by_language(self) -> Dict:
        lang_groups: Dict[str, List[float]] = {}
        for p in self.profiles:
            lang_groups.setdefault(p.language, []).append(p.total_ttfb_ms)
        return {
            lang: self._stats(vals)
            for lang, vals in lang_groups.items()
        }

    def print_report(self) -> None:
        r = self.report()
        if "error" in r:
            print(r["error"])
            return

        W = 65
        print()
        print("═" * W)
        print("  INDIC VOICE CLONER — LATENCY BENCHMARK REPORT")
        print("═" * W)
        print(f"  Samples: {r['samples']}  |  SLA threshold: {self.sla_ms}ms")
        print()

        headers = ["Stage", "Mean", "p50", "p95", "p99", "Max"]
        stage_map = [
            ("vad",        "VAD"),
            ("asr",        "ASR"),
            ("slm_ttft",   "SLM TTFT"),
            ("tts_first",  "TTS 1st Chunk"),
            ("total_ttfb", "Total TTFB"),
        ]
        sla_thresholds = {"vad": 50, "asr": 200, "slm_ttft": 120, "tts_first": 300, "total_ttfb": self.sla_ms}

        print(f"  {'Stage':<18} {'Mean':>7}  {'p50':>7}  {'p95':>7}  {'p99':>7}  {'Max':>7}")
        print("  " + "─" * (W - 2))
        for key, label in stage_map:
            s = r[key]
            ok = "✅" if s["p95"] <= sla_thresholds[key] else "⚠️ "
            print(
                f"  {ok} {label:<16} "
                f"{s['mean']:>6.1f}ms  {s['p50']:>6.1f}ms  "
                f"{s['p95']:>6.1f}ms  {s['p99']:>6.1f}ms  {s['max']:>6.1f}ms"
            )

        print()
        print(f"  SLA (TTFB < {self.sla_ms}ms) Pass Rate: {r['sla_pass_rate']}%")

        print()
        print("  By Language:")
        for lang, stats in r["by_language"].items():
            ok = "✅" if stats["p95"] <= self.sla_ms else "⚠️ "
            print(f"    {ok} {lang.upper():<4}  p95={stats['p95']:>6.1f}ms  mean={stats['mean']:>6.1f}ms")

        print("═" * W)


def simulate_benchmark(n: int = 50, sla_ms: int = 1000) -> PipelineProfiler:
    """
    Simulate pipeline runs with realistic latency distributions.
    Use this to validate reporting before connecting real components.
    """
    random.seed(42)
    profiler = PipelineProfiler(sla_ms=sla_ms)

    for _ in range(n):
        lang = random.choice(["ta", "te", "kn", "ml", "hi", "en"])
        vad = random.gauss(22, 4)
        asr = random.gauss(145, 25)
        slm = random.gauss(85, 15)
        tts = random.gauss(210, 30)

        profiler.add(LatencyProfile(
            vad_ms=max(10, vad),
            asr_ms=max(80, asr),
            slm_ttft_ms=max(40, slm),
            tts_first_chunk_ms=max(120, tts),
            total_ttfb_ms=max(10, vad) + max(80, asr) + max(40, slm) + max(120, tts) + 50,
            language=lang,
        ))

    return profiler


if __name__ == "__main__":
    profiler = simulate_benchmark(n=100)
    profiler.print_report()
