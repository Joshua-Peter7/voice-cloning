"""
tests/integration/test_e2e.py
──────────────────────────────
End-to-end pipeline integration test.
Uses mock components but tests the full orchestration flow.

For real component tests, set:
  IVC_ASR__MODEL=large-v3
  IVC_TTS__BACKEND=parler
And run with:  pytest tests/integration/ -v --real-models
"""

from __future__ import annotations

import asyncio
import time
import numpy as np
import pytest

from core.interfaces import LatencyProfile
from core.profiler import simulate_benchmark, SOUTH_INDIC_TEST_PHRASES


@pytest.fixture
def south_indic_audio():
    """Synthetic audio for each South Indian language."""
    return {
        lang: np.random.randn(16000 * 3).astype(np.float32) * 0.05
        for lang, _ in SOUTH_INDIC_TEST_PHRASES
    }


class TestE2ESouthIndic:
    @pytest.mark.asyncio
    async def test_all_south_indic_languages(self):
        """
        Verify the benchmark reports SLA pass for all 4 South Indian languages.
        """
        profiler = simulate_benchmark(n=30)
        report = profiler.report()

        by_lang = report["by_language"]
        south_indic = ["ta", "te", "kn", "ml"]

        for lang in south_indic:
            assert lang in by_lang, f"No profiles for {lang}"
            p95 = by_lang[lang]["p95"]
            assert p95 < 1000, f"{lang} p95 TTFB {p95}ms exceeds SLA"

    def test_benchmark_all_languages(self):
        """Print a full benchmark report for visual inspection."""
        profiler = simulate_benchmark(n=100)
        profiler.print_report()
        report = profiler.report()
        assert report["sla_pass_rate"] >= 90.0, "SLA pass rate below 90%"

    def test_latency_budget_distribution(self):
        """
        Verify target latency budget is met on average:
          VAD: ~20ms, ASR: ~150ms, SLM: ~80ms, TTS: ~200ms → Total ~600ms
        """
        profiler = simulate_benchmark(n=100)
        report = profiler.report()

        assert report["vad"]["mean"] < 50, "VAD mean too high"
        assert report["asr"]["mean"] < 250, "ASR mean too high"
        assert report["slm_ttft"]["mean"] < 150, "SLM TTFT mean too high"
        assert report["tts_first"]["mean"] < 300, "TTS first chunk mean too high"
        assert report["total_ttfb"]["mean"] < 800, "Total TTFB mean too high"
