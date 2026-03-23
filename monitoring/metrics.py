"""
monitoring/metrics.py
─────────────────────
Prometheus metrics for the voice cloning pipeline.
Metrics exposed at /metrics (or a dedicated port).

Key SLAs tracked:
  • TTFB (Time to First Audio Byte) < 1000ms
  • ASR latency < 200ms
  • SLM TTFT < 100ms
  • TTS first chunk < 250ms
"""

from __future__ import annotations

import threading
import time

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    start_http_server,
    CONTENT_TYPE_LATEST,
)

from core.interfaces import BaseMonitor, LatencyProfile
from core.logger import get_logger
from config.settings import MonitoringConfig

log = get_logger("monitoring.metrics")


class PrometheusMonitor(BaseMonitor):
    """
    Prometheus metrics server.

    Grafana dashboard recommended panels:
      - TTFB p50/p95/p99 time series
      - SLA violation rate (TTFB > 1000ms)
      - Requests per second by language
      - Active WebSocket sessions
      - GPU VRAM utilization
      - Error rate by stage
    """

    def __init__(self, cfg: MonitoringConfig) -> None:
        self.cfg = cfg
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self._start_server(cfg.prometheus_port)
        if cfg.enable_gpu_metrics:
            self._start_gpu_poller(cfg.gpu_poll_interval_s)

    def _setup_metrics(self) -> None:
        r = self.registry

        # ── Latency Histograms ─────────────────────────────────────────
        self.ttfb = Histogram(
            "ivc_ttfb_ms",
            "Time to first audio byte (ms)",
            buckets=[100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000],
            registry=r,
        )
        self.asr_latency = Histogram(
            "ivc_asr_latency_ms",
            "ASR inference latency (ms)",
            buckets=[50, 100, 150, 200, 300, 500],
            registry=r,
        )
        self.slm_ttft = Histogram(
            "ivc_slm_ttft_ms",
            "SLM time to first token (ms)",
            buckets=[20, 40, 60, 80, 100, 150, 200],
            registry=r,
        )
        self.tts_first_chunk = Histogram(
            "ivc_tts_first_chunk_ms",
            "TTS first chunk latency (ms)",
            buckets=[100, 150, 200, 250, 300, 400, 500],
            registry=r,
        )

        # ── Counters ──────────────────────────────────────────────────
        self.requests_total = Counter(
            "ivc_requests_total",
            "Total voice requests processed",
            ["language", "status"],
            registry=r,
        )
        self.sla_violations = Counter(
            "ivc_sla_violations_total",
            f"TTFB > {self.cfg.sla_ttfb_ms}ms SLA violations",
            registry=r,
        )
        self.errors = Counter(
            "ivc_errors_total",
            "Pipeline errors by stage",
            ["stage"],
            registry=r,
        )
        self.ws_connections_total = Counter(
            "ivc_websocket_connections_total",
            "Total WebSocket connections opened",
            registry=r,
        )

        # ── Gauges ────────────────────────────────────────────────────
        self.active_sessions = Gauge(
            "ivc_active_sessions",
            "Current active WebSocket sessions",
            registry=r,
        )
        self.gpu_vram_used_gb = Gauge(
            "ivc_gpu_vram_used_gb",
            "GPU VRAM currently allocated (GB)",
            registry=r,
        )
        self.gpu_vram_total_gb = Gauge(
            "ivc_gpu_vram_total_gb",
            "GPU total VRAM (GB)",
            registry=r,
        )

    def _start_server(self, port: int) -> None:
        try:
            start_http_server(port, registry=self.registry)
            log.info("Prometheus metrics server started", port=port)
        except OSError as exc:
            log.warning("Metrics server port in use", port=port, error=str(exc))

    def _start_gpu_poller(self, interval: int) -> None:
        """Background thread to poll GPU stats."""
        def _poll():
            try:
                import torch
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    self.gpu_vram_total_gb.set(props.total_memory / 1e9)
            except Exception:
                pass

            while True:
                try:
                    import torch
                    if torch.cuda.is_available():
                        used = torch.cuda.memory_allocated(0) / 1e9
                        self.gpu_vram_used_gb.set(round(used, 3))
                except Exception:
                    pass
                time.sleep(interval)

        t = threading.Thread(target=_poll, daemon=True, name="gpu-metrics-poller")
        t.start()

    # ── BaseMonitor interface ─────────────────────────────────────────

    def record_request(self, profile: LatencyProfile, status: str = "success") -> None:
        self.ttfb.observe(profile.total_ttfb_ms)
        self.asr_latency.observe(profile.asr_ms)
        self.slm_ttft.observe(profile.slm_ttft_ms)
        self.tts_first_chunk.observe(profile.tts_first_chunk_ms)
        self.requests_total.labels(
            language=profile.language, status=status
        ).inc()
        if profile.total_ttfb_ms > self.cfg.sla_ttfb_ms:
            self.sla_violations.inc()
            log.warning(
                "SLA violation: TTFB exceeded",
                ttfb_ms=profile.total_ttfb_ms,
                threshold_ms=self.cfg.sla_ttfb_ms,
                session_id=profile.session_id,
            )

    def increment_error(self, stage: str) -> None:
        self.errors.labels(stage=stage).inc()

    def set_active_sessions(self, count: int) -> None:
        self.active_sessions.set(count)

    def get_metrics_text(self) -> bytes:
        """Return Prometheus text format metrics (for /metrics endpoint)."""
        return generate_latest(self.registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST


class NullMonitor(BaseMonitor):
    """No-op monitor for testing or when monitoring is disabled."""

    def record_request(self, profile: LatencyProfile, status: str = "success") -> None:
        pass

    def increment_error(self, stage: str) -> None:
        pass

    def set_active_sessions(self, count: int) -> None:
        pass


def create_monitor(cfg: MonitoringConfig) -> BaseMonitor:
    """Factory: create the metrics backend."""
    try:
        return PrometheusMonitor(cfg)
    except Exception as exc:
        log.error("Failed to start Prometheus monitor", error=str(exc))
        return NullMonitor()
