# 🎙️ Indic Voice Cloner
### Enterprise-Grade Voice Cloning for South Indian Languages

**Tamil · Telugu · Kannada · Malayalam** · + Hindi, Bengali, English, and more

---

## Architecture

```
[Microphone / Audio Input]
        │
        ▼
   [VAD: Silero]  ← ~20ms per 32ms chunk
        │ utterance detected
        ▼
  [ASR: faster-whisper large-v3]  ← ~150ms for 5s utterance
        │ text + language
        ▼
  [SLM: Phi-3.5-mini @ vLLM AWQ]  ← ~80ms time-to-first-token
        │ token stream
        ▼
  [TTS: Indic-Parler / XTTS v2 Streaming]  ← ~200ms first chunk
        │ audio chunks
        ▼
   [Client Speaker]

Total TTFB target: < 600ms  |  SLA: < 1000ms
```

---

## Quickstart

### 1. Install
```bash
# GPU environment (CUDA 12.1+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Run locally
```bash
python main.py serve
# API: http://localhost:8000
# Metrics: http://localhost:9090/metrics
```

### 3. Docker
```bash
cd deploy/docker
docker compose up --build
```

### 4. Kubernetes
```bash
kubectl apply -f deploy/k8s/deployment.yaml
```

---

## Supported Languages

| Code | Language   | Primary Focus |
|------|-----------|---------------|
| `ta` | Tamil     | ✅ Primary     |
| `te` | Telugu    | ✅ Primary     |
| `kn` | Kannada   | ✅ Primary     |
| `ml` | Malayalam | ✅ Primary     |
| `hi` | Hindi     | Extended      |
| `bn` | Bengali   | Extended      |
| `mr` | Marathi   | Extended      |
| `gu` | Gujarati  | Extended      |
| `pa` | Punjabi   | Extended      |
| `en` | English   | Extended      |

---

## Configuration

All settings live in `config/default.yaml`. Override any value with environment variables:

```bash
IVC_ASR__MODEL=distil-large-v3      # Faster ASR (trade accuracy for speed)
IVC_TTS__BACKEND=xtts               # Switch to XTTS for voice cloning
IVC_SESSION__BACKEND=memory         # In-memory sessions (no Redis needed)
IVC_API__PORT=9000                  # Change API port
IVC_APP__LOG_LEVEL=DEBUG            # Verbose logging
```

### Customize South Indian voice descriptions
Edit `config/default.yaml` → `tts.parler.voice_descriptions`:
```yaml
tts:
  parler:
    voice_descriptions:
      ta: "A warm, natural Tamil female voice with Chennai accent."
      kn: "A crisp, clear Kannada male voice with Bangalore accent."
```

---

## API Reference

### WebSocket: `/ws/voice`

**Client → Server:**
```json
{"type": "audio_chunk", "audio": "<base64 float32 PCM @ 16kHz>", "language": "ta"}
{"type": "text_query",  "text": "வணக்கம்", "language": "ta"}
{"type": "reset"}
{"type": "update_voice", "path": "/path/to/reference.wav"}
```

**Server → Client:**
```json
{"type": "asr_result",   "text": "...", "language": "ta", "confidence": 0.98}
{"type": "audio_chunk",  "audio": "<base64 WAV>", "chunk_index": 0}
{"type": "ttfb",         "ms": 432.1}
{"type": "turn_complete","session_id": "..."}
{"type": "error",        "stage": "asr", "msg": "..."}
```

### REST: `POST /tts`
```json
{
  "text": "வணக்கம், நான் உங்களுக்கு உதவ ஆர்வமாக இருக்கிறேன்.",
  "language": "ta",
  "session_id": "optional-session-id"
}
```

### REST: `POST /clone`
```json
{
  "reference_wav_b64": "<base64 6-30s WAV>",
  "text": "Your text here",
  "language": "ta"
}
```

---

## Modifying Components

The pipeline uses **dependency injection**. Swap any component without touching other code:

### Custom TTS Backend
```python
from core.interfaces import BaseTTSBackend
import numpy as np

class MyCustomTTS(BaseTTSBackend):
    def synthesize(self, text, language, reference_wav=None, **kwargs) -> np.ndarray:
        # your synthesis code
        return audio_array

    @property
    def sample_rate(self) -> int:
        return 22050

    @property
    def supported_languages(self):
        return ["ta", "te", "kn", "ml"]

# Register it
from tts.engine import create_tts_engine
engine = create_tts_engine(cfg.tts)
engine.add_backend("my_model", MyCustomTTS())
```

### Custom ASR
```python
from core.interfaces import BaseASR, ASRResult

class MyASR(BaseASR):
    def transcribe(self, audio_array, language="auto") -> ASRResult:
        ...

# Inject into pipeline
from core.pipeline import PipelineOrchestrator
pipeline = PipelineOrchestrator(
    settings=settings,
    asr=MyASR(),   # ← swap here
    ...
)
```

### Custom SLM (any LLM provider)
```python
from core.interfaces import BaseSLM, SLMToken

class OpenAISLM(BaseSLM):
    async def generate_stream(self, user_text, history, language="auto"):
        async for chunk in openai_client.chat.completions.create(..., stream=True):
            yield SLMToken(text=chunk.choices[0].delta.content or "")

    async def generate(self, user_text, history, language="auto"):
        ...
```

---

## Performance Tuning

### Latency Budget (GPU)
| Stage         | Target  | Tuning Lever                         |
|---------------|---------|--------------------------------------|
| VAD           | ~20ms   | `vad.chunk_samples` (fixed at 512)   |
| ASR           | ~150ms  | Use `distil-large-v3` for -50ms      |
| SLM TTFT      | ~80ms   | Lower `slm.max_tokens`; AWQ quant    |
| TTS 1st chunk | ~200ms  | Lower `tts.stream_chunk_words` (3–5) |
| Network RTT   | ~50ms   | Deploy in `ap-south-1` region        |
| **Total TTFB**| **~600ms** |                                   |

### GPU Memory Budget (single L4 / 24GB VRAM)
| Component | VRAM   |
|-----------|--------|
| Whisper large-v3 | ~3GB  |
| Phi-3.5 mini AWQ | ~2.5GB|
| Indic-Parler-TTS | ~8GB  |
| XTTS v2          | ~5GB  |
| **Total**        | **~14–18GB** |

Use XTTS **or** Parler — not both — to stay within 24GB VRAM.

---

## Running Tests
```bash
pytest tests/ -v                      # All unit tests
pytest tests/ -v --cov=.             # With coverage
pytest tests/integration/ -v         # Integration tests
python main.py benchmark              # Latency benchmark
```

---

## Project Structure

```
indic_voice_cloner/
├── config/
│   ├── default.yaml          ← All settings (edit here first)
│   └── settings.py           ← Pydantic settings loader + env overrides
├── core/
│   ├── interfaces.py         ← Abstract base classes (swap any component)
│   ├── pipeline.py           ← Main orchestrator (DI-wired)
│   ├── profiler.py           ← Latency benchmarking
│   ├── logger.py             ← Structured logging (JSON in prod)
│   └── device.py             ← GPU/CPU device helpers
├── vad/
│   └── silero_vad.py         ← Silero VAD + WebRTC fallback
├── asr/
│   └── whisper_asr.py        ← faster-whisper (South Indian tuned)
├── tts/
│   └── engine.py             ← Parler + XTTS streaming engine
├── slm/
│   └── vllm_engine.py        ← vLLM + AWQ + HF fallback
├── session/
│   └── manager.py            ← Redis + in-memory session store
├── monitoring/
│   └── metrics.py            ← Prometheus metrics
├── api/
│   └── app.py                ← FastAPI WebSocket + REST API
├── tests/
│   ├── unit/                 ← Component unit tests (mocked)
│   └── integration/          ← E2E pipeline tests
├── deploy/
│   ├── docker/               ← Dockerfile + docker-compose.yml
│   └── k8s/                  ← Kubernetes deployment + HPA
├── main.py                   ← CLI entrypoint
└── requirements.txt
```

---

## Monitoring

Grafana dashboard at `http://localhost:3000` (admin / indicvoice123)

Key panels to set up:
- `ivc_ttfb_ms` — TTFB histogram (p50/p95/p99)
- `ivc_sla_violations_total` — SLA breach counter
- `ivc_requests_total` by `language` label — traffic by language
- `ivc_active_sessions` — live WebSocket sessions
- `ivc_gpu_vram_used_gb` — VRAM utilization
- `ivc_errors_total` by `stage` — error rate by pipeline stage

---

## License
Proprietary. See LICENSE for terms.
