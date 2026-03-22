# ─────────────────────────────────────────────────────────────────────────────
# Indic Voice Cloner — Production Dockerfile
# Base: NVIDIA CUDA 12.1 + Python 3.10 (Ubuntu 22.04)
# GPU: NVIDIA L4 / A10G / A100 recommended
# Region target: ap-south-1 (Mumbai) for South Indian users
# ─────────────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ── Environment ───────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev python3.10-venv \
    ffmpeg libsndfile1 portaudio19-dev \
    git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python python python3.10 1

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt

# ── Copy application ───────────────────────────────────────────────────────────
COPY . .

# ── Pre-download models for faster cold starts ────────────────────────────────
# Comment out to skip model bake-in (reduces image size but increases cold-start time)
RUN python3 -c "\
from faster_whisper import WhisperModel; \
WhisperModel('large-v3', device='cpu', compute_type='int8', download_root='models/whisper'); \
print('Whisper model cached')"

# ── Create required directories ───────────────────────────────────────────────
RUN mkdir -p models audio_samples logs configs

# ── Health check ──────────────────────────────────────────────────────────────
EXPOSE 8000 9090

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "info"]
