"""
config/settings.py
──────────────────
Loads YAML config + overrides from environment variables.

Override any field via: IVC_<SECTION>__<KEY>=value
Example:  IVC_ASR__MODEL=distil-large-v3
          IVC_API__PORT=9000
          IVC_SESSION__BACKEND=memory
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Sub-models ────────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    name: str = "indic-voice-cloner"
    version: str = "1.0.0"
    env: str = "production"
    log_level: str = "INFO"


class LanguageConfig(BaseModel):
    primary: List[str] = ["ta", "te", "kn", "ml"]
    extended: List[str] = ["hi", "bn", "mr", "gu", "pa", "en"]

    @property
    def all(self) -> List[str]:
        return self.primary + self.extended


class VADConfig(BaseModel):
    model: str = "silero"
    threshold: float = 0.50
    min_silence_ms: int = 300
    speech_pad_ms: int = 30
    sample_rate: int = 16000
    chunk_samples: int = 512
    onnx_mode: bool = True


class ASRConfig(BaseModel):
    model: str = "large-v3"
    compute_type: str = "float16"
    device: str = "auto"
    beam_size: int = 1
    vad_filter: bool = False
    word_timestamps: bool = True
    condition_on_previous: bool = False
    download_root: str = "models/whisper"
    language_hints: Dict[str, List[str]] = Field(default_factory=dict)


class ParlerConfig(BaseModel):
    model_id: str = "ai4bharat/indic-parler-tts"
    dtype: str = "float16"
    device: str = "auto"
    voice_descriptions: Dict[str, str] = Field(default_factory=lambda: {
        "ta": "A clear, natural Tamil female voice with neutral Chennai accent.",
        "te": "A clear, natural Telugu male voice with neutral Hyderabad accent.",
        "kn": "A clear, natural Kannada female voice with neutral Bangalore accent.",
        "ml": "A clear, natural Malayalam male voice with neutral Kerala accent.",
        "hi": "A clear, natural Hindi voice with neutral accent.",
        "en": "A clear, natural English voice with Indian accent.",
        "default": "A clear, natural, neutral voice.",
    })

    def voice_for(self, lang: str) -> str:
        return self.voice_descriptions.get(lang, self.voice_descriptions["default"])


class XTTSConfig(BaseModel):
    model_id: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = "auto"
    reference_min_seconds: int = 6
    reference_max_seconds: int = 30
    split_sentences: bool = False


class F5Config(BaseModel):
    model_id: str = "F5-TTS"
    device: str = "auto"
    dtype: str = "float16"


class SvaraConfig(BaseModel):
    model_id: str = "kenpath/svara-tts-voiceclone-beta"
    device: str = "auto"
    min_reference_seconds: float = 3.0
    max_reference_seconds: float = 15.0
    reference_target_seconds: float = 10.0
    max_new_tokens: int = 1200
    temperature: float = 0.55
    top_p: float = 0.90


class TTSConfig(BaseModel):
    backend: str = "auto"
    stream_chunk_words: int = 5
    sample_rate: int = 24000
    sentence_ends: List[str] = [".", "!", "?", "।", "؟"]
    svara: SvaraConfig = Field(default_factory=SvaraConfig)
    parler: ParlerConfig = Field(default_factory=ParlerConfig)
    xtts: XTTSConfig = Field(default_factory=XTTSConfig)
    f5: F5Config = Field(default_factory=F5Config)


class AWQQuantConfig(BaseModel):
    zero_point: bool = True
    q_group_size: int = 128
    w_bit: int = 4
    version: str = "GEMM"


class SpeculativeConfig(BaseModel):
    enabled: bool = False
    draft_model: str = "microsoft/phi-1_5"
    num_tokens: int = 5


class SLMConfig(BaseModel):
    model_id: str = "microsoft/Phi-3.5-mini-instruct"
    quantization: str = "awq"
    max_tokens: int = 256
    temperature: float = 0.7
    tensor_parallel: int = 1
    gpu_memory_utilization: float = 0.85
    speculative: SpeculativeConfig = Field(default_factory=SpeculativeConfig)
    awq_save_path: str = "models/phi-3.5-mini-awq"
    awq_quant_config: AWQQuantConfig = Field(default_factory=AWQQuantConfig)


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    socket_timeout: int = 2
    key_prefix: str = "ivc:session:"


class SessionConfig(BaseModel):
    backend: str = "redis"
    ttl_seconds: int = 1800
    max_history_turns: int = 6
    redis: RedisConfig = Field(default_factory=RedisConfig)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = ["*"]
    max_ws_connections: int = 100
    audio_format: str = "wav"


class MonitoringConfig(BaseModel):
    prometheus_port: int = 9090
    enable_gpu_metrics: bool = True
    gpu_poll_interval_s: int = 5
    sla_ttfb_ms: int = 1000


class InfraConfig(BaseModel):
    region: str = "ap-south-1"
    model_cache_dir: str = "models"
    audio_samples_dir: str = "audio_samples"
    log_dir: str = "logs"


# ── Root Settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Master settings object.
    Loaded from YAML, then overridden by IVC_* env vars.
    """
    model_config = SettingsConfigDict(
        env_prefix="IVC_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    app: AppConfig = Field(default_factory=AppConfig)
    languages: LanguageConfig = Field(default_factory=LanguageConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    slm: SLMConfig = Field(default_factory=SLMConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from YAML file + environment variable overrides.

    Priority (highest → lowest):
      1. Environment variables (IVC_*)
      2. YAML config file
      3. Pydantic field defaults
    """
    yaml_data: Dict[str, Any] = {}

    # Resolve config file path
    if config_path is None:
        config_path = os.environ.get(
            "IVC_CONFIG_PATH",
            str(Path(__file__).parent / "default.yaml")
        )

    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
    else:
        import warnings
        warnings.warn(f"Config file not found: {config_file}. Using defaults.")

    return Settings(**yaml_data)


# ── Singleton ─────────────────────────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Force reload settings (useful for testing)."""
    global _settings
    _settings = load_settings(config_path)
    return _settings
