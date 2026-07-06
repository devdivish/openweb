from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


DEFAULT_SYSTEM_PROMPT = (
    "You are a file-grounded QA assistant. Use source context when it is relevant. "
    "Cite sources with bracket numbers like [1]. If the sources are insufficient, say so clearly."
)

DEFAULT_RAG_TEMPLATE = (
    "Use the following source context to answer the user's question.\n\n"
    "{context}\n\n"
    "Question: {question}"
)


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("APP_DATA_DIR", ".data")))
    embedding_provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local_hash"))
    embedding_dimensions: int = field(default_factory=lambda: _int_env("EMBEDDING_DIMENSIONS", 384))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "extractive"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_chat_model: str = field(default_factory=lambda: os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"))
    ollama_embedding_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
    top_k: int = field(default_factory=lambda: _int_env("TOP_K", 5))
    relevance_threshold: float = field(default_factory=lambda: _float_env("RELEVANCE_THRESHOLD", 0.05))
    chat_context_messages: int = field(default_factory=lambda: _int_env("CHAT_CONTEXT_MESSAGES", 12))
    chat_memory_items: int = field(default_factory=lambda: _int_env("CHAT_MEMORY_ITEMS", 40))
    chat_memory_context_chars: int = field(default_factory=lambda: _int_env("CHAT_MEMORY_CONTEXT_CHARS", 3000))
    chunk_size: int = field(default_factory=lambda: _int_env("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _int_env("CHUNK_OVERLAP", 180))
    full_context_max_chars: int = field(default_factory=lambda: _int_env("FULL_CONTEXT_MAX_CHARS", 24000))
    system_prompt: str = field(default_factory=lambda: os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT))
    rag_template: str = field(default_factory=lambda: os.getenv("RAG_TEMPLATE", DEFAULT_RAG_TEMPLATE))
    # External OCR extraction service (ocr_detection_worker_updated /extract).
    # When set, every uploaded file is extracted via this endpoint instead of
    # native pypdf/docx2txt parsing.
    ocr_extraction_url: str = field(default_factory=lambda: os.getenv("OCR_EXTRACTION_URL", ""))
    ocr_extraction_format: str = field(default_factory=lambda: os.getenv("OCR_EXTRACTION_FORMAT", "markdown"))
    ocr_extraction_timeout: float = field(default_factory=lambda: _float_env("OCR_EXTRACTION_TIMEOUT", 300.0))
    ocr_fallback_native: bool = field(
        default_factory=lambda: (os.getenv("OCR_FALLBACK_NATIVE", "true").strip().lower() in {"1", "true", "yes", "on"})
    )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "state.json"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "backend.sqlite3"


def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
