"""Application settings: every constant and environment variable lives here.

Values are read from environment variables (and from a local ``.env`` file in
development). Secrets are typed as ``SecretStr`` so they never appear in
``repr()``, logs or error messages by accident; code must call
``.get_secret_value()`` explicitly at the single place a key is needed.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"


@dataclass(frozen=True)
class ChunkConfig:
    """One chunking strategy: target chunk size and overlap, in characters."""

    name: str
    chunk_size: int
    chunk_overlap: int


# The two configurations compared in the evaluation (see DESIGN.md / EVAL.md).
CHUNK_CONFIGS: dict[str, ChunkConfig] = {
    "A": ChunkConfig(name="A", chunk_size=1000, chunk_overlap=150),
    "B": ChunkConfig(name="B", chunk_size=500, chunk_overlap=75),
}

# File types accepted for upload, keyed by lower-case extension.
SUPPORTED_EXTENSIONS: dict[str, str] = {".pdf": "pdf", ".docx": "docx"}

# Request limits used by the API schemas (fixed, so they appear in the OpenAPI docs).
MAX_QUESTION_CHARS = 1000
MAX_TOP_K = 10
MAX_DOCUMENT_FILTER = 20


class Settings(BaseSettings):
    """All configurable values for the RAG service."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # unknown variables in .env are ignored, not errors
    )

    # --- Application -------------------------------------------------------
    app_name: str = "Production RAG Service"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", description="development | test | production")
    log_level: str = "INFO"

    # --- Storage -----------------------------------------------------------
    # Runtime data (uploads, Chroma files, SQLite registry). Gitignored.
    # Can point outside OneDrive to avoid file-lock problems with sync.
    data_dir: Path = PROJECT_ROOT / "data" / "runtime"

    # --- LLM: Groq, called through the OpenAI-compatible endpoint ----------
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str | None = Field(default=None, description="Groq model ID for answers (OPEN decision)")
    judge_model: str | None = Field(default=None, description="Different Groq model ID for the eval judge")
    llm_timeout_seconds: float = 30.0

    # --- Embeddings: Gemini, called through the OpenAI-compatible endpoint -
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    embedding_model: str | None = Field(default=None, description="Gemini embedding model ID (OPEN decision)")
    embedding_dimensions: int = 768
    embedding_timeout_seconds: float = 20.0
    embedding_batch_size: int = Field(default=50, ge=1, le=100, description="Texts per embedding request")

    # --- Answer generation -------------------------------------------------
    llm_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    # For reasoning models the hidden reasoning counts toward this limit too.
    llm_max_output_tokens: int = Field(default=2000, ge=100, le=8000)
    # Sent only to reasoning-model families that accept it (see generation.py); "" disables it.
    llm_reasoning_effort: str = Field(default="low", description="low | medium | high, or empty")
    # Subject area of the corpus. Questions clearly outside it are refused.
    corpus_topic: str = (
        "AI risk management, trustworthy and responsible AI, generative AI risks, "
        "and security risks of large language model (LLM) applications"
    )

    # --- Retrieval ---------------------------------------------------------
    top_k: int = Field(default=5, ge=1, le=MAX_TOP_K, description="Chunks retrieved per question")
    # Cosine-similarity cut-offs, calibrated on the golden set with gemini-embedding-001
    # at 768 dimensions (see EVAL.md). Recalibrate if the embedding model changes.
    # Off-topic questions scored 0.43-0.48 and in-domain questions 0.64 or more.
    # Below the floor the question is refused as off-topic without calling the LLM.
    off_topic_floor: float = Field(default=0.55, ge=0.0, le=1.0)
    # Answerable questions scored at least 0.72. Below this score the documents are
    # not trusted to answer, so no cited answer is given.
    answer_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    # --- Reliability -------------------------------------------------------
    external_max_retries: int = Field(default=2, ge=0, le=5, description="Retries for LLM/embedding calls")

    # --- Rate limiting (per client IP, in-memory, one server process) -----
    questions_per_minute: int = Field(default=10, ge=1)
    uploads_per_minute: int = Field(default=10, ge=1)

    # --- Upload limits -----------------------------------------------------
    max_upload_mb: int = Field(default=20, ge=1, description="Largest accepted upload, in MB")
    max_pdf_pages: int = Field(default=500, ge=1)
    max_docx_uncompressed_mb: int = Field(default=100, ge=1, description="ZIP-bomb guard for DOCX files")
    max_docx_zip_entries: int = Field(default=5000, ge=10)

    # --- Chunking ----------------------------------------------------------
    chunk_config: Literal["A", "B"] = Field(default="B", description="Active chunking configuration")
    min_chunk_chars: int = Field(default=20, ge=1, description="Drop chunks shorter than this (non-space chars)")

    @model_validator(mode="after")
    def _check_thresholds(self) -> "Settings":
        if self.off_topic_floor > self.answer_threshold:
            raise ValueError("OFF_TOPIC_FLOOR must not be higher than ANSWER_THRESHOLD.")
        return self

    @property
    def max_upload_bytes(self) -> int:
        """Upload size limit in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def active_chunk_config(self) -> ChunkConfig:
        """The chunking configuration selected by ``chunk_config``."""
        return CHUNK_CONFIGS[self.chunk_config]

    @property
    def groq_configured(self) -> bool:
        """True when a Groq key and an answer model are both set."""
        return bool(self.groq_api_key and self.groq_api_key.get_secret_value() and self.llm_model)

    @property
    def gemini_configured(self) -> bool:
        """True when a Gemini key and an embedding model are both set."""
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value() and self.embedding_model)

    def secret_values(self) -> list[str]:
        """Return the raw secret strings so the log filter can redact them."""
        secrets = [self.groq_api_key, self.gemini_api_key]
        return [s.get_secret_value() for s in secrets if s and s.get_secret_value()]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings (read once, then cached)."""
    return Settings()
