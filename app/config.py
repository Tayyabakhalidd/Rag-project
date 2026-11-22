"""Application settings: every constant and environment variable lives here.

Values are read from environment variables (and from a local ``.env`` file in
development). Secrets are typed as ``SecretStr`` so they never appear in
``repr()``, logs or error messages by accident; code must call
``.get_secret_value()`` explicitly at the single place a key is needed.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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

    # --- Reliability -------------------------------------------------------
    external_max_retries: int = Field(default=2, ge=0, le=5, description="Retries for LLM/embedding calls")

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
