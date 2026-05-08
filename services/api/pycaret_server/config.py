"""Environment-driven configuration via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Server configuration. Read from env vars with prefix ``PYCARET_`` or from
    a ``.env`` file at the project root.

    Examples:
        PYCARET_DATABASE_URL=sqlite:///./pycaret.db
        PYCARET_JWT_SECRET=<hex>
        PYCARET_ARTIFACT_DIR=./artifacts
    """

    model_config = SettingsConfigDict(
        env_prefix="PYCARET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app identity ---
    app_name: str = "PyCaret Server"
    environment: str = Field(default="dev", description="dev | staging | prod")
    debug: bool = False

    # --- database ---
    database_url: str = "sqlite:///./pycaret.db"

    # --- authentication ---
    # JWT secret — override in prod via PYCARET_JWT_SECRET env var. Dev default
    # is deliberately weak and shows up in logs so it never ships accidentally.
    jwt_secret: str = "dev-only-jwt-secret-do-not-use-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_ttl_minutes: int = 60
    refresh_token_ttl_days: int = 30

    # --- secrets encryption ---
    # Fernet key for at-rest encryption of LLM API keys + future cloud creds.
    # Generate one with:
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    # If unset, an ephemeral per-process key is generated and a warning is
    # logged; encrypted values do NOT survive restart. Set this in any
    # environment that stores real credentials.
    secrets_key: str | None = None

    # --- artifact storage ---
    # Root dir where Run outputs (run.ipynb, fitted_pipeline.pkl, leaderboard.json,
    # events.jsonl, preview.html) land. In prod, a persistent volume or S3.
    artifact_dir: Path = Path("./artifacts")

    # --- CORS ---
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # --- feature flags (platform phase progression) ---
    enable_deployments: bool = True  # in-house serving
    enable_websocket: bool = True  # event-stream fan-out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
