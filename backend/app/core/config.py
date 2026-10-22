"""Application settings, read from environment variables / .env at the repo root."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> repo root is three parents above backend/app/core
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AFTERSIGHT_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    outputs_dir: Path = Path("outputs")
    frontend_origin: str = "http://localhost:5200"

    def resolve(self, path: Path) -> Path:
        """Turn a repo-relative path from config into an absolute path."""
        return path if path.is_absolute() else REPO_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
