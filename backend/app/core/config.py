"""
Environment-based configuration for ML Playground.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "ML Playground"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = int(os.getenv("WORKERS", "4"))

    # CORS
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_CREDENTIALS: bool = os.getenv("CORS_CREDENTIALS", "true").lower() == "true"

    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite+aiosqlite:///./ml_playground.db"
    )

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = DATA_DIR / "models"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"

    # ML
    MAX_DATASET_ROWS: int = int(os.getenv("MAX_DATASET_ROWS", "100000"))
    MAX_TRAINING_SECONDS: int = int(os.getenv("MAX_TRAINING_SECONDS", "120"))
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "4"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    LOG_JSON: bool = os.getenv("LOG_JSON", "true").lower() == "true"

    # Security
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

    # Caching
    EDA_CACHE_TTL: int = int(os.getenv("EDA_CACHE_TTL", "3600"))  # 1 hour
    PROFILE_CACHE_TTL: int = int(os.getenv("PROFILE_CACHE_TTL", "86400"))  # 24 hours

    def __init__(self):
        # Ensure directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()