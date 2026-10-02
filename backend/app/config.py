"""
Application configuration module using Pydantic Settings.
Manages hyper-parameters, model identifiers, reproducibility seeds, and system paths.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Application Metadata
    APP_NAME: str = "Grammar Error Detection & Correction API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: Literal["development", "production", "testing"] = "development"

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8001

    # Project Root and Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    CHECKPOINTS_DIR: Path = BASE_DIR / "models" / "checkpoints"

    # Model & Inference Settings
    # Primary pretrained model for GEC (high quality, fast on CPU)
    MODEL_NAME_OR_PATH: str = "vennify/t5-base-grammar-correction"
    FALLBACK_MODEL_NAME: str = "t5-small"
    DEVICE: str = "cpu"
    MAX_INPUT_LENGTH: int = 128
    MAX_TARGET_LENGTH: int = 128
    NUM_BEAMS: int = 4
    NUM_RETURN_SEQUENCES: int = 3
    TEMPERATURE: float = 1.0
    TOP_P: float = 0.95
    RANDOM_SEED: int = 42

    # Evaluation Configuration
    GLEU_MIN_N: int = 1
    GLEU_MAX_N: int = 4
    FBETA_BETA: float = 0.5  # Standard F0.5 for GEC shared tasks (CoNLL/BEA)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
