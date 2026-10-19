"""
RescuGuard AI - Application Configuration
------------------------------------------
Defines centralized application settings, path resolutions, and environment
overrides suitable for both local development and containerized production environments.
"""

import os
from pathlib import Path
from typing import List


class Settings:
    """
    Centralized configuration settings for the RescuGuard AI engine.
    Supports dynamic environment variable overrides for Docker and cloud deployments.
    """

    # Project Metadata
    PROJECT_NAME: str = os.getenv(
        "PROJECT_NAME", 
        "RescuGuard AI — Accident Severity & Dispatch Prioritization API"
    )
    VERSION: str = "1.1.0"
    DESCRIPTION: str = (
        "Real-time decision-support engine for traffic accident severity prediction "
        "and automated emergency dispatch prioritization (INSA Cyber Talent 2026)."
    )

    # Network & Server Runtime
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Directory Path Anchors
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    FRONTEND_DIR: Path = BASE_DIR / "frontend"

    # Trained Model Artifact
    MODEL_PATH: Path = Path(
        os.getenv(
            "MODEL_PATH", 
            str(PROCESSED_DATA_DIR / "model_pipeline.joblib")
        )
    )

    # CORS Allowed Origins
    _cors_raw: str = os.getenv("CORS_ORIGINS", "*")
    CORS_ORIGINS: List[str] = (
        [origin.strip() for origin in _cors_raw.split(",")]
        if _cors_raw != "*"
        else ["*"]
    )


# Global settings singleton
settings = Settings()