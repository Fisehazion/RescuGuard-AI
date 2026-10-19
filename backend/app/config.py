import os
from pathlib import Path


class Settings:
    PROJECT_NAME: str = "Traffic Accident Severity & Response Prioritization API"
    VERSION: str = "1.0.0"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_PATH: Path = BASE_DIR / "data" / "processed" / "model_pipeline.joblib"


settings = Settings()