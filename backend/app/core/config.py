import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Research Mind AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Gemini API Key
    GEMINI_API_KEY: str = ""

    # Server Configuration
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # Storage Paths (resolved relative to project root or config location)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CHROMA_PERSIST_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "data" / "chromadb")
    UPLOAD_DIR: str = str(Path(__file__).resolve().parent.parent.parent / "data" / "uploads")

    # RAG Hyperparameters
    CHUNK_SIZE_CHARS: int = 1000
    CHUNK_OVERLAP_CHARS: int = 200
    TOP_K_DENSE: int = 6
    TOP_K_SPARSE: int = 6
    RRF_K: int = 60  # Reciprocal Rank Fusion constant

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        os.makedirs(self.CHROMA_PERSIST_DIR, exist_ok=True)
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)


settings = Settings()
settings.ensure_directories()
