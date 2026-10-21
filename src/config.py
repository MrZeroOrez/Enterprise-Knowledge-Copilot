"""Configuration module for Enterprise Knowledge Copilot.

Loads and validates environment variables from .env or system environment.
Ensures secrets are never exposed in string representations or logs.
Provides clear, actionable error messages when configuration is invalid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Default constants (fallback values when not specified in environment)
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-004"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_UPLOAD_DIR = "data/uploads"
DEFAULT_CHROMA_DIR = "data/chromadb"
DEFAULT_CHROMA_COLLECTION_NAME = "enterprise_documents"
DEFAULT_MAX_UPLOAD_SIZE_MB = 10

PLACEHOLDER_API_KEY = "your_gemini_api_key_here"


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration dataclass."""

    gemini_api_key: str | None
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    upload_dir: Path
    chroma_persist_dir: Path
    chroma_collection_name: str = DEFAULT_CHROMA_COLLECTION_NAME
    max_upload_size_mb: int = DEFAULT_MAX_UPLOAD_SIZE_MB

    @property
    def max_upload_size_bytes(self) -> int:
        """Return maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    def validate_api_key(self) -> None:
        """Validate that the Gemini API key is configured and not a placeholder.

        Note: Phase 1 & 2 do not require an API key to run tests or local chunking.
        This validation should be invoked when initializing Gemini services in Phase 3+.

        Raises:
            ValueError: If the API key is missing or set to the placeholder value.
        """
        if not self.gemini_api_key or self.gemini_api_key.strip() == "":
            raise ValueError(
                "GEMINI_API_KEY is not set. Please obtain an API key from "
                "Google AI Studio (https://aistudio.google.com/) and set it in your .env file."
            )
        if self.gemini_api_key.strip() == PLACEHOLDER_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is still set to the default placeholder value. "
                "Please replace it with your actual Gemini API key in your .env file."
            )

    def validate_embedding_model(self) -> None:
        """Validate embedding model configuration.

        Raises:
            ValueError: If the embedding model name is empty.
        """
        if not self.embedding_model or not self.embedding_model.strip():
            raise ValueError(
                "GEMINI_EMBEDDING_MODEL cannot be empty. Please configure a supported model "
                "such as 'text-embedding-004' in your .env file."
            )

    def validate_chunking_params(self) -> None:
        """Validate document chunking parameters.

        Raises:
            ValueError: If chunk_size or chunk_overlap violate invariants.
        """
        if self.chunk_size <= 0:
            raise ValueError(
                f"CHUNK_SIZE must be a positive integer (> 0), got: {self.chunk_size}"
            )
        if self.chunk_overlap < 0:
            raise ValueError(
                f"CHUNK_OVERLAP cannot be negative (>= 0), got: {self.chunk_overlap}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be strictly less than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )

    def validate_upload_params(self) -> None:
        """Validate upload parameters.

        Raises:
            ValueError: If max_upload_size_mb <= 0.
        """
        if self.max_upload_size_mb <= 0:
            raise ValueError(
                f"MAX_UPLOAD_SIZE_MB must be a positive integer (> 0), got: {self.max_upload_size_mb}"
            )

    def is_api_key_set(self) -> bool:
        """Check if an API key is set without exposing the key value."""
        return bool(
            self.gemini_api_key
            and self.gemini_api_key.strip()
            and self.gemini_api_key.strip() != PLACEHOLDER_API_KEY
        )

    def __repr__(self) -> str:
        """Safe string representation that masks sensitive secrets."""
        key_status = "[SET]" if self.is_api_key_set() else "[NOT SET]"
        return (
            f"AppConfig("
            f"gemini_api_key='{key_status}', "
            f"chat_model='{self.chat_model}', "
            f"embedding_model='{self.embedding_model}', "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}, "
            f"max_upload_size_mb={self.max_upload_size_mb}, "
            f"upload_dir='{self.upload_dir}', "
            f"chroma_persist_dir='{self.chroma_persist_dir}', "
            f"chroma_collection_name='{self.chroma_collection_name}')"
        )


def load_config(env_path: Path | str | None = None) -> AppConfig:
    """Load configuration from environment variables or a specific .env file.

    Args:
        env_path: Optional custom path to a .env file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ValueError: If configuration values cannot be parsed or are invalid.
    """
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv(override=False)

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    chat_model = (
        os.getenv("GEMINI_CHAT_MODEL") or os.getenv("GEMINI_MODEL") or DEFAULT_CHAT_MODEL
    ).strip()
    if not chat_model:
        raise ValueError("GEMINI_CHAT_MODEL cannot be empty.")

    embedding_model = (
        os.getenv("GEMINI_EMBEDDING_MODEL") or os.getenv("GEMINI_EMBED_MODEL") or DEFAULT_EMBEDDING_MODEL
    ).strip()
    if not embedding_model:
        raise ValueError("GEMINI_EMBEDDING_MODEL cannot be empty.")

    chunk_size_raw = os.getenv("CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE))
    try:
        chunk_size = int(chunk_size_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"CHUNK_SIZE must be a valid integer, got: '{chunk_size_raw}'") from exc

    chunk_overlap_raw = os.getenv("CHUNK_OVERLAP", str(DEFAULT_CHUNK_OVERLAP))
    try:
        chunk_overlap = int(chunk_overlap_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"CHUNK_OVERLAP must be a valid integer, got: '{chunk_overlap_raw}'") from exc

    upload_dir = Path(os.getenv("UPLOAD_DIR", DEFAULT_UPLOAD_DIR))
    chroma_dir = Path(
        os.getenv("CHROMA_PERSIST_DIRECTORY") or os.getenv("CHROMA_PERSIST_DIR") or DEFAULT_CHROMA_DIR
    )
    chroma_collection_name = (
        os.getenv("CHROMA_COLLECTION_NAME") or DEFAULT_CHROMA_COLLECTION_NAME
    ).strip()
    if not chroma_collection_name:
        raise ValueError("CHROMA_COLLECTION_NAME cannot be empty.")

    max_upload_size_raw = os.getenv("MAX_UPLOAD_SIZE_MB", str(DEFAULT_MAX_UPLOAD_SIZE_MB))
    try:
        max_upload_size_mb = int(max_upload_size_raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"MAX_UPLOAD_SIZE_MB must be a valid integer, got: '{max_upload_size_raw}'"
        ) from exc

    config = AppConfig(
        gemini_api_key=gemini_api_key,
        chat_model=chat_model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        upload_dir=upload_dir,
        chroma_persist_dir=chroma_dir,
        chroma_collection_name=chroma_collection_name,
        max_upload_size_mb=max_upload_size_mb,
    )

    # Validate non-API parameters on load to fail-fast on malformed settings
    config.validate_chunking_params()
    config.validate_upload_params()
    config.validate_embedding_model()

    return config
