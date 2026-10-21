"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
from src.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    PLACEHOLDER_API_KEY,
    AppConfig,
    load_config,
)


class TestAppConfig:
    """Tests for the AppConfig dataclass and validation methods."""

    def test_default_config_initialization(self, monkeypatch):
        """Should load default values safely when environment variables are not set."""
        # Clear any environment variables that might be set on the host system
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
        monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("CHUNK_SIZE", raising=False)
        monkeypatch.delenv("CHUNK_OVERLAP", raising=False)

        config = load_config()

        assert config.chat_model == DEFAULT_CHAT_MODEL
        assert config.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert config.chunk_size == DEFAULT_CHUNK_SIZE
        assert config.chunk_overlap == DEFAULT_CHUNK_OVERLAP
        assert config.upload_dir == Path("data/uploads")
        assert config.chroma_persist_dir == Path("data/chromadb")
        assert config.gemini_api_key is None
        assert not config.is_api_key_set()

    def test_custom_environment_values(self, monkeypatch):
        """Should correctly parse and assign custom environment variables."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key-12345")
        monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "custom-embed-model")
        monkeypatch.setenv("CHUNK_SIZE", "800")
        monkeypatch.setenv("CHUNK_OVERLAP", "150")

        config = load_config()

        assert config.gemini_api_key == "test-secret-key-12345"
        assert config.chat_model == "gemini-1.5-flash"
        assert config.embedding_model == "custom-embed-model"
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
        assert config.is_api_key_set()

    def test_invalid_chunk_size_in_env(self, monkeypatch):
        """Should raise ValueError when CHUNK_SIZE cannot be parsed or is <= 0."""
        monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
        with pytest.raises(ValueError, match="CHUNK_SIZE must be a valid integer"):
            load_config()

        monkeypatch.setenv("CHUNK_SIZE", "-50")
        with pytest.raises(ValueError, match="CHUNK_SIZE must be a positive integer"):
            load_config()

    def test_invalid_chunk_overlap_in_env(self, monkeypatch):
        """Should raise ValueError when CHUNK_OVERLAP >= CHUNK_SIZE."""
        monkeypatch.setenv("CHUNK_SIZE", "500")
        monkeypatch.setenv("CHUNK_OVERLAP", "500")
        with pytest.raises(ValueError, match="CHUNK_OVERLAP .* must be strictly less than"):
            load_config()

    def test_api_key_validation_missing(self):
        """validate_api_key should raise ValueError when key is None or empty."""
        config = AppConfig(
            gemini_api_key=None,
            chat_model="gemini-2.5-flash",
            embedding_model="text-embedding-004",
            chunk_size=1000,
            chunk_overlap=200,
            upload_dir=Path("data/uploads"),
            chroma_persist_dir=Path("data/chromadb"),
        )
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not set"):
            config.validate_api_key()

    def test_api_key_validation_placeholder(self):
        """validate_api_key should raise ValueError when key is the placeholder value."""
        config = AppConfig(
            gemini_api_key=PLACEHOLDER_API_KEY,
            chat_model="gemini-2.5-flash",
            embedding_model="text-embedding-004",
            chunk_size=1000,
            chunk_overlap=200,
            upload_dir=Path("data/uploads"),
            chroma_persist_dir=Path("data/chromadb"),
        )
        with pytest.raises(ValueError, match="still set to the default placeholder"):
            config.validate_api_key()

    def test_api_key_validation_success(self):
        """validate_api_key should succeed when a real key is present."""
        config = AppConfig(
            gemini_api_key="AIzaSyRealLookingKeyForUnitTestingOnly12345",
            chat_model="gemini-2.5-flash",
            embedding_model="text-embedding-004",
            chunk_size=1000,
            chunk_overlap=200,
            upload_dir=Path("data/uploads"),
            chroma_persist_dir=Path("data/chromadb"),
        )
        # Should not raise
        config.validate_api_key()
        assert config.is_api_key_set()

    def test_repr_masks_api_key(self):
        """String representation of AppConfig must NEVER reveal the secret API key."""
        secret_key = "super-confidential-api-key-9999"
        config = AppConfig(
            gemini_api_key=secret_key,
            chat_model="gemini-2.5-flash",
            embedding_model="text-embedding-004",
            chunk_size=1000,
            chunk_overlap=200,
            upload_dir=Path("data/uploads"),
            chroma_persist_dir=Path("data/chromadb"),
        )

        repr_str = repr(config)
        assert secret_key not in repr_str
        assert "gemini_api_key='[SET]'" in repr_str

        # When not set
        unset_config = AppConfig(
            gemini_api_key=None,
            chat_model="gemini-2.5-flash",
            embedding_model="text-embedding-004",
            chunk_size=1000,
            chunk_overlap=200,
            upload_dir=Path("data/uploads"),
            chroma_persist_dir=Path("data/chromadb"),
        )
        assert "gemini_api_key='[NOT SET]'" in repr(unset_config)
