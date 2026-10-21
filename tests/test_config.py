"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest
from src.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RAG_MAX_CONTEXT_CHARS,
    DEFAULT_RETRIEVAL_MAX_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    PLACEHOLDER_API_KEY,
    AppConfig,
    load_config,
)


class TestAppConfig:
    """Tests for the AppConfig dataclass and validation methods."""

    def test_default_config_initialization(self, monkeypatch, tmp_path):
        """Should load default values safely when environment variables are not set."""
        # Clear any environment variables that might be set on the host system
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
        monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("CHUNK_SIZE", raising=False)
        monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
        monkeypatch.delenv("MAX_UPLOAD_SIZE_MB", raising=False)
        monkeypatch.delenv("RAG_MAX_CONTEXT_CHARS", raising=False)
        monkeypatch.delenv("RAG_TOP_K", raising=False)

        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        config = load_config(env_path=empty_env)

        assert config.chat_model == DEFAULT_CHAT_MODEL
        assert config.embedding_model == DEFAULT_EMBEDDING_MODEL
        assert config.chunk_size == DEFAULT_CHUNK_SIZE
        assert config.chunk_overlap == DEFAULT_CHUNK_OVERLAP
        assert config.max_upload_size_mb == 10
        assert config.max_upload_size_bytes == 10 * 1024 * 1024
        assert config.rag_max_context_chars == DEFAULT_RAG_MAX_CONTEXT_CHARS
        assert config.log_level == "INFO"
        assert config.log_file == "logs/app.log"
        assert config.upload_dir == Path("data/uploads")
        assert config.chroma_persist_dir == Path("data/chromadb")
        assert config.chroma_collection_name == "enterprise_documents"
        assert config.gemini_api_key is None
        assert not config.is_api_key_set()

    def test_custom_environment_values(self, monkeypatch):
        """Should correctly parse and assign custom environment variables."""
        monkeypatch.setenv("GEMINI_API_KEY", "test-secret-key-12345")
        monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
        monkeypatch.setenv("GEMINI_EMBEDDING_MODEL", "custom-embed-model")
        monkeypatch.setenv("CHUNK_SIZE", "800")
        monkeypatch.setenv("CHUNK_OVERLAP", "150")
        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "25")
        monkeypatch.setenv("CHROMA_COLLECTION_NAME", "custom_docs")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("LOG_FILE", "logs/custom.log")

        config = load_config()

        assert config.gemini_api_key == "test-secret-key-12345"
        assert config.chat_model == "gemini-1.5-flash"
        assert config.embedding_model == "custom-embed-model"
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
        assert config.max_upload_size_mb == 25
        assert config.max_upload_size_bytes == 25 * 1024 * 1024
        assert config.chroma_collection_name == "custom_docs"
        assert config.log_level == "DEBUG"
        assert config.log_file == "logs/custom.log"
        assert config.is_api_key_set()

    def test_environment_variable_aliases(self, monkeypatch, tmp_path):
        """Should support GEMINI_MODEL, GEMINI_EMBED_MODEL, and CHROMA_PERSIST_DIR aliases."""
        monkeypatch.delenv("GEMINI_CHAT_MODEL", raising=False)
        monkeypatch.delenv("GEMINI_EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("CHROMA_PERSIST_DIRECTORY", raising=False)

        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash")
        monkeypatch.setenv("GEMINI_EMBED_MODEL", "text-embedding-004")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "data/custom_chroma")

        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        config = load_config(env_path=empty_env)
        assert config.chat_model == "gemini-2.0-flash"
        assert config.embedding_model == "text-embedding-004"
        assert config.chroma_persist_dir == Path("data/custom_chroma")

    def test_invalid_max_upload_size_in_env(self, monkeypatch):
        """Should raise ValueError when MAX_UPLOAD_SIZE_MB is not integer or is <= 0."""
        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "invalid")
        with pytest.raises(ValueError, match="MAX_UPLOAD_SIZE_MB must be a valid integer"):
            load_config()

        monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "0")
        with pytest.raises(ValueError, match="MAX_UPLOAD_SIZE_MB must be a positive integer"):
            load_config()

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


class TestRetrievalConfig:
    """Tests for retrieval configuration options and validation."""

    def test_default_retrieval_values(self, monkeypatch, tmp_path):
        """Should load default retrieval settings when env vars are absent."""
        monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
        monkeypatch.delenv("RETRIEVAL_MAX_TOP_K", raising=False)
        monkeypatch.delenv("RETRIEVAL_DISTANCE_THRESHOLD", raising=False)

        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        cfg = load_config(env_path=empty_env)
        assert cfg.retrieval_top_k == DEFAULT_RETRIEVAL_TOP_K
        assert cfg.retrieval_max_top_k == DEFAULT_RETRIEVAL_MAX_TOP_K
        assert cfg.retrieval_distance_threshold is None

    def test_custom_retrieval_values(self, monkeypatch, tmp_path):
        """Should correctly parse custom retrieval settings."""
        monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
        monkeypatch.setenv("RETRIEVAL_MAX_TOP_K", "30")
        monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "0.45")

        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        cfg = load_config(env_path=empty_env)
        assert cfg.retrieval_top_k == 10
        assert cfg.retrieval_max_top_k == 30
        assert cfg.retrieval_distance_threshold == 0.45

    def test_invalid_retrieval_top_k(self, monkeypatch, tmp_path):
        """Should raise ValueError on invalid RETRIEVAL_TOP_K."""
        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        monkeypatch.setenv("RETRIEVAL_TOP_K", "not_a_number")
        with pytest.raises(ValueError, match="RETRIEVAL_TOP_K must be a valid integer"):
            load_config(env_path=empty_env)

        monkeypatch.setenv("RETRIEVAL_TOP_K", "-5")
        with pytest.raises(ValueError, match="RETRIEVAL_TOP_K must be a positive integer"):
            load_config(env_path=empty_env)

    def test_top_k_exceeding_max_top_k(self, monkeypatch, tmp_path):
        """Should raise ValueError when RETRIEVAL_TOP_K > RETRIEVAL_MAX_TOP_K."""
        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        monkeypatch.setenv("RETRIEVAL_TOP_K", "60")
        monkeypatch.setenv("RETRIEVAL_MAX_TOP_K", "50")
        with pytest.raises(ValueError, match="cannot exceed RETRIEVAL_MAX_TOP_K"):
            load_config(env_path=empty_env)

    def test_invalid_distance_threshold(self, monkeypatch, tmp_path):
        """Should raise ValueError when RETRIEVAL_DISTANCE_THRESHOLD is out of range."""
        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "2.5")
        with pytest.raises(ValueError, match="RETRIEVAL_DISTANCE_THRESHOLD must be between 0.0 and 2.0"):
            load_config(env_path=empty_env)

        monkeypatch.setenv("RETRIEVAL_DISTANCE_THRESHOLD", "-0.1")
        with pytest.raises(ValueError, match="RETRIEVAL_DISTANCE_THRESHOLD must be between 0.0 and 2.0"):
            load_config(env_path=empty_env)


class TestRAGConfig:
    """Tests for Phase 5 RAG configuration and validation."""

    def test_default_rag_values(self, monkeypatch, tmp_path):
        """Should load default RAG values when environment variables are not set."""
        monkeypatch.delenv("RAG_MAX_CONTEXT_CHARS", raising=False)
        monkeypatch.delenv("RAG_TOP_K", raising=False)

        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        cfg = load_config(env_path=empty_env)
        assert cfg.rag_max_context_chars == DEFAULT_RAG_MAX_CONTEXT_CHARS

    def test_custom_rag_context_chars(self, monkeypatch, tmp_path):
        """Should parse custom RAG_MAX_CONTEXT_CHARS and RAG_TOP_K alias."""
        monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "15000")
        monkeypatch.delenv("RETRIEVAL_TOP_K", raising=False)
        monkeypatch.setenv("RAG_TOP_K", "8")

        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        cfg = load_config(env_path=empty_env)
        assert cfg.rag_max_context_chars == 15000
        assert cfg.retrieval_top_k == 8

    def test_invalid_rag_context_chars(self, monkeypatch, tmp_path):
        """Should raise ValueError on invalid or non-positive RAG_MAX_CONTEXT_CHARS."""
        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "not_an_int")
        with pytest.raises(ValueError, match="RAG_MAX_CONTEXT_CHARS must be a valid integer"):
            load_config(env_path=empty_env)

        monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "-100")
        with pytest.raises(ValueError, match="RAG_MAX_CONTEXT_CHARS must be a positive integer"):
            load_config(env_path=empty_env)

    def test_empty_chat_model_raises_error(self, monkeypatch, tmp_path):
        """Should raise ValueError if chat model name is empty."""
        empty_env = tmp_path / "empty.env"
        empty_env.touch()

        monkeypatch.setenv("GEMINI_CHAT_MODEL", "   ")
        with pytest.raises(ValueError, match="GEMINI_CHAT_MODEL cannot be empty"):
            load_config(env_path=empty_env)


