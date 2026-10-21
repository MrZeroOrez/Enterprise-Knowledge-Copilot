"""Unit tests for structured application logging and secret redaction."""

from __future__ import annotations

import logging
from pathlib import Path
import pytest

from src.logger import (
    SensitiveDataFilter,
    get_logger,
    setup_logging,
)


class TestSensitiveDataFilter:
    """Tests for the secret and credential redaction filter."""

    def test_masks_gemini_api_key(self):
        """Standard Google AI Studio / Gemini API key format (AIza...) must be masked."""
        fil = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Connecting using Gemini API key: AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q",
            args=(),
            exc_info=None,
        )
        assert fil.filter(record) is True
        assert "AIzaSy" not in record.msg
        assert "[REDACTED_API_KEY]" in record.msg

    def test_masks_bearer_token(self):
        """Authorization Bearer tokens must be masked."""
        fil = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Header Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            args=(),
            exc_info=None,
        )
        assert fil.filter(record) is True
        assert "eyJhbGci" not in record.msg
        assert "[REDACTED_TOKEN]" in record.msg

    def test_masks_param_assignments(self):
        """Query or parameter assignments containing secrets must be masked."""
        fil = SensitiveDataFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Init with api_key=super_secret_credentials_value_12345",
            args=(),
            exc_info=None,
        )
        assert fil.filter(record) is True
        assert "super_secret_credentials_value_12345" not in record.msg
        assert "[REDACTED_SECRET]" in record.msg

    def test_masks_args_tuple_and_dict(self):
        """Secrets passed inside record args (tuple or dict) must be masked."""
        fil = SensitiveDataFilter()
        record_tuple = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Key was: %s",
            args=("AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q",),
            exc_info=None,
        )
        fil.filter(record_tuple)
        assert "[REDACTED_API_KEY]" in record_tuple.args[0]

        record_dict = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="Key was: %(k)s",
            args=({"k": "AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q"},),
            exc_info=None,
        )
        fil.filter(record_dict)
        assert "[REDACTED_API_KEY]" in record_dict.args["k"]


class TestSetupLogging:
    """Tests for setup_logging lifecycle and file rotation."""

    def test_setup_logging_creates_file_and_writes_logs(self, tmp_path):
        """setup_logging should create directory and log file, and write log entries."""
        log_dir = tmp_path / "custom_logs"
        log_file = "test_run.log"

        root = setup_logging(
            log_level="DEBUG",
            log_dir=log_dir,
            log_file=log_file,
            force_reconfigure=True,
        )

        test_logger = get_logger("copilot.test")
        test_logger.info("Verification message for file logger test.")

        # Flush handlers
        for h in root.handlers:
            h.flush()

        expected_file = log_dir / log_file
        assert expected_file.exists()
        content = expected_file.read_text(encoding="utf-8")
        assert "Verification message for file logger test." in content
        assert "copilot.test" in content

    def test_setup_logging_idempotence(self, tmp_path):
        """Calling setup_logging multiple times should not add duplicate handlers."""
        log_dir = tmp_path / "idempotent_logs"
        log_file = "app.log"

        root1 = setup_logging(
            log_level="INFO",
            log_dir=log_dir,
            log_file=log_file,
            force_reconfigure=True,
        )
        initial_handlers_count = len(root1.handlers)

        # Call setup_logging again without force_reconfigure
        root2 = setup_logging(
            log_level="INFO",
            log_dir=log_dir,
            log_file=log_file,
            force_reconfigure=False,
        )
        assert len(root2.handlers) == initial_handlers_count

    def test_get_logger_returns_named_logger(self):
        """get_logger should return a logger instance with the requested module name."""
        lg = get_logger("my_unique_module_name")
        assert lg.name == "my_unique_module_name"
