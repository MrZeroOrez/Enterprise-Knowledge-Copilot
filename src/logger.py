"""Structured logging infrastructure for Enterprise Knowledge Copilot.

Provides:
- Rotating file handler with automatic directory creation and size caps.
- Console stream handler with formatted output.
- Secret and API key redaction filter to prevent credential leakage.
- Idempotent configuration safe for Streamlit re-runs.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import sys
from typing import Any

# Regex to detect Google AI Studio / Gemini API keys and general token formats
GEMINI_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z-_]{35}")
BEARER_TOKEN_PATTERN = re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{20,}", re.IGNORECASE)
PARAM_KEY_PATTERN = re.compile(
    r"(api[-_]?key|secret|token|password)[=:]\s*['\"]?([A-Za-z0-9_\-]{16,})['\"]?",
    re.IGNORECASE,
)

# Marker attribute to avoid duplicating handlers across Streamlit reruns
_COPILOT_HANDLER_MARKER = "_copilot_configured"


class SensitiveDataFilter(logging.Filter):
    """Logging filter that redacts API keys, credentials, and tokens from log records."""

    def __init__(self, name: str = "", extra_patterns: list[str] | None = None) -> None:
        super().__init__(name)
        self.extra_patterns = extra_patterns or []

    def _redact_text(self, text: str) -> str:
        """Sanitize text by replacing sensitive patterns with redaction placeholders."""
        if not text:
            return text

        # 1. Redact explicit Gemini / Google API key pattern
        text = GEMINI_KEY_PATTERN.sub("[REDACTED_API_KEY]", text)

        # 2. Redact Bearer tokens
        text = BEARER_TOKEN_PATTERN.sub(r"\1[REDACTED_TOKEN]", text)

        # 3. Redact param assignments (e.g., api_key=xyz)
        text = PARAM_KEY_PATTERN.sub(r"\1=[REDACTED_SECRET]", text)

        # 4. Redact any known exact secrets from environment (e.g. GEMINI_API_KEY)
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key and len(env_key) >= 8 and env_key != "your_gemini_api_key_here":
            text = text.replace(env_key, "[REDACTED_API_KEY]")

        for pattern in self.extra_patterns:
            if pattern and len(pattern) >= 8:
                text = text.replace(pattern, "[REDACTED]")

        return text

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact record message, arguments, and exception info."""
        try:
            if isinstance(record.msg, str):
                record.msg = self._redact_text(record.msg)

            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        k: self._redact_text(str(v)) if isinstance(v, str) else v
                        for k, v in record.args.items()
                    }
                elif isinstance(record.args, (tuple, list)):
                    new_args = []
                    for a in record.args:
                        if isinstance(a, str):
                            new_args.append(self._redact_text(a))
                        elif isinstance(a, dict):
                            new_args.append({
                                k: self._redact_text(str(v)) if isinstance(v, str) else v
                                for k, v in a.items()
                            })
                        else:
                            new_args.append(a)
                    record.args = tuple(new_args) if isinstance(record.args, tuple) else new_args

            if record.exc_text:
                record.exc_text = self._redact_text(record.exc_text)

        except Exception:
            # Filtering must never break logging
            pass

        return True


def setup_logging(
    log_level: str = "INFO",
    log_dir: str | Path = "logs",
    log_file: str = "app.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 5,
    force_reconfigure: bool = False,
) -> logging.Logger:
    """Configure structured file and console logging for the application.

    Idempotent: Avoids duplicate handlers when called across multiple Streamlit
    script re-runs unless `force_reconfigure` is True.

    Args:
        log_level: Minimum severity level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_dir: Directory where log files are stored.
        log_file: Base name of the log file.
        max_bytes: Maximum size in bytes before rotating the log file (default: 5MB).
        backup_count: Number of rotated backup log files to keep (default: 5).
        force_reconfigure: If True, clears existing handlers and reattaches them.

    Returns:
        The configured root logger.
    """
    root_logger = logging.getLogger()
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(resolved_level)

    # Check if handlers have already been configured
    existing_copilot_handlers = [
        h for h in root_logger.handlers if getattr(h, _COPILOT_HANDLER_MARKER, False)
    ]

    if existing_copilot_handlers and not force_reconfigure:
        # Already configured; update level and return
        for h in existing_copilot_handlers:
            h.setLevel(resolved_level)
        return root_logger

    if force_reconfigure:
        for h in list(root_logger.handlers):
            if getattr(h, _COPILOT_HANDLER_MARKER, False):
                root_logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

    # Ensure log directory exists
    dir_path = Path(log_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / log_file

    redaction_filter = SensitiveDataFilter()

    # 1. Rotating File Handler
    file_handler = RotatingFileHandler(
        filename=str(file_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | [%(name)s:%(lineno)d] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(redaction_filter)
    setattr(file_handler, _COPILOT_HANDLER_MARKER, True)
    root_logger.addHandler(file_handler)

    # 2. Console Stream Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(resolved_level)
    stream_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    stream_handler.setFormatter(stream_formatter)
    stream_handler.addFilter(redaction_filter)
    setattr(stream_handler, _COPILOT_HANDLER_MARKER, True)
    root_logger.addHandler(stream_handler)

    root_logger.info(
        "Application logging initialized. Log file: %s (level=%s, max_bytes=%d, backups=%d)",
        file_path,
        logging.getLevelName(resolved_level),
        max_bytes,
        backup_count,
    )

    return root_logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Obtain a logger instance by name, ensuring logging is configured.

    Args:
        name: Logger name (typically __name__).

    Returns:
        logging.Logger instance.
    """
    root_logger = logging.getLogger()
    if not any(getattr(h, _COPILOT_HANDLER_MARKER, False) for h in root_logger.handlers):
        setup_logging()
    return logging.getLogger(name)
