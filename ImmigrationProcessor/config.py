"""Central configuration for the Immigration Document Processor.

This module is the single source of truth for:

* environment driven settings (Azure OpenAI / Azure Document Intelligence),
* filesystem layout (``data/``, ``logs/``, ``Clients/``),
* business thresholds (confidence threshold, OCR trigger),
* application wide logging configuration.

Nothing in this module imports application code, which keeps the dependency
graph acyclic: ``config`` -> (nothing), ``modules.*`` -> ``config``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ENV_FILE: Final[Path] = BASE_DIR / ".env"


# --------------------------------------------------------------------------- #
# .env loading
# --------------------------------------------------------------------------- #
def _load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from *path* into ``os.environ``.

    ``python-dotenv`` is used when available; a minimal parser is used as a
    fallback so that the application also starts on a bare interpreter.
    Existing environment variables always win over the file.
    """
    if not path.is_file():
        return
    try:  # preferred path - handles quoting/escaping edge cases
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:  # pragma: no cover - only hit without python-dotenv
        pass

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(ENV_FILE)


def _env(name: str, default: str = "") -> str:
    """Return a trimmed environment variable."""
    return (os.getenv(name) or default).strip()


def _env_float(name: str, default: float) -> float:
    """Return a float environment variable, falling back on parse errors."""
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Return an int environment variable, falling back on parse errors."""
    try:
        return int(float(_env(name) or default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment variable (``1/true/yes/on``)."""
    value = _env(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Settings objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AzureOpenAISettings:
    """Credentials and tuning parameters for Azure OpenAI chat completions."""

    api_key: str
    endpoint: str
    deployment: str
    api_version: str
    temperature: float
    max_tokens: int
    timeout: float

    @property
    def is_configured(self) -> bool:
        """True when all mandatory Azure OpenAI variables are present."""
        return bool(self.api_key and self.endpoint and self.deployment)

    @classmethod
    def load(cls) -> "AzureOpenAISettings":
        return cls(
            api_key=_env("AZURE_OPENAI_API_KEY"),
            endpoint=_env("AZURE_OPENAI_ENDPOINT"),
            deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
            api_version=_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            temperature=_env_float("AZURE_OPENAI_TEMPERATURE", 0.0),
            max_tokens=_env_int("AZURE_OPENAI_MAX_TOKENS", 2000),
            timeout=_env_float("AZURE_OPENAI_TIMEOUT_SECONDS", 90.0),
        )


@dataclass(frozen=True)
class DocumentIntelligenceSettings:
    """Credentials for Azure AI Document Intelligence (OCR)."""

    endpoint: str
    key: str
    model_id: str

    @property
    def is_configured(self) -> bool:
        """True when endpoint and key are present."""
        return bool(self.endpoint and self.key)

    @classmethod
    def load(cls) -> "DocumentIntelligenceSettings":
        return cls(
            endpoint=_env("DOCUMENT_INTELLIGENCE_ENDPOINT"),
            key=_env("DOCUMENT_INTELLIGENCE_KEY"),
            model_id=_env("DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-read"),
        )


@dataclass(frozen=True)
class Paths:
    """Filesystem layout of the application."""

    base: Path
    data: Path
    incoming: Path
    processed: Path
    review: Path
    temp: Path
    logs: Path
    prompts: Path
    clients: Path
    log_file: Path

    @classmethod
    def load(cls, base: Path = BASE_DIR) -> "Paths":
        data = base / "data"
        processed = data / "processed"
        clients_raw = _env("CLIENTS_ROOT")
        clients = Path(clients_raw).expanduser() if clients_raw else processed / "Clients"
        if not clients.is_absolute():
            clients = (base / clients).resolve()
        logs = base / "logs"
        return cls(
            base=base,
            data=data,
            incoming=data / "incoming",
            processed=processed,
            review=data / "review",
            temp=data / "temp",
            logs=logs,
            prompts=base / "prompts",
            clients=clients,
            log_file=logs / "processing.log",
        )

    def ensure(self) -> None:
        """Create every managed directory if it does not exist yet."""
        for directory in (
            self.data,
            self.incoming,
            self.processed,
            self.review,
            self.temp,
            self.logs,
            self.prompts,
            self.clients,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Aggregate application settings."""

    azure_openai: AzureOpenAISettings
    document_intelligence: DocumentIntelligenceSettings
    paths: Paths
    confidence_threshold: float
    heuristic_confidence_cap: float
    ocr_min_chars_per_page: int
    max_upload_mb: int
    log_level: str
    unknown_token: str

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            azure_openai=AzureOpenAISettings.load(),
            document_intelligence=DocumentIntelligenceSettings.load(),
            paths=Paths.load(),
            confidence_threshold=_env_float("CONFIDENCE_THRESHOLD", 0.90),
            # Heuristic (non-AI) results can never reach the auto-export
            # threshold: a human always reviews them.
            heuristic_confidence_cap=_env_float("HEURISTIC_CONFIDENCE_CAP", 0.55),
            ocr_min_chars_per_page=_env_int("OCR_MIN_CHARS_PER_PAGE", 60),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            unknown_token=_env("UNKNOWN_TOKEN", "UNKNOWN"),
        )

    @property
    def ai_enabled(self) -> bool:
        """True when Azure OpenAI extraction is available."""
        return self.azure_openai.is_configured

    @property
    def ocr_enabled(self) -> bool:
        """True when Azure Document Intelligence OCR is available."""
        return self.document_intelligence.is_configured


SETTINGS: Final[Settings] = Settings.load()
SETTINGS.paths.ensure()


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M"
_logging_configured = False


def configure_logging(settings: Settings = SETTINGS) -> None:
    """Attach a rotating file handler and a console handler to the root logger.

    Safe to call repeatedly (Streamlit re-executes the script on every
    interaction); handlers are installed only once.
    """
    global _logging_configured
    if _logging_configured:
        return

    settings.paths.logs.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        settings.paths.log_file, maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Third party libraries are chatty at INFO level.
    for noisy in ("azure", "urllib3", "httpx", "openai", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*."""
    configure_logging()
    return logging.getLogger(name)
