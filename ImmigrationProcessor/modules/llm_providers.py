"""LLM backends used as a fallback behind the deterministic rule engine.

Two providers implement the same contract:

``OllamaExtractor``
    Talks to a locally running `Ollama <https://ollama.com>`_ server over plain
    HTTP (stdlib only, no extra dependency). The model runs on your own
    hardware: no API contract, no per-token cost, and no client data leaving
    the machine - which is usually the deciding argument for immigration files.

``AzureOpenAIExtractor``
    Talks to an Azure OpenAI chat deployment.

Both share everything except the actual completion call: prompt assembly, JSON
repair, and the mapping of the answer onto page ranges live in
:class:`LLMDocumentExtractor`.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from config import SETTINGS, Settings, get_logger
from modules.classifier import (
    DOCUMENT_TYPES,
    ClassificationError,
    ClassifiedSegment,
    ExtractionResult,
)
from modules.ocr import DocumentText

logger = get_logger(__name__)

PROMPT_FILE = "document_classifier.txt"


# --------------------------------------------------------------------------- #
# Shared base
# --------------------------------------------------------------------------- #
class LLMDocumentExtractor(ABC):
    """Common behaviour of every LLM backed document extractor."""

    #: Value written into ``ExtractionResult.source``.
    source_name: str = "llm"
    #: Human readable backend name for logs and the UI.
    display_name: str = "LLM"

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._prompt: str | None = None

    # -- contract ---------------------------------------------------------- #
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True when this backend is configured and usable."""

    @abstractmethod
    def _complete(self, system_prompt: str, user_message: str) -> str:
        """Send one chat completion and return the raw answer text.

        Raises:
            ClassificationError: On any transport or backend level failure.
        """

    # -- public API -------------------------------------------------------- #
    def analyze(self, document: DocumentText) -> list[ClassifiedSegment]:
        """Return one :class:`ClassifiedSegment` per logical document.

        Raises:
            ClassificationError: When the backend is unavailable or its answer
                cannot be mapped onto page ranges.
        """
        if not self.is_available:
            raise ClassificationError(f"{self.display_name} is not available")
        if document.is_empty:
            raise ClassificationError("document contains no text to classify")

        payload = self._call_with_retry(self._build_user_message(document))
        return self._segments_from_payload(payload, document)

    def analyze_single(
        self, text: str, page_count: int = 1, hint: str = ""
    ) -> ExtractionResult | None:
        """Extract the fields of exactly ONE document from *text*.

        Used by the rules-first pipeline: the boundaries are already known, only
        the field extraction of a single weak segment is delegated to the model.

        Returns:
            The extracted result, or ``None`` when the model produced nothing
            usable.
        """
        if not self.is_available or not text.strip():
            return None
        message = (
            "This text contains exactly ONE document"
            f"{f' ({hint})' if hint else ''} on {page_count} page(s). "
            "Return it as a single entry in the documents array with "
            f"start_page 1 and end_page {page_count}.\n\n{text[:12000]}"
        )
        try:
            payload = self._call_with_retry(message, attempts=2)
        except ClassificationError as exc:
            logger.warning("%s single-document extraction failed: %s", self.display_name, exc)
            return None

        documents = payload.get("documents")
        entry: Any = None
        if isinstance(documents, list) and documents:
            entry = documents[0]
        elif isinstance(payload.get("document_type"), str):
            entry = payload  # some models answer with a bare object
        if not isinstance(entry, Mapping):
            return None
        return ExtractionResult.from_payload(entry, source=self.source_name)

    # -- prompt ------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        """Load (and cache) the system prompt, injecting the type catalogue."""
        if self._prompt is not None:
            return self._prompt
        prompt_path: Path = self._settings.paths.prompts / PROMPT_FILE
        try:
            raw = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ClassificationError(f"Prompt file missing: {prompt_path} ({exc})") from exc
        self._prompt = raw.replace("{{DOCUMENT_TYPES}}", self._catalogue_block())
        return self._prompt

    @staticmethod
    def _catalogue_block() -> str:
        """Render the allowed document types for the prompt."""
        lines = []
        for key, spec in sorted(DOCUMENT_TYPES.items(), key=lambda item: item[1].label_template):
            extras = []
            if spec.requires_validity:
                extras.append("needs valid_until")
            if spec.requires_period:
                extras.append("needs period (YYYY-MM)")
            if spec.requires_city:
                extras.append("needs city")
            suffix = f" [{', '.join(extras)}]" if extras else ""
            lines.append(f'- "{key}": {spec.description}{suffix}')
        return "\n".join(lines)

    def _build_user_message(self, document: DocumentText) -> str:
        """Render the user message: page-marked document text."""
        return (
            f"File name: {document.path.name}\n"
            f"Total pages: {document.page_count}\n"
            f"OCR used: {'yes' if document.ocr_used else 'no'}\n\n"
            "Document text (page markers included):\n"
            f"{document.snippet()}"
        )

    # -- answer handling --------------------------------------------------- #
    def _call_with_retry(self, user_message: str, attempts: int = 3) -> dict[str, Any]:
        """Call the backend and return the parsed JSON answer."""
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self._parse_json(self._complete(self._system_prompt(), user_message))
            except ClassificationError as exc:
                last_error = exc
                logger.warning(
                    "%s call failed (attempt %d/%d): %s", self.display_name, attempt, attempts, exc
                )
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 8))
        raise ClassificationError(f"{self.display_name} request failed: {last_error}")

    def _segments_from_payload(
        self, payload: Mapping[str, Any], document: DocumentText
    ) -> list[ClassifiedSegment]:
        """Map a model answer onto validated page ranges."""
        documents = payload.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ClassificationError("model returned no documents")

        segments: list[ClassifiedSegment] = []
        for entry in documents:
            if not isinstance(entry, Mapping):
                continue
            start = self._coerce_page(entry.get("start_page"), 1, document.page_count)
            end = self._coerce_page(entry.get("end_page"), start, document.page_count)
            if end < start:
                start, end = end, start
            result = ExtractionResult.from_payload(entry, source=self.source_name)
            segments.append(ClassifiedSegment(start_page=start, end_page=end, result=result))

        if not segments:
            raise ClassificationError("model answer could not be mapped to page ranges")
        logger.info(
            "%s detected %d document(s) in %s: %s",
            self.display_name, len(segments), document.path.name,
            ", ".join(
                f"{s.page_label}={s.result.document_type} ({s.result.confidence:.0%})"
                for s in segments
            ),
        )
        return segments

    @staticmethod
    def _coerce_page(value: Any, default: int, maximum: int) -> int:
        """Clamp a page number returned by the model into the valid range."""
        try:
            page = int(value)
        except (TypeError, ValueError):
            page = default
        return max(1, min(page, max(maximum, 1)))

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse a JSON object out of the answer (tolerates fences and prose)."""
        if not content or not content.strip():
            raise ClassificationError("empty model answer")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise ClassificationError(f"model answer is not JSON: {cleaned[:200]}")
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ClassificationError(f"model answer is not JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ClassificationError("model answer is not a JSON object")
        return payload


# --------------------------------------------------------------------------- #
# Ollama (local, no API key, no data leaving the machine)
# --------------------------------------------------------------------------- #
class OllamaExtractor(LLMDocumentExtractor):
    """Chat completions against a local Ollama server."""

    source_name = "ollama"

    def __init__(self, settings: Settings = SETTINGS) -> None:
        super().__init__(settings)
        self._config = settings.ollama
        self._reachable: bool | None = None
        self.display_name = f"Ollama ({self._config.model})"

    # -- availability ------------------------------------------------------ #
    @property
    def is_available(self) -> bool:
        """True when the server answers and the configured model is present."""
        if not self._config.is_configured:
            return False
        if self._reachable is None:
            self._reachable = self._check_server()
        return self._reachable

    def installed_models(self, timeout: float = 3.0) -> list[str]:
        """Return the model names the local server has pulled."""
        try:
            with urllib.request.urlopen(self._config.tags_url, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as exc:
            logger.debug("Ollama not reachable at %s: %s", self._config.host, exc)
            return []
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        return [str(item.get("name", "")) for item in models if isinstance(item, Mapping)]

    def _check_server(self) -> bool:
        """Health check: server up, and the configured model pulled."""
        models = self.installed_models()
        if not models:
            logger.warning(
                "Ollama is not reachable at %s - falling back to the rule engine only",
                self._config.host,
            )
            return False
        wanted = self._config.model
        # Ollama reports "name:tag"; accept a bare name as a prefix match.
        if not any(name == wanted or name.split(":")[0] == wanted.split(":")[0] for name in models):
            logger.warning(
                "Ollama at %s does not have model %r (available: %s). Run: ollama pull %s",
                self._config.host, wanted, ", ".join(models) or "none", wanted,
            )
            return False
        logger.info("Ollama ready at %s (model: %s)", self._config.host, wanted)
        return True

    def refresh(self) -> bool:
        """Re-run the health check, e.g. after the user started Ollama."""
        self._reachable = None
        return self.is_available

    # -- completion -------------------------------------------------------- #
    def _complete(self, system_prompt: str, user_message: str) -> str:
        """POST one chat request to ``/api/chat`` and return the answer text."""
        body = json.dumps(
            {
                "model": self._config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "format": "json",  # ask Ollama to constrain the output to JSON
                "keep_alive": self._config.keep_alive,
                "options": {
                    "temperature": self._config.temperature,
                    "num_ctx": self._config.num_ctx,
                },
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            self._config.chat_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._config.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ClassificationError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ClassificationError(f"Ollama unreachable at {self._config.host}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ClassificationError(f"Ollama sent invalid JSON: {exc}") from exc

        message = payload.get("message")
        if isinstance(message, Mapping):
            return str(message.get("content") or "")
        if isinstance(payload.get("response"), str):  # /api/generate shape
            return payload["response"]
        raise ClassificationError("Ollama answer contained no message content")


# --------------------------------------------------------------------------- #
# Azure OpenAI (cloud)
# --------------------------------------------------------------------------- #
class AzureOpenAIExtractor(LLMDocumentExtractor):
    """Chat completions against an Azure OpenAI deployment."""

    source_name = "azure_openai"

    def __init__(self, settings: Settings = SETTINGS) -> None:
        super().__init__(settings)
        self._config = settings.azure_openai
        self._client: Any | None = None
        self.display_name = f"Azure OpenAI ({self._config.deployment or 'no deployment'})"

    @property
    def is_available(self) -> bool:
        """True when endpoint, key and deployment are configured."""
        return self._config.is_configured

    def _get_client(self) -> Any:
        """Create (once) and return the Azure OpenAI client."""
        if self._client is not None:
            return self._client
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise ClassificationError(
                "openai is not installed - run 'pip install -r requirements.txt'"
            ) from exc

        self._client = AzureOpenAI(
            api_key=self._config.api_key,
            azure_endpoint=self._config.endpoint,
            api_version=self._config.api_version,
            timeout=self._config.timeout,
        )
        logger.info("Azure OpenAI client initialised (deployment: %s)", self._config.deployment)
        return self._client

    def _complete(self, system_prompt: str, user_message: str) -> str:
        """Send one chat completion request to Azure OpenAI."""
        try:
            response = self._get_client().chat.completions.create(
                model=self._config.deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                response_format={"type": "json_object"},
            )
        except ClassificationError:
            raise
        except Exception as exc:  # network, rate limit, SDK errors
            raise ClassificationError(str(exc)) from exc
        return (response.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_llm_extractor(settings: Settings = SETTINGS) -> LLMDocumentExtractor | None:
    """Return the configured LLM fallback, or ``None`` when rules run alone.

    Honours ``LLM_PROVIDER`` (``auto`` | ``ollama`` | ``azure`` | ``none``); in
    ``auto`` mode a reachable local Ollama wins over the cloud.
    """
    provider = settings.active_llm_provider
    if provider == "ollama":
        extractor = OllamaExtractor(settings)
        if extractor.is_available:
            return extractor
        # Configured but not running: fall through to Azure when allowed.
        if settings.llm_provider == "auto" and settings.azure_openai.is_configured:
            logger.info("Ollama unavailable - using Azure OpenAI as the fallback backend")
            return AzureOpenAIExtractor(settings)
        return None
    if provider == "azure":
        return AzureOpenAIExtractor(settings)
    return None
