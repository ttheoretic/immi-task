"""Text extraction and OCR.

Strategy (cheapest first):

1. **PyMuPDF** reads the embedded text layer of every page.  Digital PDFs are
   fully handled here - no cloud call, no cost.
2. Pages whose text layer is empty or suspiciously short (scans, photos) are
   sent to **Azure AI Document Intelligence** (``prebuilt-read``) for OCR.
3. If Azure is not configured, an optional local Tesseract engine is used when
   installed.  Otherwise the page is reported as empty and the document is
   routed to manual review by the confidence gate.

The module never raises for a single unreadable page: it degrades and records
what happened so the UI and the log can show it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

from config import SETTINGS, Settings, get_logger

logger = get_logger(__name__)

SOURCE_PYMUPDF: Final[str] = "pymupdf"
SOURCE_AZURE: Final[str] = "azure-document-intelligence"
SOURCE_TESSERACT: Final[str] = "tesseract"
SOURCE_EMPTY: Final[str] = "empty"


class TextExtractionError(RuntimeError):
    """Raised when a PDF cannot be opened at all."""


def _import_pymupdf():
    """Import PyMuPDF under either module name (``pymupdf`` since 1.24.3)."""
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        import fitz  # PyMuPDF < 1.24.3

        return fitz


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #
@dataclass
class PageText:
    """Text of a single PDF page."""

    number: int  # 1-based
    text: str = ""
    source: str = SOURCE_EMPTY

    @property
    def char_count(self) -> int:
        """Number of non-whitespace characters on the page."""
        return len("".join(self.text.split()))

    @property
    def is_empty(self) -> bool:
        """True when the page carries no usable text."""
        return self.char_count == 0


@dataclass
class DocumentText:
    """Full text of one PDF, page by page."""

    path: Path
    pages: list[PageText] = field(default_factory=list)
    ocr_used: bool = False
    engines: set[str] = field(default_factory=set)

    @property
    def page_count(self) -> int:
        """Number of pages in the document."""
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """All pages concatenated with page markers (used for AI prompts)."""
        return "\n".join(f"[PAGE {page.number}]\n{page.text}".rstrip() for page in self.pages)

    @property
    def char_count(self) -> int:
        """Total number of non-whitespace characters."""
        return sum(page.char_count for page in self.pages)

    @property
    def is_empty(self) -> bool:
        """True when no page produced any text."""
        return self.char_count == 0

    def page(self, number: int) -> PageText:
        """Return the page with the given 1-based *number*."""
        for page in self.pages:
            if page.number == number:
                return page
        raise KeyError(f"page {number} not found in {self.path.name}")

    def text_for_range(self, start_page: int, end_page: int) -> str:
        """Return the concatenated text of an inclusive 1-based page range."""
        return "\n".join(
            page.text for page in self.pages if start_page <= page.number <= end_page
        ).strip()

    def snippet(self, max_chars: int = 12_000) -> str:
        """Return the document text truncated to *max_chars* for prompting."""
        text = self.full_text
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n[... truncated ...]"


# --------------------------------------------------------------------------- #
# OCR engines
# --------------------------------------------------------------------------- #
class AzureDocumentIntelligenceEngine:
    """Thin wrapper around the Azure AI Document Intelligence ``prebuilt-read`` model."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings.document_intelligence
        self._client: Any | None = None

    @property
    def is_available(self) -> bool:
        """True when endpoint and key are configured."""
        return self._settings.is_configured

    def _get_client(self) -> Any:
        """Create (once) and return the Azure SDK client."""
        if self._client is not None:
            return self._client
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise TextExtractionError(
                "azure-ai-documentintelligence is not installed - run 'pip install -r requirements.txt'"
            ) from exc

        self._client = DocumentIntelligenceClient(
            endpoint=self._settings.endpoint,
            credential=AzureKeyCredential(self._settings.key),
        )
        logger.info("Azure Document Intelligence client initialised (%s)", self._settings.model_id)
        return self._client

    def _begin_analyze(self, client: Any, data: bytes) -> Any:
        """Start the analysis, tolerating the different SDK call signatures."""
        model_id = self._settings.model_id
        attempts = (
            lambda: client.begin_analyze_document(model_id, body=data, content_type="application/pdf"),
            lambda: client.begin_analyze_document(model_id, data, content_type="application/pdf"),
        )
        errors: list[Exception] = []
        for attempt in attempts:
            try:
                return attempt()
            except TypeError as exc:  # signature mismatch -> try the next variant
                errors.append(exc)

        # Older preview SDKs require an explicit request object.
        try:
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

            return client.begin_analyze_document(
                model_id, AnalyzeDocumentRequest(bytes_source=data)
            )
        except Exception as exc:  # pragma: no cover - very old/new SDK
            errors.append(exc)
        raise TextExtractionError(f"Document Intelligence call failed: {errors[-1]}")

    def ocr_pdf(self, pdf_bytes: bytes) -> dict[int, str]:
        """OCR a whole PDF and return ``{page_number: text}`` (1-based)."""
        client = self._get_client()
        poller = self._begin_analyze(client, pdf_bytes)
        result = poller.result()

        pages: dict[int, str] = {}
        for index, page in enumerate(getattr(result, "pages", []) or [], start=1):
            number = int(getattr(page, "page_number", index) or index)
            lines = getattr(page, "lines", None) or []
            text = "\n".join(getattr(line, "content", "") for line in lines).strip()
            if not text:  # some models only return words
                words = getattr(page, "words", None) or []
                text = " ".join(getattr(word, "content", "") for word in words).strip()
            pages[number] = text

        if not pages and getattr(result, "content", ""):
            pages[1] = str(result.content)
        logger.info("Azure OCR returned text for %d page(s)", len(pages))
        return pages


class TesseractEngine:
    """Optional local OCR fallback (used only when ``pytesseract`` is installed)."""

    def __init__(self, dpi: int = 220) -> None:
        self._dpi = dpi
        self._checked = False
        self._available = False

    @property
    def is_available(self) -> bool:
        """True when both ``pytesseract`` and the tesseract binary are usable."""
        if self._checked:
            return self._available
        self._checked = True
        try:
            import pytesseract
            from PIL import Image

            pytesseract.get_tesseract_version()
            self._available = Image is not None
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.debug("Local OCR fallback unavailable: %s", exc)
            self._available = False
        return self._available

    def ocr_page(self, document: Any, page_index: int) -> str:
        """OCR a single already-open PyMuPDF page and return its text."""
        import pytesseract
        from PIL import Image

        page = document.load_page(page_index)
        pixmap = page.get_pixmap(dpi=self._dpi)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image, lang="deu+eng").strip()


# --------------------------------------------------------------------------- #
# Extractor
# --------------------------------------------------------------------------- #
class PdfTextExtractor:
    """Extracts the text of a PDF, running OCR on pages that need it."""

    def __init__(
        self,
        settings: Settings = SETTINGS,
        azure_engine: AzureDocumentIntelligenceEngine | None = None,
        tesseract_engine: TesseractEngine | None = None,
    ) -> None:
        self._settings = settings
        self._azure = azure_engine or AzureDocumentIntelligenceEngine(settings)
        self._tesseract = tesseract_engine or TesseractEngine()

    def extract(self, pdf_path: Path) -> DocumentText:
        """Extract the text of *pdf_path*.

        Args:
            pdf_path: Path of an existing PDF file.

        Returns:
            A :class:`DocumentText` with one :class:`PageText` per page.

        Raises:
            TextExtractionError: If the file cannot be opened as a PDF.
        """
        try:
            fitz = _import_pymupdf()
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise TextExtractionError(
                "PyMuPDF is not installed - run 'pip install -r requirements.txt'"
            ) from exc

        pdf_path = Path(pdf_path)
        document_text = DocumentText(path=pdf_path)

        try:
            document = fitz.open(pdf_path)
        except Exception as exc:
            raise TextExtractionError(f"Cannot open {pdf_path.name}: {exc}") from exc

        with document:
            for index in range(document.page_count):
                try:
                    raw = document.load_page(index).get_text("text") or ""
                except Exception as exc:  # a broken page must not kill the batch
                    logger.warning("Page %d of %s could not be read: %s", index + 1, pdf_path.name, exc)
                    raw = ""
                page = PageText(number=index + 1, text=raw.strip())
                page.source = SOURCE_PYMUPDF if page.char_count else SOURCE_EMPTY
                document_text.pages.append(page)

            if any(page.source == SOURCE_PYMUPDF for page in document_text.pages):
                document_text.engines.add(SOURCE_PYMUPDF)

            scanned = [
                page for page in document_text.pages
                if page.char_count < self._settings.ocr_min_chars_per_page
            ]
            if scanned:
                logger.info(
                    "%s: %d of %d page(s) need OCR",
                    pdf_path.name, len(scanned), document_text.page_count,
                )
                self._run_ocr(pdf_path, document, document_text, scanned)

        logger.info(
            "Extracted %d characters from %s (%d page(s), engines: %s)",
            document_text.char_count,
            pdf_path.name,
            document_text.page_count,
            ", ".join(sorted(document_text.engines)) or "none",
        )
        return document_text

    # -- internals --------------------------------------------------------- #
    def _run_ocr(
        self,
        pdf_path: Path,
        document: Any,
        document_text: DocumentText,
        scanned_pages: Sequence[PageText],
    ) -> None:
        """Fill in the text of *scanned_pages* using the best available engine."""
        if self._azure.is_available:
            try:
                ocr_pages = self._azure.ocr_pdf(pdf_path.read_bytes())
                applied = 0
                for page in scanned_pages:
                    text = (ocr_pages.get(page.number) or "").strip()
                    if text:
                        page.text = text
                        page.source = SOURCE_AZURE
                        applied += 1
                if applied:
                    document_text.ocr_used = True
                    document_text.engines.add(SOURCE_AZURE)
                    logger.info("Azure OCR filled %d page(s) of %s", applied, pdf_path.name)
                    return
                logger.warning("Azure OCR returned no text for %s", pdf_path.name)
            except Exception as exc:
                logger.error("Azure OCR failed for %s: %s", pdf_path.name, exc)

        if self._tesseract.is_available:
            applied = 0
            for page in scanned_pages:
                try:
                    text = self._tesseract.ocr_page(document, page.number - 1)
                except Exception as exc:  # pragma: no cover - optional path
                    logger.error("Local OCR failed on page %d: %s", page.number, exc)
                    continue
                if text:
                    page.text = text
                    page.source = SOURCE_TESSERACT
                    applied += 1
            if applied:
                document_text.ocr_used = True
                document_text.engines.add(SOURCE_TESSERACT)
                logger.info("Local OCR filled %d page(s) of %s", applied, pdf_path.name)
                return

        logger.warning(
            "No OCR engine available for %s - %d page(s) stay empty and will require review",
            pdf_path.name, len(scanned_pages),
        )
