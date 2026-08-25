"""Splitting of combined PDFs and page rendering, both based on PyMuPDF.

The splitter is deliberately unaware of document types: it receives plain page
ranges (produced by :mod:`modules.classifier`) and writes one PDF per range.
That keeps it reusable and trivially testable.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from config import SETTINGS, Settings, get_logger

logger = get_logger(__name__)


class PdfSplitError(RuntimeError):
    """Raised when a PDF cannot be split."""


@dataclass(frozen=True)
class PageRange:
    """An inclusive, 1-based page range of a source PDF."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise ValueError(f"invalid page range {self.start}-{self.end}")

    @property
    def page_count(self) -> int:
        """Number of pages covered by this range."""
        return self.end - self.start + 1

    def label(self) -> str:
        """Human readable label, e.g. ``p1-2``."""
        return f"p{self.start}" if self.start == self.end else f"p{self.start}-{self.end}"


@dataclass(frozen=True)
class SplitPart:
    """One physical PDF produced by the splitter."""

    path: Path
    page_range: PageRange
    is_copy_of_source: bool = False


def _import_fitz():
    """Import PyMuPDF under either module name (``pymupdf`` since 1.24.3)."""
    try:
        import pymupdf

        return pymupdf
    except ImportError:
        pass
    try:
        import fitz

        return fitz
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise PdfSplitError(
            "PyMuPDF is not installed - run 'pip install -r requirements.txt'"
        ) from exc


def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages of *pdf_path* (0 when unreadable)."""
    fitz = _import_fitz()
    try:
        with fitz.open(pdf_path) as document:
            return int(document.page_count)
    except Exception as exc:
        logger.error("Cannot read page count of %s: %s", pdf_path, exc)
        return 0


def render_page_png(pdf_path: Path, page_number: int = 1, dpi: int = 110) -> bytes | None:
    """Render one page as PNG bytes for the review preview.

    Args:
        pdf_path: PDF to render.
        page_number: 1-based page number.
        dpi: Rendering resolution.

    Returns:
        PNG bytes, or ``None`` when the page cannot be rendered.
    """
    fitz = _import_fitz()
    try:
        with fitz.open(pdf_path) as document:
            if not 1 <= page_number <= document.page_count:
                return None
            pixmap = document.load_page(page_number - 1).get_pixmap(dpi=dpi)
            return pixmap.tobytes("png")
    except Exception as exc:
        logger.error("Preview rendering failed for %s page %d: %s", pdf_path, page_number, exc)
        return None


class PdfSplitter:
    """Writes one PDF per page range into a target directory."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings

    def split(
        self,
        source: Path,
        ranges: Sequence[PageRange],
        output_dir: Path,
        *,
        stem: str | None = None,
    ) -> list[SplitPart]:
        """Split *source* into one PDF per entry of *ranges*.

        A range that covers the whole document results in a plain copy, which
        keeps the original bytes (and any signatures) untouched.

        Args:
            source: The uploaded PDF.
            ranges: Page ranges to extract, in document order.
            output_dir: Directory the parts are written to (created if needed).
            stem: Base file name for the parts; defaults to the source stem.

        Returns:
            The produced :class:`SplitPart` objects, in the order of *ranges*.

        Raises:
            PdfSplitError: If the source cannot be opened or nothing was written.
        """
        source = Path(source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_stem = stem or source.stem

        if not ranges:
            raise PdfSplitError(f"no page ranges given for {source.name}")

        total_pages = get_page_count(source)
        if total_pages <= 0:
            raise PdfSplitError(f"{source.name} has no readable pages")

        parts: list[SplitPart] = []
        single_full_range = len(ranges) == 1 and ranges[0].start == 1 and ranges[0].end >= total_pages

        if single_full_range:
            target = output_dir / f"{base_stem}.pdf"
            shutil.copy2(source, target)
            logger.info("%s contains a single document - copied to %s", source.name, target.name)
            return [SplitPart(path=target, page_range=ranges[0], is_copy_of_source=True)]

        fitz = _import_fitz()
        try:
            with fitz.open(source) as document:
                for index, page_range in enumerate(ranges, start=1):
                    start = max(1, page_range.start)
                    end = min(page_range.end, total_pages)
                    if end < start:
                        logger.warning("Skipping empty range %s of %s", page_range.label(), source.name)
                        continue
                    target = output_dir / f"{base_stem}_part{index:02d}_{page_range.label()}.pdf"
                    part_document = fitz.open()
                    try:
                        part_document.insert_pdf(document, from_page=start - 1, to_page=end - 1)
                        part_document.save(target)
                    finally:
                        part_document.close()
                    parts.append(SplitPart(path=target, page_range=PageRange(start, end)))
                    logger.info(
                        "Split %s %s -> %s", source.name, PageRange(start, end).label(), target.name
                    )
        except PdfSplitError:
            raise
        except Exception as exc:
            raise PdfSplitError(f"Splitting {source.name} failed: {exc}") from exc

        if not parts:
            raise PdfSplitError(f"Splitting {source.name} produced no output")
        return parts
