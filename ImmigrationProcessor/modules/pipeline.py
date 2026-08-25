"""Use-case orchestration: upload -> analyse -> classify -> split -> rename -> review -> export.

The pipeline is pure Python and knows nothing about Streamlit; the UI only
feeds it :class:`UploadedDocument` objects and renders the
:class:`ProcessedDocument` objects it returns.  That keeps the presentation
layer replaceable (CLI, batch job, REST API) and the business logic testable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

from config import SETTINGS, Settings, get_logger
from modules.classifier import (
    UNKNOWN_TYPE_KEY,
    ClassifiedSegment,
    DocumentClassifier,
    ExtractionResult,
    get_spec,
    score_confidence,
)
from modules.export import ExportResult, Exporter
from modules.folder_manager import FolderManager, StorageError
from modules.ocr import DocumentText, PdfTextExtractor, TextExtractionError
from modules.pdf_splitter import PageRange, PdfSplitError, PdfSplitter, render_page_png
from modules.renamer import FileNameBuilder
from modules.validation import STATUS_READY, DocumentValidator, ValidationReport

logger = get_logger(__name__)

#: Ordered workflow stages, mirrored by the progress bar in the UI.
STAGES: tuple[str, ...] = ("Upload", "Analyze", "Classify", "Split", "Rename", "Review", "Export")

ProgressCallback = Callable[[str, float], None]


class PipelineError(RuntimeError):
    """Raised when a whole file cannot be processed."""


@dataclass(frozen=True)
class UploadedDocument:
    """A PDF handed over by the UI."""

    name: str
    data: bytes

    @property
    def size_mb(self) -> float:
        """Upload size in megabytes."""
        return len(self.data) / 1_048_576


@dataclass
class ProcessedDocument:
    """One logical document ready for review and export."""

    id: str
    source_name: str          # name of the uploaded file
    source_path: Path         # original in data/incoming
    working_path: Path        # split part in data/temp
    page_range: PageRange
    result: ExtractionResult
    report: ValidationReport
    filename: str
    approved: bool = False
    exported_path: Path | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    # -- derived ----------------------------------------------------------- #
    @property
    def status(self) -> str:
        """``READY`` or ``REVIEW REQUIRED``."""
        if self.approved:
            return STATUS_READY
        return self.report.status

    @property
    def requires_review(self) -> bool:
        """True while the document must not be exported automatically."""
        return self.report.requires_review and not self.approved

    @property
    def is_exported(self) -> bool:
        """True once the document has been filed."""
        return self.exported_path is not None

    @property
    def client_key(self) -> str:
        """``LastName_FirstName`` of the main applicant."""
        return FileNameBuilder(SETTINGS).client_key(self.result)

    @property
    def type_label(self) -> str:
        """Human readable document type."""
        return get_spec(self.result.document_type).description

    def preview_png(self, dpi: int = 110) -> bytes | None:
        """Render the first page of this document for the review screen."""
        return render_page_png(self.working_path, 1, dpi=dpi)

    def as_row(self) -> dict[str, object]:
        """Flat representation for the review data table."""
        return {
            "Status": self.status,
            "Source": self.source_name,
            "Pages": self.page_range.label(),
            "Type": self.result.document_type,
            "Confidence": f"{self.result.confidence:.0%}",
            "File name": self.filename,
            "Issues": self.report.summary(),
        }


class ProcessingPipeline:
    """Runs the full document workflow for a batch of uploads."""

    def __init__(
        self,
        settings: Settings = SETTINGS,
        extractor: PdfTextExtractor | None = None,
        classifier: DocumentClassifier | None = None,
        splitter: PdfSplitter | None = None,
        renamer: FileNameBuilder | None = None,
        folders: FolderManager | None = None,
        validator: DocumentValidator | None = None,
        exporter: Exporter | None = None,
    ) -> None:
        self._settings = settings
        self._folders = folders or FolderManager(settings)
        self._extractor = extractor or PdfTextExtractor(settings)
        self._classifier = classifier or DocumentClassifier(settings)
        self._splitter = splitter or PdfSplitter(settings)
        self._renamer = renamer or FileNameBuilder(settings)
        self._validator = validator or DocumentValidator(settings)
        self._exporter = exporter or Exporter(settings, self._folders)

    # -- properties -------------------------------------------------------- #
    @property
    def folders(self) -> FolderManager:
        """The folder manager used by this pipeline."""
        return self._folders

    @property
    def exporter(self) -> Exporter:
        """The exporter used by this pipeline."""
        return self._exporter

    @property
    def ai_enabled(self) -> bool:
        """True when an LLM fallback backs up the rule engine."""
        return self._classifier.ai_enabled

    @property
    def llm_name(self) -> str:
        """Name of the active fallback backend, or ``"rules only"``."""
        return self._classifier.llm_name

    # -- processing -------------------------------------------------------- #
    def process_batch(
        self,
        uploads: Sequence[UploadedDocument],
        progress: ProgressCallback | None = None,
    ) -> list[ProcessedDocument]:
        """Process every upload and return all detected documents.

        Args:
            uploads: The PDFs selected in the drag-and-drop area.
            progress: Optional ``callback(message, fraction)`` for the UI.

        Returns:
            All :class:`ProcessedDocument` objects of the batch, including the
            failed ones (carrying ``error``).
        """
        report = progress or (lambda message, fraction: None)
        batch_dir = self._folders.new_batch_dir()
        documents: list[ProcessedDocument] = []
        total = max(len(uploads), 1)

        logger.info("=" * 70)
        logger.info("Batch started: %d file(s) -> %s", len(uploads), batch_dir.name)

        for index, upload in enumerate(uploads):
            base = index / total
            report(f"{upload.name}: storing upload", base)
            try:
                documents.extend(
                    self._process_single(upload, batch_dir, report, base, 1 / total)
                )
            except PipelineError as exc:
                logger.error("Processing failed for %s: %s", upload.name, exc)
                documents.append(self._failed_document(upload.name, str(exc)))
            except Exception as exc:  # never let one bad file kill the batch
                logger.exception("Unexpected error while processing %s", upload.name)
                documents.append(self._failed_document(upload.name, f"unexpected error: {exc}"))

        self._propagate_within_batch(documents)

        report("Batch finished", 1.0)
        ready = sum(1 for document in documents if not document.requires_review and not document.error)
        logger.info(
            "Batch finished: %d document(s), %d ready, %d in review, %d failed",
            len(documents), ready,
            sum(1 for document in documents if document.requires_review and not document.error),
            sum(1 for document in documents if document.error),
        )
        return documents

    def _process_single(
        self,
        upload: UploadedDocument,
        batch_dir: Path,
        report: ProgressCallback,
        base: float,
        span: float,
    ) -> list[ProcessedDocument]:
        """Run the workflow for one uploaded PDF."""
        max_bytes = self._settings.max_upload_mb * 1_048_576
        if len(upload.data) > max_bytes:
            raise PipelineError(
                f"file is larger than the {self._settings.max_upload_mb} MB limit"
            )

        source_path = self._folders.save_upload(upload.name, upload.data)

        # 1) Analyse - text layer + OCR
        report(f"{upload.name}: extracting text / OCR", base + span * 0.15)
        try:
            document_text: DocumentText = self._extractor.extract(source_path)
        except TextExtractionError as exc:
            raise PipelineError(str(exc)) from exc

        if document_text.ocr_used:
            logger.info("OCR applied to %s", upload.name)

        # 2) Classify - type detection, boundary detection, field extraction
        report(f"{upload.name}: classifying", base + span * 0.45)
        segments: list[ClassifiedSegment] = self._classifier.analyze(document_text)
        if len(segments) > 1:
            logger.info(
                "%s contains %d different documents - splitting", upload.name, len(segments)
            )

        # 3) Split
        report(f"{upload.name}: splitting", base + span * 0.7)
        ranges = [PageRange(segment.start_page, segment.end_page) for segment in segments]
        try:
            parts = self._splitter.split(source_path, ranges, batch_dir, stem=source_path.stem)
        except PdfSplitError as exc:
            raise PipelineError(str(exc)) from exc

        # 4) Validate + rename
        report(f"{upload.name}: renaming", base + span * 0.9)
        documents: list[ProcessedDocument] = []
        for segment, part in zip(segments, parts):
            documents.append(self._build_document(upload.name, source_path, segment, part.path, part.page_range))
        return documents

    def _propagate_within_batch(self, documents: Sequence[ProcessedDocument]) -> None:
        """Fill in the employer from other documents of the same batch.

        A passport never names an employer, a marriage certificate neither -
        but the CV or the payslip in the same upload does, and a case worker
        would copy it across by hand. Only documents belonging to the same
        person are used, or a single employer that the whole batch agrees on.
        Every propagated value is marked in the evidence.
        """
        by_person: dict[tuple[str, str], str] = {}
        companies: set[str] = set()
        for document in documents:
            result = document.result
            if document.error or not result.company:
                continue
            companies.add(result.company)
            key = (result.last_name.lower(), result.first_name.lower())
            if all(key) and key not in by_person:
                by_person[key] = result.company

        # The batch-wide fallback is only safe when the whole upload is about one
        # person and one employer; otherwise names must match document by document.
        people = {
            (d.result.last_name.lower(), d.result.first_name.lower())
            for d in documents
            if not d.error and d.result.last_name and d.result.first_name
        }
        single_company = (
            next(iter(companies)) if len(companies) == 1 and len(people) <= 1 else ""
        )
        if not by_person and not single_company:
            return

        for document in documents:
            result = document.result
            if document.error or result.company:
                continue
            key = (result.last_name.lower(), result.first_name.lower())
            company = by_person.get(key) or (single_company if all(key) else "")
            if not company:
                continue

            result.company = company
            result.evidence.append("company from another document in this batch")
            spec = get_spec(result.document_type)
            result.confidence = score_confidence(result, spec, self._settings)
            document.report = self._validator.validate(result, spec)
            document.filename = self._renamer.build(result, spec)
            logger.info(
                "Employer %r taken from another document of this batch for %s -> %s (%.0f%%, %s)",
                company, document.source_name, document.filename,
                result.confidence * 100, document.report.status,
            )

    def _build_document(
        self,
        source_name: str,
        source_path: Path,
        segment: ClassifiedSegment,
        working_path: Path,
        page_range: PageRange,
    ) -> ProcessedDocument:
        """Validate a segment, name its file and log the outcome."""
        spec = get_spec(segment.result.document_type)
        report = self._validator.validate(segment.result, spec)
        filename = self._renamer.build(segment.result, spec)

        logger.info(
            "%s detected (%s of %s) | confidence %.0f%% | %s",
            spec.description, page_range.label(), source_name,
            segment.result.confidence * 100, report.status,
        )
        if report.issues:
            logger.info("Validation notes: %s", report.summary())
        logger.info("File renamed: %s", filename)

        return ProcessedDocument(
            id=uuid.uuid4().hex[:12],
            source_name=source_name,
            source_path=source_path,
            working_path=working_path,
            page_range=page_range,
            result=segment.result,
            report=report,
            filename=filename,
        )

    def _failed_document(self, source_name: str, message: str) -> ProcessedDocument:
        """Build a placeholder entry so failures stay visible in the UI."""
        result = ExtractionResult(document_type=UNKNOWN_TYPE_KEY, confidence=0.0, notes=message)
        report = self._validator.validate(result, get_spec(UNKNOWN_TYPE_KEY))
        return ProcessedDocument(
            id=uuid.uuid4().hex[:12],
            source_name=source_name,
            source_path=Path(source_name),
            working_path=Path(source_name),
            page_range=PageRange(1, 1),
            result=result,
            report=report,
            filename="",
            error=message,
        )

    # -- review ------------------------------------------------------------ #
    def apply_review(self, document: ProcessedDocument, changes: dict[str, object]) -> ProcessedDocument:
        """Apply manual corrections, re-validate and re-generate the file name.

        Args:
            document: The reviewed document.
            changes: Field values entered by the case worker (only the keys
                present are applied).

        Returns:
            The same (mutated) document.
        """
        result = document.result
        for key in (
            "first_name", "last_name", "company", "document_type",
            "valid_until", "city", "relationship", "dependent_name", "period", "title",
        ):
            if key in changes:
                setattr(result, key, str(changes[key] or "").strip())
        if "child_index" in changes:
            try:
                value = changes["child_index"]
                result.child_index = int(value) if value not in (None, "", 0) else None
            except (TypeError, ValueError):
                result.child_index = None
        if "confidence" in changes:
            try:
                result.confidence = float(changes["confidence"])
            except (TypeError, ValueError):
                pass

        result.source = "manual"
        spec = get_spec(result.document_type)
        document.report = self._validator.validate(result, spec)
        document.filename = self._renamer.build(result, spec)
        logger.info(
            "Manual review applied to %s -> %s (%s)",
            document.source_name, document.filename, document.report.status,
        )
        return document

    # -- export ------------------------------------------------------------ #
    def export_document(self, document: ProcessedDocument) -> Path:
        """File one document in its client folder (or in ``data/review``).

        Raises:
            PipelineError: When the document cannot be stored.
        """
        if document.error:
            raise PipelineError(f"{document.source_name} failed earlier: {document.error}")
        if document.is_exported:
            return document.exported_path  # type: ignore[return-value]
        if not document.working_path.is_file():
            raise PipelineError(f"Working file is gone: {document.working_path}")

        client_key = document.client_key
        try:
            if document.requires_review:
                target = self._folders.store_review(document.working_path, document.filename, client_key)
                logger.info("REVIEW REQUIRED: %s parked in data/review", document.filename)
            else:
                target = self._folders.store_processed(document.working_path, document.filename, client_key)
                logger.info("Exported %s to Clients/%s", document.filename, client_key)
        except StorageError as exc:
            raise PipelineError(str(exc)) from exc

        document.exported_path = target
        return target

    def export_documents(
        self, documents: Iterable[ProcessedDocument]
    ) -> tuple[list[ProcessedDocument], list[tuple[ProcessedDocument, str]]]:
        """Export many documents; returns ``(exported, failures)``."""
        exported: list[ProcessedDocument] = []
        failures: list[tuple[ProcessedDocument, str]] = []
        for document in documents:
            try:
                self.export_document(document)
                exported.append(document)
            except PipelineError as exc:
                logger.error("Export failed for %s: %s", document.filename or document.source_name, exc)
                failures.append((document, str(exc)))
        return exported, failures

    def build_client_archive(self, client_key: str) -> ExportResult:
        """Create ``<client_key>.zip`` for download."""
        return self._exporter.export_client(client_key)

    def build_full_archive(self) -> ExportResult:
        """Create an archive with every processed client."""
        return self._exporter.export_all()
