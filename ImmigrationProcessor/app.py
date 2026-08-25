"""Immigration Document Processor - Streamlit presentation layer.

Workflow implemented by this screen::

    Upload -> Analyze -> Classify -> Split -> Rename -> Review -> Export

The module contains no business logic: it collects uploads, hands them to
:class:`modules.pipeline.ProcessingPipeline`, renders the results for review and
triggers the export.  Run it with::

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import streamlit as st

from config import SETTINGS, configure_logging, get_logger
from modules.classifier import DOCUMENT_TYPES, UNKNOWN_TYPE_KEY, type_choices
from modules.export import ExportError, ExportResult
from modules.pipeline import (
    STAGES,
    ProcessedDocument,
    ProcessingPipeline,
    UploadedDocument,
)
from modules.validation import STATUS_READY

configure_logging()
logger = get_logger(__name__)

APP_TITLE = "Immigration Document Processor"
RELATIONSHIP_OPTIONS = ["", "spouse", "child"]


# --------------------------------------------------------------------------- #
# Bootstrapping
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_pipeline() -> ProcessingPipeline:
    """Create the processing pipeline once per Streamlit session process."""
    logger.info("Initialising processing pipeline")
    return ProcessingPipeline(SETTINGS)


def init_state() -> None:
    """Ensure every session state key used by the UI exists."""
    defaults: dict[str, Any] = {
        "documents": [],          # list[ProcessedDocument]
        "batch_finished": False,
        "last_archive": None,     # ExportResult
        "export_summary": None,   # tuple[int, int]
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def documents() -> list[ProcessedDocument]:
    """Return the documents of the current batch."""
    return st.session_state["documents"]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(pipeline: ProcessingPipeline) -> None:
    """Render the configuration status, the client overview and the log tail."""
    with st.sidebar:
        st.header("System status")

        st.success("Rule engine: active (first stage, no data leaves this machine)")

        if pipeline.ai_enabled:
            st.success(f"LLM fallback: {pipeline.llm_name}")
        else:
            st.warning("No LLM fallback - unresolved documents go to review")

        if SETTINGS.ocr_enabled:
            st.success(f"OCR: Document Intelligence ({SETTINGS.document_intelligence.model_id})")
        else:
            st.info("OCR: local only (tesseract, if installed)")

        st.caption(
            f"Confidence threshold: **{SETTINGS.confidence_threshold:.0%}** - anything below is "
            "parked in `data/review`. The rule engine only exceeds it with hard evidence "
            "(verified passport MRZ, or every field read from a labelled anchor)."
        )

        st.divider()
        st.subheader("Clients")
        clients = pipeline.folders.list_clients()
        if not clients:
            st.caption("No processed clients yet.")
        else:
            for client in clients:
                st.write(f"**{client.display_name}** - {client.document_count} document(s)")

        review_files = pipeline.folders.list_review_files()
        if review_files:
            st.warning(f"{len(review_files)} document(s) waiting in data/review")

        st.divider()
        with st.expander("Processing log (last 40 lines)"):
            st.code(read_log_tail(40) or "log is empty", language="log")

        if st.button("Clean temp folder", use_container_width=True):
            removed = pipeline.folders.cleanup_temp(older_than_hours=0)
            st.toast(f"Removed {removed} temporary folder(s)")


def read_log_tail(lines: int = 40) -> str:
    """Return the last *lines* of the processing log."""
    log_file: Path = SETTINGS.paths.log_file
    if not log_file.is_file():
        return ""
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:  # pragma: no cover - unreadable log
        return f"log unavailable: {exc}"
    return "\n".join(content[-lines:])


# --------------------------------------------------------------------------- #
# Step 1 - upload and processing
# --------------------------------------------------------------------------- #
def render_upload(pipeline: ProcessingPipeline) -> None:
    """Render the drag-and-drop area and start the batch."""
    st.subheader("1. Upload")
    st.caption("Drag & drop one or many client PDFs - combined PDFs are split automatically.")

    uploads = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximum {SETTINGS.max_upload_mb} MB per file.",
    )

    columns = st.columns([1, 1, 2])
    start = columns[0].button(
        "Analyze documents", type="primary", disabled=not uploads, use_container_width=True
    )
    if columns[1].button("Reset batch", disabled=not documents(), use_container_width=True):
        st.session_state["documents"] = []
        st.session_state["batch_finished"] = False
        st.session_state["export_summary"] = None
        st.rerun()

    if uploads:
        columns[2].caption(
            f"{len(uploads)} file(s) selected - "
            f"{sum(len(item.getvalue()) for item in uploads) / 1_048_576:.1f} MB total"
        )

    if start and uploads:
        run_batch(pipeline, uploads)


def run_batch(pipeline: ProcessingPipeline, uploads: Sequence[Any]) -> None:
    """Execute the pipeline for the uploaded files with live progress."""
    payload = [UploadedDocument(name=item.name, data=item.getvalue()) for item in uploads]

    progress_bar = st.progress(0.0, text="Starting ...")
    status = st.empty()

    def report(message: str, fraction: float) -> None:
        progress_bar.progress(min(max(fraction, 0.0), 1.0), text=message)
        status.info(message)

    with st.spinner("Processing documents ..."):
        results = pipeline.process_batch(payload, progress=report)

    progress_bar.progress(1.0, text="Done")
    status.empty()

    st.session_state["documents"] = results
    st.session_state["batch_finished"] = True
    st.session_state["export_summary"] = None
    st.rerun()


# --------------------------------------------------------------------------- #
# Step 2 - review
# --------------------------------------------------------------------------- #
def render_summary(items: Sequence[ProcessedDocument]) -> None:
    """Render the KPI row and the overview table."""
    ready = [item for item in items if not item.requires_review and not item.error]
    review = [item for item in items if item.requires_review and not item.error]
    failed = [item for item in items if item.error]

    columns = st.columns(4)
    columns[0].metric("Documents detected", len(items))
    columns[1].metric("Ready to export", len(ready))
    columns[2].metric("Review required", len(review))
    columns[3].metric("Failed", len(failed))

    st.dataframe(
        [item.as_row() for item in items],
        use_container_width=True,
        hide_index=True,
    )


def render_document_card(pipeline: ProcessingPipeline, document: ProcessedDocument, index: int) -> None:
    """Render preview, editable fields and the approval checkbox for one document."""
    if document.error:
        st.error(f"**{document.source_name}** could not be processed: {document.error}")
        return

    icon = "✅" if not document.requires_review else "⚠️"
    if document.is_exported:
        icon = "📁"
    title = f"{icon} {document.filename or document.source_name} - {document.status}"

    with st.expander(title, expanded=document.requires_review and not document.is_exported):
        left, right = st.columns([1, 2], gap="medium")

        with left:
            preview = document.preview_png()
            if preview:
                st.image(preview, caption=f"{document.source_name} ({document.page_range.label()})",
                         use_container_width=True)
            else:
                st.info("No preview available")
            st.metric("Confidence", f"{document.result.confidence:.0%}")
            st.caption(f"Extraction source: {document.result.source}")
            if document.result.evidence:
                with st.expander("Why these values?"):
                    for item in document.result.evidence:
                        st.caption(f"- {item}")
            if document.result.notes:
                st.caption(document.result.notes)
            if document.is_exported and document.exported_path:
                st.success(f"Stored: {document.exported_path.name}")

        with right:
            if document.report.errors:
                st.error(
                    "REVIEW REQUIRED - "
                    + "; ".join(f"{issue.field_name}: {issue.message}" for issue in document.report.errors)
                )
            for warning in document.report.warnings:
                st.warning(f"{warning.field_name}: {warning.message}")

            with st.form(key=f"form_{document.id}"):
                type_options = type_choices()
                current_type = document.result.document_type or UNKNOWN_TYPE_KEY
                if current_type not in type_options:
                    current_type = UNKNOWN_TYPE_KEY

                fields: dict[str, Any] = {}
                fields["document_type"] = st.selectbox(
                    "Document type",
                    options=type_options,
                    index=type_options.index(current_type),
                    format_func=lambda key: (
                        "-- unclassified --" if key == UNKNOWN_TYPE_KEY
                        else f"{DOCUMENT_TYPES[key].label_template} ({DOCUMENT_TYPES[key].description})"
                    ),
                    key=f"type_{document.id}",
                )

                name_columns = st.columns(2)
                fields["last_name"] = name_columns[0].text_input(
                    "Last name (main applicant)", value=document.result.last_name, key=f"last_{document.id}"
                )
                fields["first_name"] = name_columns[1].text_input(
                    "First name (main applicant)", value=document.result.first_name, key=f"first_{document.id}"
                )

                detail_columns = st.columns(2)
                fields["company"] = detail_columns[0].text_input(
                    "Company", value=document.result.company, key=f"company_{document.id}"
                )
                fields["valid_until"] = detail_columns[1].text_input(
                    "Valid until (DD.MM.YYYY)", value=document.result.valid_until, key=f"valid_{document.id}"
                )

                extra_columns = st.columns(2)
                fields["city"] = extra_columns[0].text_input(
                    "City", value=document.result.city, key=f"city_{document.id}"
                )
                fields["period"] = extra_columns[1].text_input(
                    "Payslip period (YYYY-MM)", value=document.result.period, key=f"period_{document.id}"
                )

                dependent_columns = st.columns([1, 1, 1])
                fields["relationship"] = dependent_columns[0].selectbox(
                    "Relationship",
                    options=RELATIONSHIP_OPTIONS,
                    index=RELATIONSHIP_OPTIONS.index(document.result.relationship)
                    if document.result.relationship in RELATIONSHIP_OPTIONS else 0,
                    format_func=lambda value: value or "main applicant",
                    key=f"rel_{document.id}",
                )
                fields["dependent_name"] = dependent_columns[1].text_input(
                    "Child first name", value=document.result.dependent_name, key=f"dep_{document.id}"
                )
                fields["child_index"] = dependent_columns[2].number_input(
                    "Child no.",
                    min_value=0, max_value=20, step=1,
                    value=int(document.result.child_index or 0),
                    key=f"childno_{document.id}",
                )

                approve = st.checkbox(
                    "Reviewed - approve for export",
                    value=document.approved,
                    key=f"approve_{document.id}",
                    help="Overrides the confidence gate once a case worker has checked the data.",
                )

                if st.form_submit_button("Apply corrections", use_container_width=True):
                    pipeline.apply_review(document, fields)
                    document.approved = approve
                    st.rerun()

            st.caption("Resulting file name")
            st.code(document.filename or "-", language=None)
            st.caption(f"Client folder: Clients/{document.client_key}")


def render_review(pipeline: ProcessingPipeline) -> None:
    """Render the whole review screen."""
    items = documents()
    if not items:
        return

    st.subheader("2. Review")
    render_summary(items)

    order = {"REVIEW REQUIRED": 0, STATUS_READY: 1}
    for index, document in enumerate(
        sorted(items, key=lambda item: (bool(item.is_exported), order.get(item.status, 2), item.filename))
    ):
        render_document_card(pipeline, document, index)


# --------------------------------------------------------------------------- #
# Step 3 - export
# --------------------------------------------------------------------------- #
def render_export(pipeline: ProcessingPipeline) -> None:
    """Render the export actions and the ZIP downloads."""
    st.subheader("3. Export")

    items = documents()
    exportable = [item for item in items if not item.error and not item.is_exported and not item.requires_review]
    pending_review = [item for item in items if not item.error and not item.is_exported and item.requires_review]

    columns = st.columns(3)
    if columns[0].button(
        f"Export all ({len(exportable)})",
        type="primary",
        disabled=not exportable,
        use_container_width=True,
        help="Files the confidence gate accepted or a case worker approved are moved into Clients/.",
    ):
        exported, failures = pipeline.export_documents(exportable)
        st.session_state["export_summary"] = (len(exported), len(failures))
        for document, message in failures:
            st.error(f"{document.filename}: {message}")
        st.rerun()

    if columns[1].button(
        f"Park review items ({len(pending_review)})",
        disabled=not pending_review,
        use_container_width=True,
        help="Moves every REVIEW REQUIRED document into data/review.",
    ):
        exported, failures = pipeline.export_documents(pending_review)
        st.session_state["export_summary"] = (len(exported), len(failures))
        st.rerun()

    summary = st.session_state.get("export_summary")
    if summary:
        stored, failed = summary
        columns[2].success(f"{stored} file(s) stored, {failed} failure(s)")

    st.divider()
    st.markdown("**Download**")

    clients = pipeline.folders.list_clients()
    client_keys = [client.key for client in clients if client.files]

    download_columns = st.columns([2, 1, 1])
    selected = download_columns[0].selectbox(
        "Client", options=client_keys or ["- no client yet -"], disabled=not client_keys
    )

    if download_columns[1].button("Create ZIP", disabled=not client_keys, use_container_width=True):
        build_archive(lambda: pipeline.build_client_archive(selected))

    if download_columns[2].button("ZIP all clients", disabled=not client_keys, use_container_width=True):
        build_archive(pipeline.build_full_archive)

    archive: ExportResult | None = st.session_state.get("last_archive")
    if archive and archive.path.is_file():
        st.download_button(
            label=f"Download {archive.name} ({archive.file_count} file(s), {archive.size_kb:.0f} KB)",
            data=archive.read_bytes(),
            file_name=archive.name,
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )


def build_archive(factory) -> None:
    """Run an archive factory and store the result in the session state."""
    try:
        archive = factory()
    except ExportError as exc:
        st.error(str(exc))
        return
    st.session_state["last_archive"] = archive
    st.toast(f"{archive.name} created")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    """Entry point of the Streamlit application."""
    st.set_page_config(page_title=APP_TITLE, page_icon="🛂", layout="wide")
    init_state()
    pipeline = get_pipeline()

    st.title(f"🛂 {APP_TITLE}")
    st.caption(" → ".join(STAGES))

    render_sidebar(pipeline)
    render_upload(pipeline)

    if documents():
        st.divider()
        render_review(pipeline)
        st.divider()
        render_export(pipeline)
    elif st.session_state["batch_finished"]:
        st.info("No documents were detected in the last batch.")


if __name__ == "__main__":
    main()
