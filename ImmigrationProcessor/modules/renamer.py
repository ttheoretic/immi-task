"""Implementation of the file naming convention.

General rule::

    <DocumentType>_<LastName>_<FirstName>_<CompanyName>.pdf

``<DocumentType>`` is the catalogue label with its placeholders resolved, for
example ``01_passport_valid 15.08.2032``.  Dependants insert a relationship
token directly after the document type::

    01_passport_spouse_Smith_John_Microsoft.pdf
    10_birth certificate_child 1_Emily_Smith_John_Microsoft.pdf

If an optional placeholder cannot be filled (no validity date extracted, no
payslip period, no city), the corresponding label chunk is dropped instead of
writing a dummy value - the document is flagged ``REVIEW REQUIRED`` anyway, and
the case worker gets a clean name to correct.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from config import SETTINGS, Settings, get_logger
from modules.classifier import DocumentTypeSpec, ExtractionResult, get_spec

logger = get_logger(__name__)

#: Characters no mainstream file system accepts.
_ILLEGAL_CHARS: Final[re.Pattern[str]] = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{(\w+)\}")

MAX_FILENAME_LENGTH: Final[int] = 180


def sanitise_part(value: str, fallback: str = "") -> str:
    """Return *value* safe for use inside a file name.

    Spaces are preserved (the naming convention uses them), underscores are
    removed from field values so they cannot break the ``_`` separated layout.
    """
    if value is None:
        return fallback
    cleaned = _ILLEGAL_CHARS.sub(" ", str(value))
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or fallback


def unique_path(directory: Path, filename: str) -> Path:
    """Return a non-existing path in *directory* for *filename*.

    Collisions get a ``(2)``, ``(3)``, ... suffix so nothing is overwritten.
    """
    directory = Path(directory)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


class FileNameBuilder:
    """Builds convention compliant file names from an :class:`ExtractionResult`."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._unknown = settings.unknown_token

    # -- public API -------------------------------------------------------- #
    def build(self, result: ExtractionResult, spec: DocumentTypeSpec | None = None) -> str:
        """Return the file name (including ``.pdf``) for *result*."""
        spec = spec or get_spec(result.document_type)
        parts: list[str] = [self._document_type_part(result, spec)]

        relationship_part = self._relationship_part(result, spec)
        if relationship_part:
            parts.append(relationship_part)

        parts.append(sanitise_part(result.last_name, self._unknown))
        parts.append(sanitise_part(result.first_name, self._unknown))
        parts.append(sanitise_part(result.company, self._unknown))

        filename = "_".join(part for part in parts if part) + ".pdf"
        return self._truncate(filename)

    def build_path(
        self,
        directory: Path,
        result: ExtractionResult,
        spec: DocumentTypeSpec | None = None,
    ) -> Path:
        """Return a collision free path for *result* inside *directory*."""
        return unique_path(directory, self.build(result, spec))

    def client_key(self, result: ExtractionResult) -> str:
        """Return the client folder name ``LastName_FirstName``.

        Dependants are always filed under the main applicant, whose name is the
        one carried by the file name.
        """
        last = sanitise_part(result.last_name, self._unknown).replace(" ", "")
        first = sanitise_part(result.first_name, self._unknown).replace(" ", "")
        return f"{last}_{first}"

    # -- internals --------------------------------------------------------- #
    def _document_type_part(self, result: ExtractionResult, spec: DocumentTypeSpec) -> str:
        """Resolve the label placeholders, dropping chunks that stay empty."""
        values = {
            "validity": sanitise_part(result.valid_until),
            "period": sanitise_part(result.period),
            "city": sanitise_part(result.city),
            # Documents outside the convention carry their own title, e.g.
            # "MB Business Contact_Smith_John_Microsoft.pdf".
            "title": sanitise_part(result.title),
        }
        chunks: list[str] = []
        for chunk in spec.label_template.split("_"):
            placeholders = _PLACEHOLDER.findall(chunk)
            if placeholders and not all(values.get(name) for name in placeholders):
                logger.debug(
                    "Dropping label chunk %r of %s - missing %s",
                    chunk, spec.key, ", ".join(name for name in placeholders if not values.get(name)),
                )
                continue
            chunks.append(_PLACEHOLDER.sub(lambda match: values.get(match.group(1), ""), chunk))
        rendered = "_".join(chunk for chunk in chunks if chunk)
        if rendered:
            return rendered
        # Nothing could be resolved (e.g. "other" without a title yet).
        fallback = spec.label_template.split("_")[0]
        return self._unknown if "{" in fallback else fallback

    def _relationship_part(self, result: ExtractionResult, spec: DocumentTypeSpec) -> str:
        """Return the dependant token (``spouse`` / ``child 1_Emily``)."""
        if spec.relationship_in_label or not result.relationship:
            return ""
        if result.relationship == "spouse":
            # The convention names the spouse token only - the main applicant's
            # name already follows it (01_passport_spouse_Smith_John_Company).
            return "spouse"
        if result.relationship == "child":
            index = result.child_index if result.child_index and result.child_index > 0 else 1
            token = f"child {index}"
            name = sanitise_part(result.dependent_name)
            return f"{token}_{name}" if name else token
        return ""

    @staticmethod
    def _truncate(filename: str) -> str:
        """Keep file names within a safe length for all file systems."""
        if len(filename) <= MAX_FILENAME_LENGTH:
            return filename
        stem = filename[: MAX_FILENAME_LENGTH - 4].rstrip(" _.")
        logger.warning("Truncated over-long file name %r", filename)
        return f"{stem}.pdf"
