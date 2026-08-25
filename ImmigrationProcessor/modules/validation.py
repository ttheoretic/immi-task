"""Field normalisation, business validation and the confidence gate.

This is the lowest layer of the domain (it only depends on :mod:`config`), so
both :mod:`modules.classifier` and :mod:`modules.pipeline` can use it without
creating an import cycle.  The document type specification is always passed in
by the caller rather than imported.

Responsibilities:

* parse the many date formats found on immigration documents and render them in
  the two formats the naming convention needs (``DD.MM.YYYY`` / ``YYYY-MM``),
* tidy names, companies and cities,
* decide whether a document may be exported automatically or has to go through
  ``data/review`` (confidence gate + completeness checks).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Final

from config import SETTINGS, Settings, get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime cycle
    from modules.classifier import DocumentTypeSpec, ExtractionResult

logger = get_logger(__name__)

STATUS_READY: Final[str] = "READY"
STATUS_REVIEW: Final[str] = "REVIEW REQUIRED"

#: Explicit date formats tried in order.
_DATE_FORMATS: Final[tuple[str, ...]] = (
    "%d.%m.%Y", "%d.%m.%y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%d %m %Y", "%m/%d/%Y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
    "%d.%b.%Y", "%Y%m%d",
)

#: German month names / abbreviations mapped to English so ``strptime`` works
#: independently of the machine locale.
_MONTH_TRANSLATIONS: Final[dict[str, str]] = {
    "januar": "January", "februar": "February", "marz": "March", "märz": "March",
    "mai": "May", "juni": "June", "juli": "July", "oktober": "October",
    "dezember": "December", "jan": "Jan", "feb": "Feb", "mrz": "Mar",
    "apr": "Apr", "jun": "Jun", "jul": "Jul", "aug": "Aug", "sep": "Sep",
    "sept": "Sep", "okt": "Oct", "nov": "Nov", "dez": "Dec",
}

_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b("
    r"\d{1,2}[.\-/\s]\d{1,2}[.\-/\s]\d{2,4}"      # 15.08.2032 / 15-08-32
    r"|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"             # 2032-08-15
    r"|\d{1,2}\s?[A-Za-zÄÖÜäöü]{3,9}\s?\d{4}"     # 15 AUG 2032
    r")\b"
)

_PERIOD_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(0?[1-9]|1[0-2])\s?[./-]\s?(20\d{2})\b|\b(20\d{2})\s?[./-]\s?(0?[1-9]|1[0-2])\b"
)

_ILLEGAL_TEXT = re.compile(r"[\r\n\t]+")


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #
def _translate_months(value: str) -> str:
    """Replace German month names with their English equivalent."""
    lowered = value.lower()
    for german, english in _MONTH_TRANSLATIONS.items():
        if german in lowered:
            lowered = lowered.replace(german, english.lower())
    return lowered


def parse_date(value: Any) -> date | None:
    """Parse *value* into a :class:`datetime.date`, or ``None`` when impossible.

    Accepts ``date``/``datetime`` objects and the common European, ISO and
    English/German textual formats found on immigration documents.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    text = _ILLEGAL_TEXT.sub(" ", str(value)).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", _translate_months(text))
    # Normalise separators for the numeric formats.
    candidates = [text, text.replace(",", " ").strip()]

    for candidate in candidates:
        for fmt in _DATE_FORMATS:
            try:
                parsed = datetime.strptime(candidate.title() if "%b" in fmt or "%B" in fmt else candidate, fmt)
            except ValueError:
                continue
            if parsed.year < 100:  # two digit years -> 20xx
                parsed = parsed.replace(year=parsed.year + 2000)
            return parsed.date()
    return None


def format_date(value: date | None) -> str:
    """Render a date in the naming convention format ``DD.MM.YYYY``."""
    return value.strftime("%d.%m.%Y") if value else ""


def normalise_date_string(value: Any) -> str:
    """Return *value* as ``DD.MM.YYYY`` or ``""`` when it cannot be parsed."""
    return format_date(parse_date(value))


def find_dates(text: str) -> list[date]:
    """Return every parseable date contained in *text* (document order)."""
    found: list[date] = []
    for match in _DATE_PATTERN.finditer(text):
        parsed = parse_date(match.group(1))
        if parsed:
            found.append(parsed)
    return found


def normalise_period(value: Any) -> str:
    """Return a payslip period as ``YYYY-MM`` or ``""``."""
    if not value:
        return ""
    text = str(value).strip()
    iso = re.fullmatch(r"(20\d{2})-(0[1-9]|1[0-2])", text)
    if iso:
        return text
    parsed = parse_date(text)
    if parsed:
        return f"{parsed.year:04d}-{parsed.month:02d}"
    match = _PERIOD_PATTERN.search(_translate_months(text))
    if match:
        if match.group(1) and match.group(2):
            return f"{int(match.group(2)):04d}-{int(match.group(1)):02d}"
        if match.group(3) and match.group(4):
            return f"{int(match.group(3)):04d}-{int(match.group(4)):02d}"
    return ""


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def tidy_name(value: str) -> str:
    """Clean a person name: strip noise, fix ALL CAPS coming from OCR/MRZ."""
    if not value:
        return ""
    cleaned = _ILLEGAL_TEXT.sub(" ", str(value))
    cleaned = re.sub(r"[^\w\s'\-.äöüÄÖÜßéèêáàâçñ]", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -.")
    if not cleaned:
        return ""
    if cleaned.isupper() or cleaned.islower():
        cleaned = " ".join(part.capitalize() for part in cleaned.split(" "))
    return cleaned


def tidy_company(value: str) -> str:
    """Clean a company name (keeps legal forms such as ``GmbH`` intact)."""
    if not value:
        return ""
    cleaned = _ILLEGAL_TEXT.sub(" ", str(value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,;-.")
    return cleaned


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ValidationIssue:
    """A single problem found on an extracted document."""

    field_name: str
    message: str
    severity: str = "error"  # "error" forces review, "warning" is informational

    def __str__(self) -> str:  # pragma: no cover - convenience for the UI
        return f"[{self.severity}] {self.field_name}: {self.message}"


@dataclass
class ValidationReport:
    """Outcome of validating one extracted document."""

    issues: list[ValidationIssue] = field(default_factory=list)
    confidence: float = 0.0
    threshold: float = 0.90

    @property
    def errors(self) -> list[ValidationIssue]:
        """Issues that force a manual review."""
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        """Informational issues that do not block an export."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def requires_review(self) -> bool:
        """True when the document must not be exported automatically."""
        return bool(self.errors) or self.confidence < self.threshold

    @property
    def status(self) -> str:
        """``READY`` or ``REVIEW REQUIRED``."""
        return STATUS_REVIEW if self.requires_review else STATUS_READY

    def summary(self) -> str:
        """One-line summary for logs and the review screen."""
        if not self.issues:
            return "no issues"
        return "; ".join(f"{issue.field_name}: {issue.message}" for issue in self.issues)


class DocumentValidator:
    """Normalises extracted fields and applies the business rules."""

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings

    # -- public API -------------------------------------------------------- #
    def normalise(self, result: "ExtractionResult") -> "ExtractionResult":
        """Clean and reformat the extracted fields in place, returning *result*."""
        result.first_name = tidy_name(result.first_name)
        result.last_name = tidy_name(result.last_name)
        result.dependent_name = tidy_name(result.dependent_name)
        result.company = tidy_company(result.company)
        result.city = tidy_name(result.city)
        normalised_date = normalise_date_string(result.valid_until)
        if result.valid_until and not normalised_date:
            logger.debug("Could not parse validity date %r", result.valid_until)
        result.valid_until = normalised_date or ""
        result.period = normalise_period(result.period)
        if result.relationship == "child" and result.child_index is None:
            result.child_index = 1
        if result.relationship != "child":
            # The given name of a dependant only appears in child file names.
            result.child_index = None
            result.dependent_name = ""
        return result

    def validate(
        self,
        result: "ExtractionResult",
        spec: "DocumentTypeSpec",
        *,
        normalise: bool = True,
    ) -> ValidationReport:
        """Validate *result* against the requirements of *spec*.

        Args:
            result: The extracted document data (modified in place when
                *normalise* is True).
            spec: Catalogue entry of the detected document type.
            normalise: Whether to clean the fields before validating.

        Returns:
            A :class:`ValidationReport` carrying every issue plus the resulting
            ``READY`` / ``REVIEW REQUIRED`` status.
        """
        if normalise:
            self.normalise(result)

        report = ValidationReport(
            confidence=float(result.confidence or 0.0),
            threshold=self._settings.confidence_threshold,
        )
        add = report.issues.append

        if spec.key == "unknown" or not result.document_type or result.document_type == "unknown":
            add(ValidationIssue("document_type", "document type could not be determined"))

        if not result.last_name:
            add(ValidationIssue("last_name", "missing"))
        if not result.first_name:
            add(ValidationIssue("first_name", "missing"))
        if not result.company:
            add(ValidationIssue("company", "missing", severity="warning"))

        if spec.requires_validity:
            if not result.valid_until:
                add(ValidationIssue("valid_until", "validity date required for this document type"))
            else:
                expiry = parse_date(result.valid_until)
                if expiry and expiry < date.today():
                    add(ValidationIssue("valid_until", f"document expired on {format_date(expiry)}", severity="warning"))

        if spec.requires_period and not result.period:
            add(ValidationIssue("period", "payslip period (YYYY-MM) required"))

        if spec.requires_city and not result.city:
            add(ValidationIssue("city", "city required for town hall documents"))

        if result.relationship == "child" and not result.dependent_name:
            add(ValidationIssue("dependent_name", "given name of the child required", severity="warning"))

        if report.confidence < self._settings.confidence_threshold:
            add(
                ValidationIssue(
                    "confidence",
                    f"{report.confidence:.0%} is below the {self._settings.confidence_threshold:.0%} threshold",
                    severity="warning",
                )
            )

        logger.debug("Validation for %s -> %s (%s)", spec.key, report.status, report.summary())
        return report
