"""Document type catalogue, AI extraction and multi-document boundary detection.

The module exposes three things:

``DOCUMENT_TYPES``
    The closed catalogue of the 22 document types the naming convention allows.

``ExtractionResult``
    The structured payload extracted per document (the mandatory AI schema plus
    a few fields required by the dependent naming rules).

``DocumentClassifier``
    Facade that turns the text of one uploaded PDF into a list of
    :class:`ClassifiedSegment` objects (page ranges + extracted fields).  It
    prefers Azure OpenAI and transparently degrades to a deterministic
    keyword/regex engine when Azure is unavailable or fails.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from config import SETTINGS, Settings, get_logger
from modules.ocr import DocumentText
from modules.validation import (
    find_dates,
    normalise_date_string,
    normalise_period,
)

logger = get_logger(__name__)

UNKNOWN_TYPE_KEY = "unknown"


class ClassificationError(RuntimeError):
    """Raised when the AI classification backend cannot deliver a result."""


# --------------------------------------------------------------------------- #
# Document type catalogue
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DocumentTypeSpec:
    """One entry of the closed document type catalogue.

    Attributes:
        key: Stable machine identifier, also used in AI prompts.
        label_template: Naming-convention prefix; may contain the placeholders
            ``{validity}`` (``DD.MM.YYYY``), ``{period}`` (``YYYY-MM``) and
            ``{city}``.
        description: Human readable description shown in the review UI.
        keywords: Weighted detection keywords (lower case, EN + DE).
        requires_validity / requires_period / requires_city: Which extra field
            the naming convention needs for this type.
        relationship_in_label: ``True`` when the label already encodes the
            relationship (``08_POA_spouse``) so it must not be appended twice.
    """

    key: str
    label_template: str
    description: str
    keywords: tuple[tuple[str, float], ...] = ()
    requires_validity: bool = False
    requires_period: bool = False
    requires_city: bool = False
    relationship_in_label: bool = False

    @property
    def sort_key(self) -> str:
        """Numeric prefix of the label, used to order documents in the UI."""
        return self.label_template.split("_", 1)[0]


def _spec(
    key: str,
    label_template: str,
    description: str,
    keywords: Sequence[tuple[str, float]] = (),
    **flags: bool,
) -> DocumentTypeSpec:
    """Small helper keeping the catalogue below readable."""
    return DocumentTypeSpec(
        key=key,
        label_template=label_template,
        description=description,
        keywords=tuple(keywords),
        **flags,
    )


#: The closed catalogue.  No other document type may ever leave the application.
DOCUMENT_TYPES: dict[str, DocumentTypeSpec] = {
    spec.key: spec
    for spec in (
        _spec(
            "01_passport",
            "01_passport_valid {validity}",
            "Passport",
            [
                ("passport", 3.0), ("reisepass", 3.0), ("passeport", 2.0),
                ("passport no", 2.5), ("passport number", 2.5), ("passnummer", 2.5),
                ("date of issue", 1.0), ("date of expiry", 1.5), ("nationality", 1.0),
                ("staatsangehorigkeit", 1.0), ("staatsangehörigkeit", 1.0),
                ("place of birth", 0.5), ("authority", 0.5), ("p<", 3.0),
            ],
            requires_validity=True,
        ),
        _spec(
            "02_university_degree",
            "02_university degree",
            "University degree / diploma",
            [
                ("bachelor", 2.5), ("master", 2.5), ("diploma", 2.5), ("diplom", 2.5),
                ("degree certificate", 3.0), ("urkunde", 2.0), ("university", 1.5),
                ("universitat", 1.5), ("universität", 1.5), ("hochschule", 1.5),
                ("faculty", 1.0), ("has been awarded", 1.5), ("academic degree", 2.0),
            ],
        ),
        _spec(
            "02.1_anabin_uni",
            "02.1_Anabin_uni_H+",
            "Anabin university status (H+)",
            [("anabin", 3.0), ("h+", 2.0), ("hochschule status", 1.5), ("institution", 0.5)],
        ),
        _spec(
            "02.2_anabin_degree",
            "02.2_Anabin_degree_entspricht",
            "Anabin degree equivalence",
            [("anabin", 3.0), ("entspricht", 2.5), ("gleichwertig", 2.0), ("abschluss", 1.5)],
        ),
        _spec(
            "03_cv",
            "03_CV",
            "Curriculum vitae",
            [
                ("curriculum vitae", 3.0), ("resume", 2.0), ("lebenslauf", 3.0),
                ("work experience", 2.5), ("berufserfahrung", 2.5),
                ("professional experience", 2.5), ("education", 1.0), ("skills", 1.0),
            ],
        ),
        _spec(
            "04_employment_contract",
            "04_employment contract_extension_amendment",
            "Employment contract / extension / amendment",
            [
                ("employment contract", 3.0), ("arbeitsvertrag", 3.0),
                ("contract of employment", 3.0), ("anstellungsvertrag", 3.0),
                ("amendment", 1.5), ("nachtrag", 1.5), ("extension", 1.0),
                ("probezeit", 1.0), ("probationary period", 1.0),
                ("gross annual salary", 1.5), ("jahresbruttogehalt", 1.5),
            ],
        ),
        _spec(
            "05_assignment_letter",
            "05_assignment letter",
            "Assignment letter",
            [
                ("assignment letter", 3.0), ("entsendevertrag", 3.0),
                ("letter of assignment", 3.0), ("secondment", 2.5),
                ("entsendung", 2.5), ("host company", 1.5), ("home company", 1.5),
            ],
        ),
        _spec(
            "06_job_description",
            "06_job description",
            "Job description",
            [
                ("job description", 3.0), ("stellenbeschreibung", 3.0),
                ("tatigkeitsbeschreibung", 2.5), ("tätigkeitsbeschreibung", 2.5),
                ("responsibilities", 1.5), ("reports to", 1.0),
            ],
        ),
        _spec(
            "07_questionnaire",
            "07_questionnaire",
            "Questionnaire",
            [
                ("questionnaire", 3.0), ("fragebogen", 3.0),
                ("please answer", 1.0), ("erklarung zum beschaftigungsverhaltnis", 2.5),
                ("erklärung zum beschäftigungsverhältnis", 2.5),
            ],
        ),
        _spec(
            "08_poa_spouse",
            "08_POA_spouse",
            "Power of attorney - spouse",
            [
                ("power of attorney", 3.0), ("vollmacht", 3.0), ("spouse", 2.0),
                ("ehegatte", 2.0), ("ehefrau", 2.0), ("ehemann", 2.0),
            ],
            relationship_in_label=True,
        ),
        _spec(
            "08_poa_child",
            "08_POA_child",
            "Power of attorney - child",
            [
                ("power of attorney", 3.0), ("vollmacht", 3.0), ("child", 2.0),
                ("kind", 2.0), ("minor", 1.5), ("legal guardian", 1.5),
            ],
            relationship_in_label=True,
        ),
        _spec(
            "09_marriage_certificate",
            "09_marriage certificate",
            "Marriage certificate",
            [
                ("marriage certificate", 3.5), ("certificate of marriage", 3.5),
                ("heiratsurkunde", 3.5), ("eheurkunde", 3.0), ("marriage", 1.5),
                ("standesamt", 1.0), ("date of marriage", 2.0),
            ],
        ),
        _spec(
            "10_birth_certificate",
            "10_birth certificate",
            "Birth certificate",
            [
                ("birth certificate", 3.5), ("geburtsurkunde", 3.5),
                ("certificate of birth", 3.5), ("date of birth", 1.0),
                ("mother", 1.0), ("father", 1.0), ("eltern", 1.0),
            ],
        ),
        _spec(
            "11_health_insurance",
            "11_health insurance",
            "Health insurance confirmation",
            [
                ("health insurance", 3.0), ("krankenversicherung", 3.5),
                ("insurance certificate", 3.0), ("versicherungsbescheinigung", 3.0),
                ("mitgliedsbescheinigung", 2.5), ("aok", 1.5), ("techniker krankenkasse", 2.0),
                ("policy number", 1.0),
            ],
        ),
        _spec(
            "12_payslip",
            "12_payslip_{period}",
            "Payslip",
            [
                ("payslip", 3.5), ("pay slip", 3.5), ("payroll", 3.0),
                ("gehaltsabrechnung", 3.5), ("lohnabrechnung", 3.5),
                ("entgeltabrechnung", 3.5), ("net pay", 1.5), ("gross pay", 1.5),
                ("steuerklasse", 1.5), ("sozialversicherung", 1.0),
            ],
            requires_period=True,
        ),
        _spec(
            "13_rental_agreement",
            "13_rental agreement",
            "Rental agreement",
            [
                ("rental agreement", 3.0), ("mietvertrag", 3.5),
                ("lease agreement", 3.0), ("landlord", 1.5), ("vermieter", 1.5),
                ("tenant", 1.5), ("mieter", 1.5), ("kaltmiete", 1.5),
            ],
        ),
        _spec(
            "14_fea_pre_approval",
            "14_FEA pre-approval note",
            "Federal Employment Agency pre-approval note",
            [
                ("pre-approval", 3.0), ("vorabzustimmung", 3.5),
                ("bundesagentur fur arbeit", 3.0), ("bundesagentur für arbeit", 3.0),
                ("federal employment agency", 3.0), ("zustimmung", 1.5),
            ],
        ),
        _spec(
            "15_visa",
            "15_visa_valid {validity}",
            "Visa",
            [
                ("visa", 3.5), ("visum", 3.5), ("schengen", 2.0),
                ("number of entries", 2.5), ("entries", 1.5), ("category", 1.5),
                ("duration of stay", 2.0), ("valid for", 1.5), ("type: d", 1.5),
            ],
            requires_validity=True,
        ),
        _spec(
            "16_town_hall_registration",
            "16_town hall registration_{city}",
            "Town hall registration (Anmeldung)",
            [
                ("meldebescheinigung", 3.5), ("anmeldung", 3.0),
                ("registration certificate", 3.0), ("town hall registration", 3.0),
                ("einwohnermeldeamt", 2.5), ("burgeramt", 2.0), ("bürgeramt", 2.0),
                ("wohnung", 1.0), ("registered address", 1.5),
            ],
            requires_city=True,
        ),
        _spec(
            "17_interim_permit",
            "17_interim permit_valid {validity}",
            "Interim / provisional residence permit (Fiktionsbescheinigung)",
            [
                ("fiktionsbescheinigung", 3.5), ("interim permit", 3.0),
                ("provisional residence", 3.0), ("vorlaufige", 1.5), ("vorläufige", 1.5),
            ],
            requires_validity=True,
        ),
        _spec(
            "18_wp",
            "18_WP_valid {validity}",
            "Work permit / residence title of the main applicant",
            [
                ("aufenthaltstitel", 3.5), ("residence permit", 3.0),
                ("work permit", 3.0), ("supplementary sheet", 2.5),
                ("zusatzblatt", 2.5), ("blaue karte eu", 3.0), ("eu blue card", 3.0),
                ("erwerbstatigkeit gestattet", 2.0), ("erwerbstätigkeit gestattet", 2.0),
            ],
            requires_validity=True,
        ),
        _spec(
            "19_rp_dependents",
            "19_RP_dependents_valid {validity}",
            "Residence permit of a dependent",
            [
                ("aufenthaltstitel", 2.0), ("residence permit", 2.0),
                ("familiennachzug", 3.0), ("family reunification", 3.0),
                ("dependent", 2.0), ("ehegattennachzug", 3.0),
            ],
            requires_validity=True,
        ),
        _spec(
            "20_town_hall_deregistration",
            "20_town hall de-registration_{city}",
            "Town hall de-registration (Abmeldung)",
            [
                ("abmeldung", 3.5), ("abmeldebescheinigung", 3.5),
                ("de-registration", 3.0), ("deregistration", 3.0),
                ("wegzug", 2.0), ("moving out", 1.5),
            ],
            requires_city=True,
        ),
    )
}

#: Internal-only pseudo type for documents that could not be classified.  Files
#: carrying it are always routed to ``data/review`` and can never be exported.
UNKNOWN_SPEC = DocumentTypeSpec(
    key=UNKNOWN_TYPE_KEY,
    label_template="00_unclassified",
    description="Could not be classified - manual review required",
)


def get_spec(document_type: str | None) -> DocumentTypeSpec:
    """Return the catalogue entry for *document_type* (``UNKNOWN_SPEC`` if absent)."""
    if not document_type:
        return UNKNOWN_SPEC
    return DOCUMENT_TYPES.get(document_type.strip(), UNKNOWN_SPEC)


def type_choices() -> list[str]:
    """Return all selectable type keys for the review UI (unknown first)."""
    return [UNKNOWN_TYPE_KEY] + sorted(DOCUMENT_TYPES, key=lambda k: DOCUMENT_TYPES[k].label_template)


# --------------------------------------------------------------------------- #
# Extraction result
# --------------------------------------------------------------------------- #
_RELATIONSHIP_ALIASES: Mapping[str, str] = {
    "spouse": "spouse", "wife": "spouse", "husband": "spouse", "ehepartner": "spouse",
    "ehegatte": "spouse", "ehefrau": "spouse", "ehemann": "spouse", "partner": "spouse",
    "child": "child", "son": "child", "daughter": "child", "kind": "child",
    "sohn": "child", "tochter": "child", "minor": "child",
    "self": "", "applicant": "", "main applicant": "", "principal": "",
    "employee": "", "none": "", "n/a": "", "-": "",
}


def normalise_relationship(value: str | None) -> str:
    """Map a free-text relationship onto ``""`` | ``"spouse"`` | ``"child"``."""
    if not value:
        return ""
    key = str(value).strip().lower()
    return _RELATIONSHIP_ALIASES.get(key, "child" if "child" in key else ("spouse" if "spouse" in key else ""))


@dataclass
class ExtractionResult:
    """Structured data extracted from a single (sub-)document.

    The first eight attributes are the mandatory AI extraction schema.  The
    remaining ones are required by the naming convention for dependents and for
    payslips, plus bookkeeping about how the values were obtained.
    """

    first_name: str = ""
    last_name: str = ""
    company: str = ""
    document_type: str = UNKNOWN_TYPE_KEY
    valid_until: str = ""          # normalised to DD.MM.YYYY by validation
    city: str = ""
    relationship: str = ""         # "" | "spouse" | "child"
    confidence: float = 0.0
    dependent_name: str = ""       # given name of the child (naming rule)
    child_index: int | None = None  # "child 1", "child 2", ...
    period: str = ""               # payslip period, YYYY-MM
    source: str = "heuristic"      # "azure_openai" | "heuristic" | "manual"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the mandatory extraction schema (plus naming extras)."""
        return {
            "first_name": self.first_name,
            "last_name": self.last_name,
            "company": self.company,
            "document_type": self.document_type,
            "valid_until": self.valid_until,
            "city": self.city,
            "relationship": self.relationship,
            "confidence": round(float(self.confidence), 4),
            "dependent_name": self.dependent_name,
            "child_index": self.child_index,
            "period": self.period,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], source: str) -> "ExtractionResult":
        """Build a result from a (possibly noisy) AI JSON payload."""

        def text(key: str) -> str:
            value = payload.get(key)
            if value is None:
                return ""
            value = str(value).strip()
            return "" if value.lower() in {"null", "none", "n/a", "unknown", "-"} else value

        try:
            confidence = float(payload.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = min(max(confidence, 0.0), 1.0)

        raw_index = payload.get("child_index")
        try:
            child_index = int(raw_index) if raw_index not in (None, "", "null") else None
        except (TypeError, ValueError):
            child_index = None

        document_type = text("document_type")
        if document_type not in DOCUMENT_TYPES:
            logger.warning("AI returned unknown document_type %r - flagged as unclassified", document_type)
            document_type = UNKNOWN_TYPE_KEY
            confidence = min(confidence, 0.4)

        return cls(
            first_name=text("first_name"),
            last_name=text("last_name"),
            company=text("company"),
            document_type=document_type,
            valid_until=text("valid_until"),
            city=text("city"),
            relationship=normalise_relationship(text("relationship")),
            confidence=confidence,
            dependent_name=text("dependent_name"),
            child_index=child_index,
            period=text("period"),
            source=source,
        )


@dataclass
class ClassifiedSegment:
    """A contiguous page range of an uploaded PDF holding one logical document."""

    start_page: int  # 1-based, inclusive
    end_page: int    # 1-based, inclusive
    result: ExtractionResult
    page_count: int = 0

    def __post_init__(self) -> None:
        self.page_count = max(0, self.end_page - self.start_page + 1)

    @property
    def page_label(self) -> str:
        """Human readable page range, e.g. ``"pages 1-2"``."""
        if self.start_page == self.end_page:
            return f"page {self.start_page}"
        return f"pages {self.start_page}-{self.end_page}"


# --------------------------------------------------------------------------- #
# Heuristic engine (deterministic fallback, no cloud calls)
# --------------------------------------------------------------------------- #
_MRZ_LINE1 = re.compile(r"P[<A-Z][A-Z<]{3}([A-Z]+(?:<[A-Z]+)*)<<([A-Z]+(?:<[A-Z]+)*)")
_MRZ_LINE2 = re.compile(r"[A-Z0-9<]{9}\d[A-Z<]{3}\d{6}\d[MFX<](\d{6})\d")

_NAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("last_name", re.compile(
        r"(?:surname|family\s*name|last\s*name|name\s*/\s*surname|nachname|familienname)"
        r"\s*[:\-/]?\s*([A-ZÄÖÜ][\w'\-äöüß]+(?:\s+[A-ZÄÖÜ][\w'\-äöüß]+)?)", re.IGNORECASE)),
    ("first_name", re.compile(
        r"(?:given\s*names?|first\s*name|forename|vorname[n]?)"
        r"\s*[:\-/]?\s*([A-ZÄÖÜ][\w'\-äöüß]+(?:\s+[A-ZÄÖÜ][\w'\-äöüß]+)?)", re.IGNORECASE)),
    ("company", re.compile(
        r"(?:employer|company|arbeitgeber|firma|unternehmen|host\s*company)"
        r"\s*[:\-]?\s*([A-Z0-9ÄÖÜ][\w&.,'\-äöüß ]{2,45})", re.IGNORECASE)),
    ("city", re.compile(
        r"(?:city|town|stadt|wohnort|ort\s*der\s*anmeldung|gemeinde|municipality)"
        r"\s*[:\-]?\s*([A-ZÄÖÜ][\w'\-äöüß]+(?:\s[A-ZÄÖÜ][\w'\-äöüß]+)?)", re.IGNORECASE)),
)

#: Words that belong to the NEXT field label and must never end up in a value
#: (e.g. "Surname: SMITH Given names: JOHN" must yield "SMITH").
_LABEL_STOP_WORDS: frozenset[str] = frozenset({
    "surname", "surnames", "family", "name", "names", "nachname", "familienname",
    "given", "first", "last", "forename", "vorname", "vornamen", "remarks",
    "date", "datum", "nationality", "staatsangehorigkeit", "staatsangehörigkeit",
    "sex", "geschlecht", "birth", "geburt", "geburtsdatum", "place", "ort",
    "employer", "arbeitgeber", "company", "firma", "city", "stadt", "wohnort",
    "abrechnungsmonat", "abrechnungszeitraum", "type", "code", "authority",
    "passport", "issue", "expiry", "valid", "no", "nr", "number",
})


def _trim_label_bleed(value: str) -> str:
    """Drop trailing words that are in fact the beginning of the next label."""
    words = value.split()
    while words and words[-1].strip(":,.").lower() in _LABEL_STOP_WORDS:
        words.pop()
    return " ".join(words)


_EXPIRY_HINTS: tuple[str, ...] = (
    "date of expiry", "valid until", "valid till", "expiry date", "expires",
    "gültig bis", "gultig bis", "gueltig bis", "befristet bis", "date d'expiration",
)

_PERIOD_HINTS: tuple[str, ...] = (
    "abrechnungsmonat", "abrechnungszeitraum", "pay period", "period", "monat", "month",
)


class KeywordClassifier:
    """Deterministic keyword scoring against the document type catalogue."""

    def __init__(self, catalogue: Mapping[str, DocumentTypeSpec] | None = None) -> None:
        self._catalogue = dict(catalogue or DOCUMENT_TYPES)

    def score(self, text: str) -> list[tuple[str, float]]:
        """Return ``[(type_key, score)]`` sorted by descending score."""
        haystack = " ".join((text or "").lower().split())
        if not haystack:
            return []
        scores: list[tuple[str, float]] = []
        for key, spec in self._catalogue.items():
            total = 0.0
            for keyword, weight in spec.keywords:
                hits = haystack.count(keyword)
                if hits:
                    total += weight * min(hits, 2)
            if total:
                scores.append((key, round(total, 2)))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def classify(self, text: str, cap: float) -> tuple[str, float]:
        """Return the best matching type key and a capped confidence value.

        Heuristic results can never exceed *cap* (default 0.55) so that they
        always land in the manual review queue.
        """
        scores = self.score(text)
        if not scores:
            return UNKNOWN_TYPE_KEY, 0.10
        best_key, best_score = scores[0]
        runner_up = scores[1][1] if len(scores) > 1 else 0.0
        if best_score < 2.5:
            return UNKNOWN_TYPE_KEY, min(cap, 0.20 + best_score / 25)

        confidence = 0.30 + min(best_score, 12.0) / 40  # 0.30 .. 0.60
        if runner_up and runner_up > 0.8 * best_score:  # ambiguous match
            confidence -= 0.10
        return best_key, round(min(cap, max(0.15, confidence)), 3)


class HeuristicFieldExtractor:
    """Regex/MRZ based extraction used when Azure OpenAI is not available."""

    def extract(self, text: str, document_type: str, confidence: float) -> ExtractionResult:
        """Extract the mandatory fields from *text* as well as regexes allow."""
        result = ExtractionResult(document_type=document_type, confidence=confidence, source="heuristic")
        spec = get_spec(document_type)

        self._apply_mrz(text, result)
        self._apply_labels(text, result)

        if spec.requires_validity and not result.valid_until:
            result.valid_until = self._find_expiry(text)
        if spec.requires_period and not result.period:
            result.period = self._find_period(text)
        if spec.relationship_in_label:
            result.relationship = "spouse" if spec.key.endswith("spouse") else "child"

        result.notes = "extracted without AI (keyword/regex engine)"
        return result

    # -- internals --------------------------------------------------------- #
    def _apply_mrz(self, text: str, result: ExtractionResult) -> None:
        """Read names and expiry date from a passport machine readable zone."""
        compact = text.replace(" ", "")
        match = _MRZ_LINE1.search(compact)
        if match:
            surname = match.group(1).replace("<", " ").strip()
            given = match.group(2).replace("<", " ").strip()
            if surname:
                result.last_name = surname.title()
            if given:
                result.first_name = given.split(" ")[0].title()
            logger.debug("MRZ names detected: %s, %s", result.last_name, result.first_name)

        expiry = _MRZ_LINE2.search(compact)
        if expiry:
            raw = expiry.group(1)  # YYMMDD
            try:
                year = 2000 + int(raw[0:2])
                result.valid_until = f"{raw[4:6]}.{raw[2:4]}.{year}"
            except (ValueError, IndexError):  # pragma: no cover - malformed MRZ
                pass

    def _apply_labels(self, text: str, result: ExtractionResult) -> None:
        """Read labelled fields such as ``Surname: ...`` from the text."""
        for field_name, pattern in _NAME_PATTERNS:
            if getattr(result, field_name):
                continue
            match = pattern.search(text)
            if not match:
                continue
            value = _trim_label_bleed(match.group(1).strip())
            if value:
                setattr(result, field_name, value)

    def _find_expiry(self, text: str) -> str:
        """Find the most plausible validity date in *text*."""
        lowered = text.lower()
        for hint in _EXPIRY_HINTS:
            index = lowered.find(hint)
            if index == -1:
                continue
            window = text[index : index + 120]
            dates = find_dates(window)
            if dates:
                return normalise_date_string(dates[0])
        # Fall back to the latest future date found anywhere in the document.
        future = [d for d in find_dates(text) if d.year >= 2000]
        return normalise_date_string(max(future)) if future else ""

    def _find_period(self, text: str) -> str:
        """Find the payslip period (``YYYY-MM``)."""
        lowered = text.lower()
        for hint in _PERIOD_HINTS:
            index = lowered.find(hint)
            if index == -1:
                continue
            period = normalise_period(text[index : index + 60])
            if period:
                return period
        dates = find_dates(text)
        return normalise_period(dates[0]) if dates else ""


class HeuristicSegmenter:
    """Detects document boundaries inside a combined PDF without AI.

    Each page is scored independently; consecutive pages that resolve to the
    same type (or that carry no strong signal of their own) are merged into one
    segment, which mirrors how combined client PDFs are usually assembled.
    """

    def __init__(self, classifier: KeywordClassifier | None = None, settings: Settings = SETTINGS) -> None:
        self._classifier = classifier or KeywordClassifier()
        self._extractor = HeuristicFieldExtractor()
        self._settings = settings

    def segment(self, document: DocumentText) -> list[ClassifiedSegment]:
        """Split *document* into classified page ranges."""
        cap = self._settings.heuristic_confidence_cap
        if not document.pages:
            return []

        page_types: list[tuple[int, str, float]] = []
        for page in document.pages:
            key, confidence = self._classifier.classify(page.text, cap)
            page_types.append((page.number, key, confidence))

        # Pages without their own signal continue the previous document.
        ranges: list[list[Any]] = []
        for number, key, confidence in page_types:
            if ranges and (key == UNKNOWN_TYPE_KEY or key == ranges[-1][2]):
                ranges[-1][1] = number
                ranges[-1][3] = max(ranges[-1][3], confidence)
                continue
            ranges.append([number, number, key, confidence])

        segments: list[ClassifiedSegment] = []
        for start, end, key, confidence in ranges:
            text = document.text_for_range(start, end)
            # Re-classify on the merged text: more context, better signal.
            merged_key, merged_confidence = self._classifier.classify(text, cap)
            if key == UNKNOWN_TYPE_KEY:
                key, confidence = merged_key, merged_confidence
            elif merged_key == key:
                confidence = max(confidence, merged_confidence)
            result = self._extractor.extract(text, key, confidence)
            segments.append(ClassifiedSegment(start_page=start, end_page=end, result=result))

        logger.info(
            "Heuristic segmentation of %s produced %d document(s): %s",
            document.path.name, len(segments),
            ", ".join(f"{s.page_label}={s.result.document_type}" for s in segments),
        )
        return segments


# --------------------------------------------------------------------------- #
# Azure OpenAI engine
# --------------------------------------------------------------------------- #
class AzureOpenAIExtractor:
    """Classifies and extracts fields with an Azure OpenAI chat deployment."""

    _DEFAULT_PROMPT_NAME = "document_classifier.txt"

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings
        self._config = settings.azure_openai
        self._client: Any | None = None
        self._prompt: str | None = None

    @property
    def is_available(self) -> bool:
        """True when Azure OpenAI credentials are configured."""
        return self._config.is_configured

    # -- prompt ------------------------------------------------------------ #
    def _system_prompt(self) -> str:
        """Load (and cache) the system prompt from ``prompts/``."""
        if self._prompt is not None:
            return self._prompt
        prompt_path: Path = self._settings.paths.prompts / self._DEFAULT_PROMPT_NAME
        try:
            self._prompt = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ClassificationError(f"Prompt file missing: {prompt_path} ({exc})") from exc
        self._prompt = self._prompt.replace("{{DOCUMENT_TYPES}}", self._catalogue_block())
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

    # -- client ------------------------------------------------------------ #
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

    # -- public API -------------------------------------------------------- #
    def analyze(self, document: DocumentText) -> list[ClassifiedSegment]:
        """Return one :class:`ClassifiedSegment` per logical document.

        Raises:
            ClassificationError: When the model cannot be reached or returns an
                unusable answer after all retries.
        """
        if not self.is_available:
            raise ClassificationError("Azure OpenAI is not configured")
        if document.is_empty:
            raise ClassificationError("document contains no text to classify")

        payload = self._call_model(self._build_user_message(document))
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
            result = ExtractionResult.from_payload(entry, source="azure_openai")
            segments.append(ClassifiedSegment(start_page=start, end_page=end, result=result))

        if not segments:
            raise ClassificationError("model answer could not be mapped to page ranges")
        logger.info(
            "Azure OpenAI detected %d document(s) in %s: %s",
            len(segments), document.path.name,
            ", ".join(f"{s.page_label}={s.result.document_type} ({s.result.confidence:.0%})" for s in segments),
        )
        return segments

    # -- internals --------------------------------------------------------- #
    @staticmethod
    def _coerce_page(value: Any, default: int, maximum: int) -> int:
        """Clamp a page number returned by the model into the valid range."""
        try:
            page = int(value)
        except (TypeError, ValueError):
            page = default
        return max(1, min(page, max(maximum, 1)))

    def _build_user_message(self, document: DocumentText) -> str:
        """Render the user message: page-marked document text."""
        return (
            f"File name: {document.path.name}\n"
            f"Total pages: {document.page_count}\n"
            f"OCR used: {'yes' if document.ocr_used else 'no'}\n\n"
            "Document text (page markers included):\n"
            f"{document.snippet()}"
        )

    def _call_model(self, user_message: str, attempts: int = 3) -> dict[str, Any]:
        """Call the chat deployment and return the parsed JSON answer."""
        client = self._get_client()
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=self._config.deployment,
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self._config.temperature,
                    max_tokens=self._config.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = (response.choices[0].message.content or "").strip()
                return self._parse_json(content)
            except Exception as exc:  # network, rate limit, malformed JSON, ...
                last_error = exc
                logger.warning("Azure OpenAI call failed (attempt %d/%d): %s", attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(min(2 ** attempt, 8))

        raise ClassificationError(f"Azure OpenAI request failed: {last_error}")

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        """Parse a JSON object out of the model answer (tolerates code fences)."""
        if not content:
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
# Facade
# --------------------------------------------------------------------------- #
class DocumentClassifier:
    """Turns the text of an uploaded PDF into classified page ranges.

    Uses Azure OpenAI when configured and falls back to the deterministic
    keyword engine on any failure, so the application always produces a result
    (a low-confidence one that lands in the review queue).
    """

    def __init__(
        self,
        settings: Settings = SETTINGS,
        ai_extractor: AzureOpenAIExtractor | None = None,
        segmenter: HeuristicSegmenter | None = None,
    ) -> None:
        self._settings = settings
        self._ai = ai_extractor or AzureOpenAIExtractor(settings)
        self._heuristics = segmenter or HeuristicSegmenter(settings=settings)

    @property
    def ai_enabled(self) -> bool:
        """True when the AI backend is configured."""
        return self._ai.is_available

    def analyze(self, document: DocumentText) -> list[ClassifiedSegment]:
        """Classify *document* and return its segments (never empty)."""
        segments: list[ClassifiedSegment] = []

        if self._ai.is_available:
            try:
                segments = self._ai.analyze(document)
            except ClassificationError as exc:
                logger.error("AI classification failed for %s: %s", document.path.name, exc)
            except Exception as exc:  # defensive: SDK surprises must not crash a batch
                logger.exception("Unexpected AI failure for %s: %s", document.path.name, exc)
        else:
            logger.warning(
                "Azure OpenAI not configured - using the keyword engine for %s", document.path.name
            )

        if not segments:
            segments = self._heuristics.segment(document)
        if not segments:  # completely empty/unreadable PDF
            result = ExtractionResult(
                document_type=UNKNOWN_TYPE_KEY,
                confidence=0.0,
                notes="no text could be extracted",
            )
            segments = [ClassifiedSegment(1, max(document.page_count, 1), result)]

        return self._normalise(segments, document)

    # -- internals --------------------------------------------------------- #
    def _normalise(self, segments: list[ClassifiedSegment], document: DocumentText) -> list[ClassifiedSegment]:
        """Sort segments, clamp them to the document and cover unassigned pages."""
        page_count = max(document.page_count, 1)
        cleaned: list[ClassifiedSegment] = []
        for segment in sorted(segments, key=lambda item: (item.start_page, item.end_page)):
            start = max(1, min(segment.start_page, page_count))
            end = max(start, min(segment.end_page, page_count))
            if cleaned and start <= cleaned[-1].end_page:  # overlap -> shift behind the previous one
                start = cleaned[-1].end_page + 1
                if start > page_count or start > end:
                    logger.warning(
                        "Dropping overlapping segment %s of %s", segment.page_label, document.path.name
                    )
                    continue
            cleaned.append(ClassifiedSegment(start, end, segment.result))

        covered = {page for segment in cleaned for page in range(segment.start_page, segment.end_page + 1)}
        missing = [number for number in range(1, page_count + 1) if number not in covered]
        if missing and cleaned:
            # Attach stray pages to the preceding segment instead of losing them.
            for number in missing:
                target = min(cleaned, key=lambda seg: abs(seg.start_page - number))
                target.end_page = max(target.end_page, number)
                target.start_page = min(target.start_page, number)
                target.page_count = target.end_page - target.start_page + 1
            logger.info("Re-attached %d unassigned page(s) of %s", len(missing), document.path.name)
        return cleaned
