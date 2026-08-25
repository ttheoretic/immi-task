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

import re
from datetime import datetime
from dataclasses import dataclass, field
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
OTHER_TYPE_KEY = "other"


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
    requires_title: bool = False
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

#: Internal-only pseudo type for documents no text could be read from. Files
#: carrying it are always routed to ``data/review``.
UNKNOWN_SPEC = DocumentTypeSpec(
    key=UNKNOWN_TYPE_KEY,
    label_template="00_unclassified",
    description="Could not be classified - manual review required",
)

#: Documents that are perfectly readable but simply are not part of the closed
#: naming convention ("MB Business Contact", "Travel Information Sheet", ...).
#: They keep the general file name rule and use their own title as the prefix:
#: ``<Title>_<LastName>_<FirstName>_<Company>.pdf``.
OTHER_SPEC = DocumentTypeSpec(
    key=OTHER_TYPE_KEY,
    label_template="{title}",
    description="Other document (outside the naming convention)",
    requires_title=True,
)


def get_spec(document_type: str | None) -> DocumentTypeSpec:
    """Return the catalogue entry for *document_type* (``UNKNOWN_SPEC`` if absent)."""
    if not document_type:
        return UNKNOWN_SPEC
    key = document_type.strip()
    if key == OTHER_TYPE_KEY:
        return OTHER_SPEC
    return DOCUMENT_TYPES.get(key, UNKNOWN_SPEC)


def type_choices() -> list[str]:
    """Return all selectable type keys for the review UI."""
    return (
        [UNKNOWN_TYPE_KEY, OTHER_TYPE_KEY]
        + sorted(DOCUMENT_TYPES, key=lambda k: DOCUMENT_TYPES[k].label_template)
    )


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
    title: str = ""                # own title of a document outside the convention
    source: str = "rules"          # "rules" | "ollama" | "azure_openai" | "manual"
    notes: str = ""
    #: Why the values are what they are - shown in the review screen.
    evidence: list[str] = field(default_factory=list)
    #: Confidence of the pure keyword match, kept so the score can be
    #: recomputed after a field was filled in later (batch propagation).
    keyword_confidence: float = 0.0

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
            "title": self.title,
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
        title = text("title")
        if document_type == OTHER_TYPE_KEY:
            if not title:  # "other" without a title cannot produce a file name
                logger.warning("Model returned 'other' without a title - needs review")
                confidence = min(confidence, 0.4)
        elif document_type not in DOCUMENT_TYPES:
            logger.warning(
                "Model returned unknown document_type %r - treated as 'other'", document_type
            )
            document_type = OTHER_TYPE_KEY if title else UNKNOWN_TYPE_KEY
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
            title=title,
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
# Rule engine (deterministic, no model involved)
# --------------------------------------------------------------------------- #
#: Machine readable zone of a passport (TD3): two lines of 44 characters.
_MRZ_LINE1 = re.compile(r"P[<A-Z][A-Z<]{3}([A-Z]+(?:<[A-Z]+)*)<<([A-Z]+(?:<[A-Z]+)*)")
_MRZ_LINE2 = re.compile(
    r"([A-Z0-9<]{9})(\d)([A-Z<]{3})(\d{6})(\d)([MFX<])(\d{6})(\d)([A-Z0-9<]{14})(\d)(\d)"
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
    "passport", "issue", "expiry", "valid", "no", "nr", "number", "geboren",
    "und", "and", "wohnhaft", "vertreten", "vater", "father", "mutter", "mother",
    "eltern", "parents", "kind", "child", "wife", "husband", "ehefrau", "ehemann",
    "standesamt", "registrar", "born", "geb",
})

_NAME_CHARS = r"[A-ZÄÖÜ][\w'\-äöüßéèêáàâç]*"
#: Horizontal whitespace only: a person name never spans two lines, so the
#: capture must stop at the line break instead of swallowing the next label.
_H = r"[^\S\r\n]"
#: Separator between a field label and its value. German forms are usually
#: bilingual ("Name des Kindes / Name of the child: Emily"), so everything up
#: to the colon on the same line is skipped; without a colon the value follows
#: the label directly.
_SEP = r"(?:[^\n:]{0,40}:)?[^\S\r\n]*"

#: Generic labelled patterns, tried for every document type.
_GENERIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("last_name", re.compile(
        r"(?:surname|family\s*name|last\s*name|name\s*/\s*surname|nachname|familienname)"
        rf"{_SEP}({_NAME_CHARS}(?:{_H}+{_NAME_CHARS})?)", re.IGNORECASE)),
    ("first_name", re.compile(
        r"(?:given\s*names?|first\s*name|forename|vorname[n]?)"
        rf"{_SEP}({_NAME_CHARS}(?:{_H}+{_NAME_CHARS})?)", re.IGNORECASE)),
    ("company", re.compile(
        r"(?:employer|company|arbeitgeber|firma|unternehmen|host\s*company)"
        rf"{_SEP}([A-Z0-9ÄÖÜ][\w&.,'\-äöüß ]{{2,45}})", re.IGNORECASE)),
    ("city", re.compile(
        r"(?:city|town|stadt|wohnort|ort\s*der\s*anmeldung|gemeinde|municipality)"
        rf"{_SEP}({_NAME_CHARS}(?:{_H}{_NAME_CHARS})?)", re.IGNORECASE)),
)

#: Per-document-type anchors. These are the standardised forms where a regex is
#: as reliable as a language model - and, unlike a model, verifiable.
_TYPE_PATTERNS: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "09_marriage_certificate": (
        ("full_name", re.compile(
            rf"(?:husband|ehemann|spouse\s*1|partner\s*1){_SEP}({_NAME_CHARS}(?:{_H}+{_NAME_CHARS}){{0,2}})",
            re.IGNORECASE)),
        ("city", re.compile(rf"standesamt{_H}+({_NAME_CHARS}(?:{_H}{_NAME_CHARS})?)", re.IGNORECASE)),
    ),
    "10_birth_certificate": (
        ("dependent_name", re.compile(
            rf"(?:name\s*(?:des\s*kindes|of\s*the\s*child)|kind|child){_SEP}({_NAME_CHARS}(?:{_H}+{_NAME_CHARS})?)",
            re.IGNORECASE)),
        ("full_name", re.compile(
            rf"(?:vater|father){_SEP}({_NAME_CHARS}(?:{_H}+{_NAME_CHARS}){{0,2}})", re.IGNORECASE)),
        ("city", re.compile(rf"standesamt{_H}+({_NAME_CHARS}(?:{_H}{_NAME_CHARS})?)", re.IGNORECASE)),
    ),
    "12_payslip": (
        ("period", re.compile(
            rf"(?:abrechnungsmonat|abrechnungszeitraum|pay{_H}*period|lohnmonat){_SEP}"
            r"(\d{1,2}\s?[./-]\s?\d{4}|\d{4}\s?-\s?\d{1,2})", re.IGNORECASE)),
    ),
    "04_employment_contract": (
        # "zwischen <Firma> und <Name>" - the standard German contract opening.
        ("company", re.compile(
            r"zwischen\s+(?:der\s+|dem\s+)?([A-Z0-9ÄÖÜ][\w&.,'\-äöüß ]{2,50}?)\s+und\b", re.IGNORECASE)),
    ),
    "03_cv": (
        # A CV rarely labels anything; the name comes from the heading (see
        # PersonNameFinder) and the employer from the work experience section.
        ("city", re.compile(
            rf"(?:address|anschrift|wohnhaft\s*in){_SEP}(?:[^\n]{{0,40}}?\b\d{{5}}{_H}+)?({_NAME_CHARS})",
            re.IGNORECASE)),
    ),
    "05_assignment_letter": (
        ("company", re.compile(
            r"(?:host\s*company|gastunternehmen|aufnehmendes\s*unternehmen)\s*[:\-]?\s*"
            r"([A-Z0-9ÄÖÜ][\w&.,'\-äöüß ]{2,45})", re.IGNORECASE)),
    ),
}

#: City anchors for the town hall documents: the line after a German postcode.
_POSTCODE_CITY = re.compile(rf"\b\d{{5}}\s+({_NAME_CHARS}(?:[- ]{_NAME_CHARS})?)")

_EXPIRY_HINTS: tuple[str, ...] = (
    "date of expiry", "valid until", "valid till", "expiry date", "expires",
    "gültig bis", "gultig bis", "gueltig bis", "befristet bis", "date d'expiration",
)

_PERIOD_HINTS: tuple[str, ...] = (
    "abrechnungsmonat", "abrechnungszeitraum", "pay period", "period", "monat", "month",
)


def _trim_label_bleed(value: str) -> str:
    """Drop trailing words that are in fact the beginning of the next label."""
    words = value.split()
    while words and words[-1].strip(":,.").lower() in _LABEL_STOP_WORDS:
        words.pop()
    return " ".join(words)


def split_full_name(value: str) -> tuple[str, str]:
    """Split a full name into ``(first_name, last_name)``.

    Handles both ``"John Smith"`` and the German ``"Smith, John"`` order.
    """
    cleaned = _trim_label_bleed(value.strip())
    if not cleaned:
        return "", ""
    if "," in cleaned:
        last, _, first = cleaned.partition(",")
        return first.strip().split(" ")[0], last.strip()
    parts = cleaned.split()
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[-1]


# --------------------------------------------------------------------------- #
# Person, company and title finders
# --------------------------------------------------------------------------- #
#: Legal forms that mark a company name beyond doubt.
_LEGAL_FORMS = (
    r"(?:GmbH\s*&\s*Co\.?\s*KGaA|GmbH\s*&\s*Co\.?\s*KG|gGmbH|GmbH|mbH|AG|SE|KGaA|KG|OHG|GbR|"
    r"e\.?\s?K\.?|Ltd\.?|Limited|LLC|L\.L\.C\.|Inc\.?|Incorporated|Corp\.?|Corporation|PLC|"
    r"B\.?V\.?|N\.?V\.?|S\.?A\.?R\.?L\.?|S\.?A\.?S\.?|S\.?A\.?|S\.?p\.?A\.?|S\.?L\.?|Oy|AB|A/S|"
    r"Pty\.?\s*Ltd\.?|Sp\.\s?z\s?o\.o\.)"
)
_COMPANY_LEGAL = re.compile(
    rf"\b([A-ZÄÖÜ][\w.&'\-äöüß]*(?:[ ][A-ZÄÖÜ0-9][\w.&'\-äöüß]*){{0,4}}[ ]{_LEGAL_FORMS})(?!\w)"
)

#: Words that disqualify a line from being a person name.
_NOT_A_PERSON: frozenset[str] = frozenset({
    "curriculum", "vitae", "lebenslauf", "resume", "cv", "profile", "profil",
    "contact", "kontakt", "address", "adresse", "experience", "erfahrung",
    "education", "ausbildung", "skills", "kenntnisse", "languages", "sprachen",
    "certificate", "urkunde", "bescheinigung", "vertrag", "contract", "invoice",
    "rechnung", "sheet", "information", "travel", "business", "passport",
    "reisepass", "visa", "visum", "gmbh", "ag", "limited", "inc", "university",
    "universitat", "universität", "hochschule", "seite", "page", "datum", "date",
    "standesamt", "registrar", "amt", "stadt", "city", "republik", "republic",
    "bundesrepublik", "federal", "deutschland", "germany", "krankenkasse",
    "versicherung", "insurance", "arbeitsvertrag", "employment", "payslip",
    "gehaltsabrechnung", "lohnabrechnung", "anmeldung", "abmeldung",
})

_PERSON_TOKEN = re.compile(r"^[A-ZÄÖÜ][A-Za-zÄÖÜäöüß'\-]{1,20}$|^[A-ZÄÖÜ]{2,20}$")

#: Titles of documents outside the naming convention that appear regularly.
#: A detected heading is snapped onto the closest entry so the same document
#: always produces the same file name.
KNOWN_OTHER_TITLES: tuple[str, ...] = (
    "MB Business Contact",
    "Travel Information Sheet",
    "Cover Letter",
    "Application Form",
    "Data Privacy Consent",
    "Certificate of Enrolment",
    "Confirmation of Employment",
    "Letter of Recommendation",
    "Bank Statement",
    "Tax Assessment",
)

_TITLE_NOISE = re.compile(r"^(?:page|seite)\s*\d+", re.IGNORECASE)


def looks_like_person_name(value: str) -> bool:
    """True when *value* reads like a person name ("John Smith", "SMITH JOHN")."""
    cleaned = " ".join((value or "").replace(",", " ").split())
    if not cleaned or any(char.isdigit() for char in cleaned):
        return False
    tokens = cleaned.split(" ")
    if not 2 <= len(tokens) <= 3:
        return False
    if any(token.lower().strip(".") in _NOT_A_PERSON for token in tokens):
        return False
    return all(_PERSON_TOKEN.match(token) for token in tokens)


class PersonNameFinder:
    """Finds the applicant's name where no label exists - CVs above all.

    A CV usually carries the name as the largest text on page one and never
    writes "Surname:" in front of it, which is exactly why the labelled
    patterns missed it.
    """

    @staticmethod
    def from_headings(headings: Sequence[str]) -> tuple[str, str]:
        """Return ``(first_name, last_name)`` from the page headings."""
        for heading in headings:
            if looks_like_person_name(heading):
                return split_full_name(heading)
        return "", ""

    @staticmethod
    def from_first_lines(text: str, max_lines: int = 8) -> tuple[str, str]:
        """Return ``(first_name, last_name)`` from the top of the document."""
        for line in (text or "").splitlines()[:max_lines]:
            candidate = line.strip(" \t|-")
            if looks_like_person_name(candidate):
                return split_full_name(candidate)
        return "", ""


class CompanyFinder:
    """Finds a company name through its legal form (GmbH, AG, Ltd, Inc, ...)."""

    @staticmethod
    def find_all(text: str) -> list[str]:
        """Return every company-like name in document order."""
        seen: list[str] = []
        for match in _COMPANY_LEGAL.finditer(text or ""):
            name = " ".join(match.group(1).split())
            if name not in seen:
                seen.append(name)
        return seen

    @classmethod
    def find(cls, text: str, prefer_after: Sequence[str] = ()) -> str:
        """Return the most plausible employer name.

        Args:
            text: Document text.
            prefer_after: Section headings (e.g. "work experience") after which
                the first hit is preferred - on a CV that is the current
                employer rather than a university or a former one.
        """
        candidates = cls.find_all(text)
        if not candidates:
            return ""
        lowered = (text or "").lower()
        for marker in prefer_after:
            index = lowered.find(marker.lower())
            if index == -1:
                continue
            section = text[index : index + 600]
            preferred = cls.find_all(section)
            if preferred:
                return preferred[0]
        return candidates[0]


class TitleFinder:
    """Derives a file name title for documents outside the naming convention."""

    @staticmethod
    def normalise(value: str) -> str:
        """Clean a heading into a title usable inside a file name."""
        cleaned = " ".join((value or "").split()).strip(" :;.-_")
        if not cleaned:
            return ""
        for known in KNOWN_OTHER_TITLES:  # snap onto a canonical spelling
            if cleaned.lower() == known.lower():
                return known
        if cleaned.isupper() or cleaned.islower():
            small = {"of", "and", "the", "for", "in", "on", "zur", "der", "die", "das", "und"}
            words = [
                word if word.lower() in small and index else word.capitalize()
                for index, word in enumerate(cleaned.split(" "))
            ]
            cleaned = " ".join(words)
        return cleaned[:60].strip()

    @classmethod
    def find(cls, headings: Sequence[str], text: str) -> str:
        """Return the document's own title, or ``""`` when nothing fits."""
        for heading in headings:
            if looks_like_person_name(heading) or _TITLE_NOISE.match(heading):
                continue
            title = cls.normalise(heading)
            if title and len(title.split()) >= 2:
                return title
        for line in (text or "").splitlines()[:6]:
            candidate = line.strip()
            if not candidate or looks_like_person_name(candidate) or _TITLE_NOISE.match(candidate):
                continue
            title = cls.normalise(candidate)
            if title and 2 <= len(title.split()) <= 8:
                return title
        return ""

# --------------------------------------------------------------------------- #
# Passport MRZ with check digit verification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MrzData:
    """Result of reading a passport machine readable zone."""

    surname: str
    given_name: str
    document_number: str
    nationality: str
    expiry_date: str          # DD.MM.YYYY
    birth_date: str           # DD.MM.YYYY
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        """True when every check digit of the MRZ matched.

        A verified MRZ is *arithmetically* confirmed, not guessed - which is
        exactly why the rule engine may trust it more than a model's reading.
        """
        return bool(self.checks) and all(self.checks.values())


class MrzParser:
    """Parses and verifies the TD3 machine readable zone of a passport."""

    _WEIGHTS = (7, 3, 1)

    @classmethod
    def check_digit(cls, value: str) -> int:
        """Compute the ICAO 9303 check digit of *value*."""
        total = 0
        for index, char in enumerate(value):
            if char.isdigit():
                digit = int(char)
            elif char == "<":
                digit = 0
            elif char.isalpha():
                digit = ord(char.upper()) - 55  # A=10 ... Z=35
            else:
                digit = 0
            total += digit * cls._WEIGHTS[index % 3]
        return total % 10

    @classmethod
    def parse(cls, text: str) -> MrzData | None:
        """Extract and verify the MRZ from *text*, or return ``None``."""
        compact = re.sub(r"[ \t]", "", text.upper())
        line1 = _MRZ_LINE1.search(compact)
        if not line1:
            return None

        surname = line1.group(1).replace("<", " ").strip()
        given = line1.group(2).replace("<", " ").strip()

        line2 = _MRZ_LINE2.search(compact[line1.end():]) or _MRZ_LINE2.search(compact)
        if not line2:
            return MrzData(
                surname=surname.title(),
                given_name=given.split(" ")[0].title() if given else "",
                document_number="", nationality="", expiry_date="", birth_date="",
            )

        number, number_cd, nationality, birth, birth_cd, _sex, expiry, expiry_cd, personal, personal_cd, composite_cd = (
            line2.groups()
        )
        checks = {
            "document_number": cls.check_digit(number) == int(number_cd),
            "birth_date": cls.check_digit(birth) == int(birth_cd),
            "expiry_date": cls.check_digit(expiry) == int(expiry_cd),
            "composite": cls.check_digit(
                f"{number}{number_cd}{birth}{birth_cd}{expiry}{expiry_cd}{personal}{personal_cd}"
            ) == int(composite_cd),
        }
        return MrzData(
            surname=surname.title(),
            given_name=given.split(" ")[0].title() if given else "",
            document_number=number.replace("<", ""),
            nationality=nationality.replace("<", ""),
            expiry_date=cls._mrz_date(expiry, future=True),
            birth_date=cls._mrz_date(birth, future=False),
            checks=checks,
        )

    @staticmethod
    def _mrz_date(raw: str, future: bool) -> str:
        """Convert a ``YYMMDD`` MRZ date into ``DD.MM.YYYY``."""
        try:
            year, month, day = int(raw[0:2]), int(raw[2:4]), int(raw[4:6])
            if not (1 <= month <= 12 and 1 <= day <= 31):
                return ""
        except (ValueError, IndexError):
            return ""
        century = 2000 if future or year <= (datetime.now().year % 100) else 1900
        return f"{day:02d}.{month:02d}.{century + year:04d}"


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

        The confidence returned here reflects the *keyword* evidence only;
        :class:`RuleFieldExtractor` raises it when hard anchors are found.
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


def required_fields(spec: DocumentTypeSpec) -> set[str]:
    """Return every field the file name of *spec* needs."""
    required = {"first_name", "last_name", "company"}
    if spec.requires_validity:
        required.add("valid_until")
    if spec.requires_period:
        required.add("period")
    if spec.requires_city:
        required.add("city")
    if spec.requires_title:
        required.add("title")
    return required


def score_confidence(
    result: ExtractionResult, spec: DocumentTypeSpec, settings: Settings = SETTINGS
) -> float:
    """Turn the evidence recorded on *result* into an honest confidence value.

    Can be called again after a field was filled in later (for example when the
    employer was taken from another document of the same batch), because every
    input is stored on the result itself.

    A guessed value never lifts a result over the review threshold, and a
    missing file name field always keeps it below - the name would otherwise
    carry ``UNKNOWN``.
    """
    threshold = settings.confidence_threshold
    guessed = any("guessed" in item for item in result.evidence)
    anchored = {item.split(" ")[0] for item in result.evidence if " from " in item}
    verified = any("MRZ verified" in item for item in result.evidence)
    required = required_fields(spec)

    if verified:
        confidence = 0.95
    elif required.issubset(anchored) and all(getattr(result, name, "") for name in required) and not guessed:
        confidence = settings.rule_trusted_confidence
    else:
        confidence = min(result.keyword_confidence, settings.heuristic_confidence_cap)

    if spec.key == UNKNOWN_TYPE_KEY:
        confidence = min(confidence, 0.30)
    if spec.key == OTHER_TYPE_KEY and not result.title:
        confidence = min(confidence, 0.30)
    if any(not getattr(result, name, "") for name in required):
        confidence = min(confidence, threshold - 0.02)
    if guessed:
        confidence = min(confidence, threshold - 0.02)
    return round(max(0.05, confidence), 3)


class RuleFieldExtractor:
    """Extracts the mandatory fields with verifiable rules.

    Order of evidence, strongest first:

    1. a passport MRZ whose check digits all match,
    2. an anchor defined for this exact document type,
    3. a generic labelled field (``Surname:``, ``Arbeitgeber:``),
    4. a positional guess (a date near "valid until").

    The confidence reflects that order, and every match is recorded in
    ``result.evidence`` so the review screen can show *why* a value is there.
    """

    def __init__(self, settings: Settings = SETTINGS) -> None:
        self._settings = settings

    #: CV sections after which the first company found is the current employer.
    _EMPLOYER_SECTIONS = (
        "work experience", "professional experience", "berufserfahrung",
        "employment history", "beruflicher werdegang", "arbeitgeber",
    )

    def extract(
        self,
        text: str,
        document_type: str,
        keyword_confidence: float,
        headings: Sequence[str] = (),
    ) -> ExtractionResult:
        """Extract every field the rules can reach from *text*.

        Args:
            text: Plain text of the (sub-)document.
            document_type: Type key detected by the keyword classifier.
            keyword_confidence: Confidence of that keyword match.
            headings: Lines set in a larger font, used for unlabelled names
                (CVs) and for the title of documents outside the convention.
        """
        spec = get_spec(document_type)
        result = ExtractionResult(
            document_type=document_type, confidence=keyword_confidence, source="rules"
        )

        self._apply_mrz(text, result)
        self._apply_type_patterns(text, result, spec)
        self._apply_generic_patterns(text, result)
        self._apply_finders(text, result, spec, headings)
        self._apply_positional(text, result, spec)
        spec = get_spec(result.document_type)  # may have become "other"

        if spec.relationship_in_label:
            result.relationship = "spouse" if spec.key.endswith("spouse") else "child"
        if result.dependent_name and not result.relationship:
            result.relationship = "child"

        result.keyword_confidence = keyword_confidence
        result.confidence = score_confidence(result, spec, self._settings)
        return result

    # -- evidence sources -------------------------------------------------- #
    def _apply_mrz(self, text: str, result: ExtractionResult) -> MrzData | None:
        """Read names and expiry from a passport MRZ (with check digits)."""
        mrz = MrzParser.parse(text)
        if not mrz:
            return None
        if mrz.surname:
            result.last_name = mrz.surname
        if mrz.given_name:
            result.first_name = mrz.given_name
        if mrz.expiry_date:
            result.valid_until = mrz.expiry_date
        if mrz.is_verified:
            result.evidence.append("MRZ verified (all check digits match)")
            logger.info(
                "MRZ verified: %s %s, expires %s", mrz.given_name, mrz.surname, mrz.expiry_date
            )
        else:
            failed = [name for name, ok in mrz.checks.items() if not ok]
            result.evidence.append(
                "MRZ read" + (f", failed checks: {', '.join(failed)}" if failed else ", unverified")
            )
        return mrz

    def _apply_type_patterns(self, text: str, result: ExtractionResult, spec: DocumentTypeSpec) -> None:
        """Apply the anchors defined for this document type."""
        for field_name, pattern in _TYPE_PATTERNS.get(spec.key, ()):  # type: ignore[arg-type]
            match = pattern.search(text)
            if not match:
                continue
            value = _trim_label_bleed(match.group(1).strip())
            if not value:
                continue
            if field_name == "full_name":
                first, last = split_full_name(value)
                if last and not result.last_name:
                    result.last_name = last
                    result.evidence.append(f"last_name from {spec.key} anchor")
                if first and not result.first_name:
                    result.first_name = first
                    result.evidence.append(f"first_name from {spec.key} anchor")
                continue
            if field_name == "dependent_name":
                value = split_full_name(value)[0] or value
            if not getattr(result, field_name, ""):
                setattr(result, field_name, value)
                result.evidence.append(f"{field_name} from {spec.key} anchor")

        if spec.requires_city and not result.city:
            postcode = _POSTCODE_CITY.search(text)
            if postcode:
                result.city = _trim_label_bleed(postcode.group(1))
                result.evidence.append("city from postal code line")

    def _apply_generic_patterns(self, text: str, result: ExtractionResult) -> None:
        """Apply the labelled patterns that work across document types."""
        for field_name, pattern in _GENERIC_PATTERNS:
            if getattr(result, field_name):
                continue
            match = pattern.search(text)
            if not match:
                continue
            value = _trim_label_bleed(match.group(1).strip())
            if value:
                setattr(result, field_name, value)
                result.evidence.append(f"{field_name} from label")

    def _apply_finders(
        self,
        text: str,
        result: ExtractionResult,
        spec: DocumentTypeSpec,
        headings: Sequence[str],
    ) -> None:
        """Fill name, company and title where no label exists.

        This is what a CV needs: the name is simply the largest line on page
        one, and the employer is recognisable by its legal form.
        """
        if not (result.first_name and result.last_name):
            first, last = PersonNameFinder.from_headings(headings)
            source = "document heading"
            if not last:
                first, last = PersonNameFinder.from_first_lines(text)
                source = "first lines"
            if last:
                if not result.last_name:
                    result.last_name = last
                    result.evidence.append(f"last_name from {source}")
                if first and not result.first_name:
                    result.first_name = first
                    result.evidence.append(f"first_name from {source}")

        if not result.company:
            prefer = self._EMPLOYER_SECTIONS if spec.key == "03_cv" else ()
            company = CompanyFinder.find(text, prefer_after=prefer)
            if company:
                result.company = company
                result.evidence.append("company from legal form")

        # A readable document that matches no catalogue entry is not "unknown":
        # it is simply another document and keeps the general naming rule.
        if result.document_type == UNKNOWN_TYPE_KEY:
            title = TitleFinder.find(headings, text)
            if title:
                result.document_type = OTHER_TYPE_KEY
                result.title = title
                result.evidence.append(
                    "title from heading" if any(
                        TitleFinder.normalise(h) == title for h in headings
                    ) else "title from first lines"
                )
                logger.info("Document outside the convention - titled %r", title)
        elif spec.requires_title and not result.title:
            title = TitleFinder.find(headings, text)
            if title:
                result.title = title
                result.evidence.append("title from heading")

    def _apply_positional(self, text: str, result: ExtractionResult, spec: DocumentTypeSpec) -> None:
        """Dates next to an explicit hint (an anchor), or a last-resort guess."""
        if spec.requires_validity and not result.valid_until:
            found, from_label = self._find_expiry(text)
            if found:
                result.valid_until = found
                result.evidence.append(
                    "valid_until from label" if from_label else "valid_until guessed from context"
                )
        if spec.requires_period and not result.period:
            found, from_label = self._find_period(text)
            if found:
                result.period = found
                result.evidence.append(
                    "period from label" if from_label else "period guessed from context"
                )

    # -- helpers ----------------------------------------------------------- #
    def _find_expiry(self, text: str) -> tuple[str, bool]:
        """Return ``(date, from_label)`` for the most plausible validity date.

        ``from_label`` is True when the date sits next to an explicit hint such
        as "valid until" or "gültig bis"; that counts as a real anchor. The
        fallback - the latest future date anywhere in the document - is a guess
        and keeps the result below the review threshold.
        """
        lowered = text.lower()
        for hint in _EXPIRY_HINTS:
            index = lowered.find(hint)
            if index == -1:
                continue
            dates = find_dates(text[index : index + 120])
            if dates:
                return normalise_date_string(dates[0]), True
        future = [d for d in find_dates(text) if d.year >= 2000]
        return (normalise_date_string(max(future)), False) if future else ("", False)

    def _find_period(self, text: str) -> tuple[str, bool]:
        """Return ``(period, from_label)`` for the payslip period (``YYYY-MM``)."""
        lowered = text.lower()
        for hint in _PERIOD_HINTS:
            index = lowered.find(hint)
            if index == -1:
                continue
            period = normalise_period(text[index : index + 60])
            if period:
                return period, True
        dates = find_dates(text)
        return (normalise_period(dates[0]), False) if dates else ("", False)


class RuleSegmenter:
    """Detects document boundaries inside a combined PDF without any model.

    Each page is scored independently; consecutive pages that resolve to the
    same type (or that carry no strong signal of their own) are merged into one
    segment, which mirrors how combined client PDFs are usually assembled.
    """

    def __init__(
        self,
        classifier: KeywordClassifier | None = None,
        extractor: RuleFieldExtractor | None = None,
        settings: Settings = SETTINGS,
    ) -> None:
        self._classifier = classifier or KeywordClassifier()
        self._extractor = extractor or RuleFieldExtractor(settings)
        self._settings = settings

    def segment(self, document: DocumentText) -> list[ClassifiedSegment]:
        """Split *document* into classified page ranges."""
        cap = self._settings.heuristic_confidence_cap
        if not document.pages:
            return []

        page_types = [
            (page.number, *self._classifier.classify(page.text, cap)) for page in document.pages
        ]

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
            result = self._extractor.extract(
                text, key, confidence, headings=document.headings_for_range(start, end)
            )
            segments.append(ClassifiedSegment(start_page=start, end_page=end, result=result))

        logger.info(
            "Rule engine segmented %s into %d document(s): %s",
            document.path.name, len(segments),
            ", ".join(
                f"{s.page_label}={s.result.document_type} ({s.result.confidence:.0%})"
                for s in segments
            ),
        )
        return segments


#: Backwards compatible aliases (the classes were renamed when the rule engine
#: was promoted from "fallback" to "first stage").
HeuristicFieldExtractor = RuleFieldExtractor
HeuristicSegmenter = RuleSegmenter


# --------------------------------------------------------------------------- #
# Facade: rules first, local LLM as the fallback
# --------------------------------------------------------------------------- #
class DocumentClassifier:
    """Turns the text of an uploaded PDF into classified page ranges.

    Strategy (``RULES_FIRST=true``, the default):

    1. The deterministic rule engine classifies, splits and extracts. Documents
       it resolves with hard evidence (a checksum-verified passport MRZ, or
       every file name field read from a labelled anchor) are done here - no
       model is asked, nothing leaves the machine, and the result is
       reproducible.
    2. Only the segments the rules could *not* resolve are handed to the LLM
       fallback (a local Ollama model by default, Azure OpenAI when configured
       that way). If the page grouping itself looks doubtful, the whole
       document is re-analysed by the model.
    3. Verified rule data always wins over model output when both exist.

    With no LLM available the pipeline still works: the weak documents simply
    stay below the confidence threshold and land in the review queue.
    """

    def __init__(
        self,
        settings: Settings = SETTINGS,
        llm_extractor: Any | None = None,
        segmenter: RuleSegmenter | None = None,
    ) -> None:
        self._settings = settings
        self._rules = segmenter or RuleSegmenter(settings=settings)
        if llm_extractor is None:
            # Imported lazily: modules.llm_providers imports this module.
            from modules.llm_providers import build_llm_extractor

            llm_extractor = build_llm_extractor(settings)
        self._llm = llm_extractor

    # -- properties -------------------------------------------------------- #
    @property
    def ai_enabled(self) -> bool:
        """True when an LLM fallback is available."""
        return bool(self._llm and self._llm.is_available)

    @property
    def llm_name(self) -> str:
        """Name of the active fallback backend, or ``"rules only"``."""
        return self._llm.display_name if self.ai_enabled else "rules only"

    # -- public API -------------------------------------------------------- #
    def analyze(self, document: DocumentText) -> list[ClassifiedSegment]:
        """Classify *document* and return its segments (never empty)."""
        if not self._settings.rules_first and self.ai_enabled:
            # Model-first mode: kept for comparison and for very messy scans.
            try:
                return self._normalise(self._llm.analyze(document), document)
            except ClassificationError as exc:
                logger.error("LLM-first analysis failed for %s: %s", document.path.name, exc)

        segments = self._rules.segment(document)
        if not segments:
            segments = [
                ClassifiedSegment(
                    1,
                    max(document.page_count, 1),
                    ExtractionResult(
                        document_type=UNKNOWN_TYPE_KEY,
                        confidence=0.0,
                        notes="no text could be extracted",
                    ),
                )
            ]

        weak = [segment for segment in segments if self._needs_help(segment)]
        if not weak:
            logger.info(
                "%s fully resolved by the rule engine - no model call needed", document.path.name
            )
            return self._normalise(segments, document)

        if not self.ai_enabled:
            logger.info(
                "%d document(s) of %s need review: no LLM fallback configured",
                len(weak), document.path.name,
            )
            return self._normalise(segments, document)

        if self._grouping_uncertain(segments):
            improved = self._retry_whole_document(document, segments)
            if improved is not None:
                return self._normalise(improved, document)

        for segment in weak:
            self._improve_segment(segment, document)
        return self._normalise(segments, document)

    # -- fallback strategy ------------------------------------------------- #
    def _needs_help(self, segment: ClassifiedSegment) -> bool:
        """True when the rules did not resolve this segment convincingly."""
        result = segment.result
        if result.document_type == UNKNOWN_TYPE_KEY:
            return True
        if result.confidence < self._settings.confidence_threshold:
            return True
        spec = get_spec(result.document_type)
        missing = not (result.first_name and result.last_name and result.company)
        if spec.requires_title and not result.title:
            missing = True
        if spec.requires_validity and not result.valid_until:
            missing = True
        if spec.requires_period and not result.period:
            missing = True
        if spec.requires_city and not result.city:
            missing = True
        return missing

    def _grouping_uncertain(self, segments: Sequence[ClassifiedSegment]) -> bool:
        """True when the page grouping itself should be re-checked by the model."""
        return any(segment.result.document_type == UNKNOWN_TYPE_KEY for segment in segments)

    def _retry_whole_document(
        self, document: DocumentText, rule_segments: Sequence[ClassifiedSegment]
    ) -> list[ClassifiedSegment] | None:
        """Let the model re-segment the document, keeping verified rule data."""
        try:
            llm_segments = self._llm.analyze(document)
        except ClassificationError as exc:
            logger.error("LLM re-analysis failed for %s: %s", document.path.name, exc)
            return None
        except Exception as exc:  # defensive: a backend must not kill the batch
            logger.exception("Unexpected LLM failure for %s: %s", document.path.name, exc)
            return None

        for llm_segment in llm_segments:
            for rule_segment in rule_segments:
                if not self._overlaps(llm_segment, rule_segment):
                    continue
                if self._is_verified(rule_segment.result):
                    self._apply_verified(rule_segment.result, llm_segment.result)
        return list(llm_segments)

    def _improve_segment(self, segment: ClassifiedSegment, document: DocumentText) -> None:
        """Ask the model about one weak segment and merge what it returns."""
        text = document.text_for_range(segment.start_page, segment.end_page)
        if not text.strip():
            return
        spec = get_spec(segment.result.document_type)
        hint = spec.description if spec.key != UNKNOWN_TYPE_KEY else ""
        improved = self._llm.analyze_single(text, segment.page_count, hint=hint)
        if improved is None:
            logger.info("LLM could not improve %s of %s", segment.page_label, document.path.name)
            return

        before = segment.result
        improved.evidence = list(before.evidence) + [f"fields completed by {self._llm.display_name}"]
        if self._is_verified(before):
            self._apply_verified(before, improved)
        # Keep rule values the model left empty.
        for name in ("first_name", "last_name", "company", "city", "valid_until", "period"):
            if not getattr(improved, name, "") and getattr(before, name, ""):
                setattr(improved, name, getattr(before, name))
        segment.result = improved
        logger.info(
            "%s of %s completed by %s (%s -> %s, %.0f%%)",
            segment.page_label, document.path.name, self._llm.display_name,
            before.document_type, improved.document_type, improved.confidence * 100,
        )

    # -- merge helpers ----------------------------------------------------- #
    @staticmethod
    def _overlaps(left: ClassifiedSegment, right: ClassifiedSegment) -> bool:
        """True when two segments share at least one page."""
        return left.start_page <= right.end_page and right.start_page <= left.end_page

    @staticmethod
    def _is_verified(result: ExtractionResult) -> bool:
        """True when the rule engine verified this result arithmetically."""
        return any("MRZ verified" in item for item in result.evidence)

    @staticmethod
    def _apply_verified(verified: ExtractionResult, target: ExtractionResult) -> None:
        """Overwrite model output with checksum-verified rule data."""
        target.document_type = verified.document_type
        for name in ("first_name", "last_name", "valid_until"):
            value = getattr(verified, name, "")
            if value:
                setattr(target, name, value)
        target.evidence = list(dict.fromkeys(list(target.evidence) + list(verified.evidence)))
        target.confidence = max(target.confidence, verified.confidence)
        logger.debug("Verified rule data kept over model output for %s", verified.document_type)

    # -- normalisation ----------------------------------------------------- #
    def _normalise(
        self, segments: Sequence[ClassifiedSegment], document: DocumentText
    ) -> list[ClassifiedSegment]:
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
            # Attach stray pages to the nearest segment instead of losing them.
            for number in missing:
                target = min(cleaned, key=lambda seg: abs(seg.start_page - number))
                target.end_page = max(target.end_page, number)
                target.start_page = min(target.start_page, number)
                target.page_count = target.end_page - target.start_page + 1
            logger.info("Re-attached %d unassigned page(s) of %s", len(missing), document.path.name)
        return cleaned
