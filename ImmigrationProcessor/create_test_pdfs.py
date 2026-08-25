"""Create demo PDFs so the application can be tested without real client data.

Run it directly (or let ``run.sh`` / ``run.bat`` do it on the first start)::

    python create_test_pdfs.py

The files are written to ``data/incoming/samples`` and can be dropped straight
into the upload area:

* ``combined_client.pdf``  - passport (p1-2) + visa (p3) + marriage certificate
  (p4-5) in one file; demonstrates the multi-document split.
* ``payslip_july.pdf``     - single payslip, demonstrates the ``YYYY-MM`` rule.
* ``birth_certificate_child.pdf`` - dependant document (child naming rule).
* ``scanned_passport.pdf`` - image only, no text layer; demonstrates the OCR
  path (and, without Azure credentials, the REVIEW REQUIRED gate).

The content is fictitious.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import pymupdf  # PyMuPDF >= 1.24.3
except ImportError:  # pragma: no cover - older PyMuPDF
    try:
        import fitz as pymupdf
    except ImportError:
        print("PyMuPDF is missing - run: pip install -r requirements.txt")
        raise SystemExit(1)

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "incoming" / "samples"

PASSPORT_PAGE_1 = """REISEPASS / PASSPORT / PASSEPORT
Type: P   Code: D   Passport No: C01X00T47
Surname / Nachname: SMITH
Given names / Vorname: JOHN
Nationality / Staatsangehoerigkeit: BRITISH
Date of birth: 12.03.1985
Date of issue: 16.08.2022
Date of expiry / Gueltig bis: 15.08.2032
Authority: London
P<GBRSMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<
C01X00T478GBR8503127M3208151<<<<<<<<<<<<<<06
"""

PASSPORT_PAGE_2 = """PASSPORT - observations / Vermerke
Passport number C01X00T47
Amendments and endorsements - none
Nationality BRITISH, date of issue 16.08.2022
"""

VISA_PAGE = """VISA / VISUM - SCHENGEN STATES
Category / Type: D    Number of entries: MULT
Valid for: GERMANY    Duration of stay: 365 days
Valid until / Gueltig bis: 30.09.2028
Surname: SMITH   Given names: JOHN
Employer: Microsoft
Remarks: Erwerbstaetigkeit gestattet
"""

MARRIAGE_PAGE_1 = """MARRIAGE CERTIFICATE - HEIRATSURKUNDE
Standesamt Munich
Date of marriage: 04.06.2016
Husband: John Smith, born 12.03.1985
Wife: Maria Smith, born 21.07.1987
"""

MARRIAGE_PAGE_2 = """Certificate of marriage - continued
Entry number 421/2016
Registrar, Munich
"""

PAYSLIP_PAGE = """Gehaltsabrechnung / Payslip
Arbeitgeber: Microsoft Deutschland GmbH
Nachname: Smith    Vorname: John
Abrechnungsmonat: 07/2026
Steuerklasse 1     Sozialversicherung
Gross pay 8.500,00 EUR    Net pay 4.900,00 EUR
"""

BIRTH_CERTIFICATE_PAGE = """GEBURTSURKUNDE - BIRTH CERTIFICATE
Standesamt Munich
Name des Kindes / Name of the child: Emily Smith
Date of birth: 02.02.2019
Vater / Father: John Smith
Mutter / Mother: Maria Smith
"""


def _write_text_pdf(path: Path, pages: list[str]) -> None:
    """Write a PDF with one text page per entry of *pages*."""
    document = pymupdf.open()
    try:
        for text in pages:
            page = document.new_page()
            page.insert_textbox(pymupdf.Rect(45, 45, 550, 780), text, fontsize=11)
        document.save(path)
    finally:
        document.close()


def _write_scanned_pdf(path: Path, text: str) -> None:
    """Write an image-only PDF (no text layer) to exercise the OCR path."""
    source = pymupdf.open()
    target = pymupdf.open()
    try:
        page = source.new_page()
        page.insert_textbox(pymupdf.Rect(45, 45, 550, 780), text, fontsize=13)
        pixmap = page.get_pixmap(dpi=140)
        try:  # JPEG keeps the demo file small
            image_bytes = pixmap.tobytes("jpeg", jpg_quality=70)
        except (ValueError, TypeError, RuntimeError):  # pragma: no cover - old PyMuPDF
            image_bytes = pixmap.tobytes("png")

        scanned_page = target.new_page(width=page.rect.width, height=page.rect.height)
        scanned_page.insert_image(scanned_page.rect, stream=image_bytes)
        target.save(path, deflate=True, garbage=3)
    finally:
        source.close()
        target.close()


def main() -> int:
    """Create every demo PDF and report where they went."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _write_text_pdf(
        OUTPUT_DIR / "combined_client.pdf",
        [PASSPORT_PAGE_1, PASSPORT_PAGE_2, VISA_PAGE, MARRIAGE_PAGE_1, MARRIAGE_PAGE_2],
    )
    _write_text_pdf(OUTPUT_DIR / "payslip_july.pdf", [PAYSLIP_PAGE])
    _write_text_pdf(OUTPUT_DIR / "birth_certificate_child.pdf", [BIRTH_CERTIFICATE_PAGE])
    _write_scanned_pdf(OUTPUT_DIR / "scanned_passport.pdf", PASSPORT_PAGE_1)

    print(f"Demo PDFs created in: {OUTPUT_DIR}")
    for pdf in sorted(OUTPUT_DIR.glob("*.pdf")):
        print(f"  - {pdf.name} ({pdf.stat().st_size / 1024:.1f} KB)")
    print("\nDrag them into the upload area of the running app.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
