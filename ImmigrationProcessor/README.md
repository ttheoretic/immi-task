# Immigration Document Processor

Desktop application for immigration case workers: drop client PDFs in, get
convention-compliant, reviewed files in the right client folder.

---

## Schnellstart (Deutsch)

**macOS / Linux**

```bash
cd ImmigrationProcessor
./run.sh
```

**Windows**

Doppelklick auf `run.bat` (oder `run.bat` in der Eingabeaufforderung).

Der Starter legt beim ersten Mal automatisch eine virtuelle Umgebung an,
installiert die Abhängigkeiten, erzeugt Demo-PDFs und öffnet die App im
Browser unter <http://localhost:8501>. Ein anderer Port:
`PORT=9000 ./run.sh` bzw. `set PORT=9000 && run.bat`.

**Direkt testen:** Die Demo-PDFs liegen nach dem ersten Start in
`data/incoming/samples/`. Zieh sie ins Upload-Feld und klick
*Analyze documents*:

| Datei | Was sie zeigt |
| --- | --- |
| `combined_client.pdf` | Ein PDF mit **drei** Dokumenten (Pass S. 1-2, Visum S. 3, Heiratsurkunde S. 4-5) - wird automatisch getrennt |
| `payslip_july.pdf` | Gehaltsabrechnung, Benennung mit Periode `YYYY-MM` |
| `birth_certificate_child.pdf` | Dokument eines Kindes (`child 1_Emily`) |
| `scanned_passport.pdf` | Reiner Scan ohne Textebene - zeigt den OCR-Pfad |

Ohne Azure-Zugangsdaten läuft die App im deterministischen Keyword-Modus;
die Konfidenz ist dabei auf 55 % gedeckelt, also landet **jedes** Dokument in
`REVIEW REQUIRED`. Genau so ist es gedacht: Nichts wird ungeprüft exportiert.
Im Review-Bereich Felder korrigieren, *Reviewed - approve for export*
anhaken, *Apply corrections*, dann *Export all* und *Create ZIP*.

Für den vollen KI-Betrieb die Werte in `.env` eintragen
(`AZURE_OPENAI_*`, `DOCUMENT_INTELLIGENCE_*`) und neu starten.

---

## In der Claude Code Desktop-App (ohne Terminal)

Die Desktop-App kann die App selbst starten und im **Browser-Pane** anzeigen.
Die passende Konfiguration liegt im Repo unter `.claude/launch.json`.

1. Session in der Desktop-App öffnen (im Web-Chat oben rechts über die drei
   Punkte: **Öffnen in → Desktop App**).
2. Wichtig: Die Session muss **lokal** laufen (Environment-Dropdown → *Local*),
   damit `localhost` dein Rechner ist. Eine reine Cloud-Session startet den
   Server im Container, den der Browser-Pane nicht erreicht.
3. Im Server-Dropdown die Konfiguration für dein System wählen:
   *Immigration Processor (macOS/Linux)* oder *(Windows)* - oder Claude
   einfach bitten, die App zu starten.

Der Preview-Server läuft dann auf `http://localhost:8501`. Beim ersten Start
legt er Umgebung, Abhängigkeiten und Demo-PDFs selbst an (dauert ca. 30 s);
danach startet er in wenigen Sekunden. `NO_BROWSER=1` sorgt dafür, dass kein
zweites Browserfenster neben dem Pane aufgeht.

---

## Requirements

* Python 3.10 or newer
* Optional for local OCR without Azure: the tesseract binary
  (`brew install tesseract tesseract-lang` / `apt-get install tesseract-ocr tesseract-ocr-deu`)

## Workflow

```
Upload -> Analyze -> Classify -> Split -> Rename -> Review -> Export
```

1. **Analyze** - PyMuPDF reads the text layer; pages without one go to Azure
   Document Intelligence (`prebuilt-read`), or to a local tesseract fallback.
2. **Classify** - Azure OpenAI detects document boundaries, the document type
   and the personal data. Without credentials a keyword/MRZ/regex engine takes
   over with a capped confidence.
3. **Split** - one PDF per detected document.
4. **Rename** - `<DocumentType>_<LastName>_<FirstName>_<CompanyName>.pdf`,
   e.g. `01_passport_valid 15.08.2032_Smith_John_Microsoft.pdf`.
   Dependants insert a token: `01_passport_spouse_...`,
   `10_birth certificate_child 1_Emily_...`.
5. **Review** - everything below the confidence threshold (default 90 %) is
   flagged `REVIEW REQUIRED` and can be corrected in the UI.
6. **Export** - approved files go to `Clients/<LastName_FirstName>/`,
   the rest to `data/review/`. ZIP downloads per client or for all clients.

## Layout

```
ImmigrationProcessor/
├── app.py                 Streamlit UI (no business logic)
├── config.py              settings from .env + logging
├── create_test_pdfs.py    demo PDFs for testing
├── run.sh / run.bat       one click launchers
├── data/                  incoming | processed (Clients/) | review | temp
├── modules/               ocr, classifier, pdf_splitter, renamer,
│                          folder_manager, export, validation, pipeline
├── prompts/               Azure OpenAI system prompt
└── logs/processing.log    every action, with confidence and file name
```

## Configuration

All settings live in `.env` (see the comments in that file). The most relevant:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIDENCE_THRESHOLD` | `0.90` | below this: `REVIEW REQUIRED`, no auto export |
| `HEURISTIC_CONFIDENCE_CAP` | `0.55` | ceiling for results produced without AI |
| `OCR_MIN_CHARS_PER_PAGE` | `60` | fewer characters on a page: treat it as scanned |
| `CLIENTS_ROOT` | `data/processed/Clients` | where the client folders live |
| `MAX_UPLOAD_MB` | `50` | upload size limit |

> `.env` ships with placeholder values only. Never commit real keys.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Python 3 was not found` | Install Python 3.10+; on Windows tick *Add python.exe to PATH* |
| Port already in use | `PORT=9000 ./run.sh` |
| Browser does not open | Open <http://localhost:8501> manually |
| Everything says `REVIEW REQUIRED` | Expected without Azure credentials - fill in `.env` |
| Dependencies look broken | Delete `.venv` and start the launcher again |
