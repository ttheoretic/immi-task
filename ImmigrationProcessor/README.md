# Immigration Document Processor

Desktop application for immigration case workers: drop client PDFs in, get
convention-compliant, reviewed files in the right client folder.

---

## Schnellstart (Deutsch)

Die App ist eine lokale Web-App: Sie startet einen Server auf deinem Rechner
und öffnet sich in deinem normalen Browser unter <http://localhost:8501>.
Nötig sind nur Python 3.10+ und die Dateien auf deinem Rechner.

**macOS - ohne Terminal**

Ordner `ImmigrationProcessor` im Finder öffnen und `run.command` doppelklicken.

**macOS / Linux - im Terminal**

```bash
cd ImmigrationProcessor
./run.sh
```

**Windows**

Doppelklick auf `run.bat` (oder `run.bat` in der Eingabeaufforderung).

**Code auf den Rechner holen** (einmalig), entweder

```bash
git clone -b claude/immigration-document-processor-knw7i0 \
  https://github.com/ttheoretic/immi-task.git
```

oder auf GitHub über **Code → Download ZIP** und entpacken.

Beim ZIP-Weg markiert macOS die Dateien als Download (Quarantäne-Flag), und
der erste Doppelklick wird mit *"stammt von einem nicht verifizierten
Entwickler"* abgelehnt. Drei Wege daran vorbei:

* **Systemeinstellungen → Datenschutz & Sicherheit** öffnen, dort steht direkt
  nach dem abgelehnten Versuch **"Dennoch öffnen"**. (Auf macOS Sequoia der
  einzige Klickweg - der frühere Rechtsklick → *Öffnen* ist dort deaktiviert;
  auf Sonoma und älter funktioniert er noch.)
* **Im Terminal starten**: `./run.sh` - Gatekeeper prüft nur Starts über den
  Finder, nicht über das Terminal.
* **Quarantäne-Flag entfernen**:
  `xattr -dr com.apple.quarantine <entpackter Ordner>`

Mit `git clone` entsteht das Flag gar nicht erst; dort funktioniert der
Doppelklick sofort.

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

## Two stages: rules first, model only as fallback

The extraction runs in two stages, and the first one needs no model at all:

1. **Rule engine** (always on, deterministic, reproducible, offline)
   * passport MRZ **including all ICAO check digits** - a verified MRZ is
     arithmetically confirmed rather than guessed, and always wins over model
     output,
   * anchors per document type (`Husband:`, `Abrechnungsmonat:`,
     `zwischen <Firma> und`, the city behind a German postal code),
   * generic labelled fields (`Surname:`, `Arbeitgeber:`, `gültig bis`).
2. **LLM fallback** - only the documents the rules could *not* resolve are
   handed to a model: a local Ollama by default, Azure OpenAI when configured
   that way. If the page grouping itself looks doubtful, the whole document is
   re-analysed by the model.

The confidence follows the evidence: a verified MRZ scores 0.95, every file
name field from a labelled anchor scores `RULE_TRUSTED_CONFIDENCE` (0.92), and
anything guessed stays below the review threshold. A missing company always
keeps a document in review - the file name would otherwise carry `UNKNOWN`.

### Local model with Ollama (no API, no data leaving the machine)

```bash
# 1. install from https://ollama.com, then pull a model
ollama pull qwen2.5:7b-instruct
# 2. in .env
LLM_PROVIDER=auto            # a reachable Ollama wins over the cloud
OLLAMA_MODEL=qwen2.5:7b-instruct
```

With `LLM_PROVIDER=none` the app runs on rules alone; everything the rules
cannot settle simply lands in `data/review`. Combined with local tesseract OCR
that is a fully offline pipeline.

## Workflow

```
Upload -> Analyze -> Classify -> Split -> Rename -> Review -> Export
```

1. **Analyze** - PyMuPDF reads the text layer; pages without one go to Azure
   Document Intelligence (`prebuilt-read`), or to a local tesseract fallback.
2. **Classify** - the two stages above: rules first, model only where needed.
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
├── run.command            macOS double-click launcher
├── data/                  incoming | processed (Clients/) | review | temp
├── modules/               ocr, classifier (rule engine), llm_providers
│                          (Ollama + Azure), pdf_splitter, renamer,
│                          folder_manager, export, validation, pipeline
├── prompts/               Azure OpenAI system prompt
└── logs/processing.log    every action, with confidence and file name
```

## Configuration

All settings live in `.env` (see the comments in that file). The most relevant:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIDENCE_THRESHOLD` | `0.90` | below this: `REVIEW REQUIRED`, no auto export |
| `LLM_PROVIDER` | `auto` | `auto` / `ollama` / `azure` / `none` - which model backs up the rules |
| `RULES_FIRST` | `true` | rules first, model only for what they cannot resolve |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | local model used as the fallback |
| `RULE_TRUSTED_CONFIDENCE` | `0.92` | confidence when every field came from a labelled anchor |
| `HEURISTIC_CONFIDENCE_CAP` | `0.55` | ceiling for keyword-only matches |
| `OCR_MIN_CHARS_PER_PAGE` | `60` | fewer characters on a page: treat it as scanned |
| `CLIENTS_ROOT` | `data/processed/Clients` | where the client folders live |
| `MAX_UPLOAD_MB` | `50` | upload size limit |

> `.env` ships with placeholder values only. Never commit real keys.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Python 3 was not found` | Install Python 3.10+; on Windows tick *Add python.exe to PATH* |
| Browser says the address is invalid on **Open in > Desktop app** | The desktop app isn't installed - that menu entry opens a `claude://` deep link. Install it, or use `git clone` + `./run.sh` instead |
| macOS: *"run.command can't be opened, unidentified developer"* | Quarantine flag from the ZIP download. System Settings > Privacy & Security > **Open Anyway**, or start `./run.sh` from Terminal, or use `git clone` instead of the ZIP |
| Port already in use | `PORT=9000 ./run.sh` |
| Browser does not open | Open <http://localhost:8501> manually |
| Everything says `REVIEW REQUIRED` | Expected without Azure credentials - fill in `.env` |
| Dependencies look broken | Delete `.venv` and start the launcher again |
