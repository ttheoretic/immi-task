#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Immigration Document Processor - one click launcher (macOS / Linux)
#
#   ./run.sh                 start the app on http://localhost:8501
#   PORT=9000 ./run.sh       start it on a different port
#   ./run.sh --no-samples    skip creating the demo PDFs
#
# Creates a local virtual environment, installs the dependencies once,
# generates demo PDFs on first start and opens the app in the browser.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$0")"

APP_PORT="${PORT:-8501}"
VENV_DIR=".venv"
STAMP="${VENV_DIR}/.requirements.installed"
MAKE_SAMPLES=1
[ "${1:-}" = "--no-samples" ] && MAKE_SAMPLES=0

# 1) Find a usable Python 3.10+ interpreter ---------------------------------
PYTHON_BIN="${PYTHON:-}"
if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then PYTHON_BIN="${candidate}"; break; fi
  done
fi
if [ -z "${PYTHON_BIN}" ]; then
  echo "ERROR: Python 3 was not found. Install it from https://www.python.org/downloads/" >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "ERROR: Python 3.10 or newer is required (found: $(${PYTHON_BIN} -V 2>&1))." >&2
  exit 1
fi

# 2) Virtual environment ----------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment in ${VENV_DIR} ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"

# 3) Dependencies (re-installed only when requirements.txt changed) ---------
if [ ! -f "${STAMP}" ] || [ requirements.txt -nt "${STAMP}" ]; then
  echo "Installing dependencies (this takes a minute on the first run) ..."
  "${VENV_PYTHON}" -m pip install --upgrade pip --quiet
  "${VENV_PYTHON}" -m pip install -r requirements.txt --quiet
  touch "${STAMP}"
fi

# 4) Demo PDFs so the app can be tested without real client documents -------
if [ "${MAKE_SAMPLES}" = "1" ] && [ ! -d "data/incoming/samples" ]; then
  "${VENV_PYTHON}" create_test_pdfs.py || echo "WARNING: demo PDFs could not be created" >&2
fi

# 5) Start ------------------------------------------------------------------
# Streamlit runs headless (that also skips its first-run e-mail prompt) and the
# browser is opened by this script as soon as the port accepts connections.
echo
echo "Starting Immigration Document Processor on http://localhost:${APP_PORT}"
echo "Press Ctrl+C to stop."
echo

"${VENV_PYTHON}" -m streamlit run app.py \
  --server.port "${APP_PORT}" \
  --server.headless true \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!
trap 'kill "${STREAMLIT_PID}" 2>/dev/null || true' INT TERM

for _ in $(seq 1 60); do
  if ! kill -0 "${STREAMLIT_PID}" 2>/dev/null; then
    break  # Streamlit stopped on its own - its output explains why
  fi
  if "${VENV_PYTHON}" -c "import socket,sys; socket.create_connection(('127.0.0.1', ${APP_PORT}), 0.5)" 2>/dev/null; then
    # NO_BROWSER=1 is set by the Claude Code desktop preview, which opens the
    # page in its own Browser pane.
    if [ -z "${NO_BROWSER:-}" ]; then
      "${VENV_PYTHON}" -c "import webbrowser; webbrowser.open('http://localhost:${APP_PORT}')" 2>/dev/null || true
    fi
    break
  fi
  sleep 0.5
done

wait "${STREAMLIT_PID}"
