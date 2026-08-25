#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Immigration Document Processor - double-click launcher for macOS.
#
# Finder runs *.command files in Terminal, so this file is the double-click
# entry point; everything else lives in run.sh.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")" || exit 1

./run.sh
STATUS=$?

if [ "${STATUS}" -ne 0 ]; then
  echo
  echo "The app stopped with exit code ${STATUS}."
  read -r -p "Press Enter to close this window ... " _
fi
exit "${STATUS}"
