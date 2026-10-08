#!/usr/bin/env bash
# One-time setup on a new computer (macOS / Linux). From the MaterialStack folder run:  bash setup.sh
# Needs Python 3.12 or newer. Takes ~5 minutes (mostly installing packages).
set -eo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

"$PY" -c 'import sys; assert sys.version_info >= (3, 12), f"Python 3.12 or newer is needed, found {sys.version.split()[0]}"'
echo "== 1/5 creating the virtual environment (.venv)"
"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
echo "== 2/5 installing packages (requirements.txt)"
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "== 3/5 building the data tables"
.venv/bin/python build_data.py
echo "== 4/5 training the model"
.venv/bin/python train.py
echo "== 5/5 validating"
.venv/bin/python validate.py | grep -E '^(A|B|C1|C2)\.'

if [ ! -f frontend/dist/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    echo "== building the web UI"
    (cd frontend && npm ci && npm run build)
  else
    echo "note: the web UI is not built and Node.js is not installed. The command line works;"
    echo "      install Node.js 20+ and run 'cd frontend && npm ci && npm run build' for the web UI."
  fi
fi

echo
echo "Done. Try:"
echo "  .venv/bin/python -m materialstack predict TiO2 MAPbI3 Spiro-OMeTAD"
echo "  .venv/bin/python -m materialstack serve      (then open http://127.0.0.1:8000)"
