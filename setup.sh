#!/usr/bin/env bash
# One-time setup on a new computer (macOS / Linux). From the MaterialStack folder run:  bash setup.sh
# Needs Python 3.12 or newer. Takes a few minutes (installing packages). The data, the trained model and the
# built web UI are in the repository, so nothing else is downloaded or built. To rebuild them yourself:
#   python build_data.py && python train.py && python validate.py      (and: cd frontend && npm ci && npm run build)
set -eo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

"$PY" -c 'import sys; assert sys.version_info >= (3, 12), f"Python 3.12 or newer is needed, found {sys.version.split()[0]}"'
echo "== 1/3 creating the virtual environment (.venv)"
"$PY" -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
echo "== 2/3 installing packages (requirements.txt)"
.venv/bin/python -m pip install --quiet -r requirements.txt
echo "== 3/3 checking with one prediction"
.venv/bin/python -m materialstack predict TiO2 MAPbI3 Spiro-OMeTAD

echo
echo "Ready. Start the web UI with:"
echo "  .venv/bin/python -m materialstack serve      (then open http://127.0.0.1:8000)"
