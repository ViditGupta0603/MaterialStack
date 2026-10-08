"""Pack the project into one zip to copy to another computer (no GitHub needed).

Included: code, curated data, the downloaded source files in data/raw/ (so the other computer needs no
downloads apart from Python packages), the built web UI (so it needs no Node.js), docs and notes.
Left out: things setup rebuilds (.venv, node_modules, model, results, built CSVs) and personal files.

Run:  python make_bundle.py        → MaterialStack_bundle.zip next to this folder
On the other computer: unzip, then run setup.bat (Windows) or bash setup.sh (macOS/Linux).
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "MaterialStack_bundle.zip"
SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", "results", ".claude"}
SKIP_BUILT = {"data/band_gaps.csv", "data/band_gaps_rejected.csv", "data/dft_gaps.csv", "data/band_edges.csv",
              "data/features.csv", "models/gap_model.joblib"}  # rebuilt by setup in seconds

if not (ROOT / "frontend" / "dist" / "index.html").exists():
    raise SystemExit("build the web UI first: cd frontend && npm run build")
if not any((ROOT / "data" / "raw").glob("*")):
    raise SystemExit("data/raw is empty: run python build_data.py first so the source files are included")

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if (p.is_dir() or set(rel.parts) & SKIP_DIRS or rel.as_posix() in SKIP_BUILT
                or p.name.startswith(".~lock")):                      # LibreOffice lock files
            continue
        z.write(p, Path("MaterialStack") / rel)
        n += 1
print(f"{OUT}: {n} files, {OUT.stat().st_size / 1e6:.0f} MB")
