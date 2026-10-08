"""Where every file lives. The whole project is a handful of CSV files, one model file and one results folder."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MATERIALSTACK_ROOT", Path(__file__).resolve().parent.parent))

DATA = ROOT / "data"
RAW = DATA / "raw"                                   # downloaded source files, never edited
CURATED = DATA / "curated"                           # hand-made, cited tables (the only data files you edit)
MEASURED_EDGES = CURATED / "measured_band_edges.csv"
MEASURED_OFFSETS = CURATED / "measured_band_offsets.csv"
ALIASES = CURATED / "aliases.csv"
VALIDATION_OFFSETS = CURATED / "validation_band_offsets.csv"   # measured offsets used only to score the tool

# Built by build_data.py
BAND_GAPS = DATA / "band_gaps.csv"                   # one measured gap per material (training labels + lookup)
BAND_GAPS_REJECTED = DATA / "band_gaps_rejected.csv"  # every rejected report, with the rule and reason
DFT_GAPS = DATA / "dft_gaps.csv"                     # one DFT gap per formula (the model's "hint")
BAND_EDGES = DATA / "band_edges.csv"                 # one VBM/CBM per material (measured or hybrid-DFT surfaces)

MODEL = ROOT / "models" / "gap_model.joblib"         # built by train.py
FEATURES = DATA / "features.csv"                     # written by train.py for inspection; never read back
RESULTS = ROOT / "results"                           # written by validate.py
GOLD_STACKS = DATA / "multilayer_gold_standard.csv"
