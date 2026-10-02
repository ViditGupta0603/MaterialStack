from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MATERIALSTACK_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
LITERATURE_DIR = DATA_DIR / "literature"
MODELS_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "materials_db.sqlite"

for _d in (DATA_DIR, CACHE_DIR, LITERATURE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Lower rank = more trustworthy when several records exist for the same material.
METHOD_RANK: dict[str, int] = {
    "experiment": 1,
    "literature": 1,
    "HSE06": 2,
    "TBmBJ": 3,
    "DFT-surface-TBmBJ": 3,
    "GLLB-SC": 4,
    "r2SCAN": 5,
    "OptB88vdW": 5,
    "DFT-surface-OptB88vdW": 5,
    "PBE": 6,
    "device": 7,
    "ml_prediction": 9,
}

# Sources that need several GB of RAM (structures for >80k materials). Opt-in via --heavy.
HEAVY_SOURCES = {"matbench_mp_gap", "mp_all_20181018"}

DEFAULT_SOURCES = [
    # experimental gaps
    "expt_gap",
    "expt_gap_kingsbury",
    "foundry_ml_exp_bandgaps",
    # hybrid / meta-GGA gaps with structures
    "snumat",
    "jarvis_dft_3d",
    "jarvis_dft_2d",
    "jarvis_halide_perovskites",
    # vacuum-referenced band edges
    "jarvis_surfacedb",
    "jarvis_interfacedb",
    "castelli_perovskites",
    "double_perovskites_gap",
    "borlido_expt",
    # PBE gaps
    "mp_nostruct_20181018",
    "wolverton_oxides",
    "dielectric_constant",
    # user-supplied
    "literature_csv",
    "aliases",
]

ALL_SOURCES = DEFAULT_SOURCES + sorted(HEAVY_SOURCES) + ["materials_project"]
