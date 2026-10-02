"""Live Materials Project pull (requires an API key). Results are cached to parquet so rebuilds are free."""
from __future__ import annotations

import os
from typing import Iterator

import pandas as pd

from materialstack.chem import to_float
from materialstack.config import CACHE_DIR

REF_MP = "Materials Project (Jain et al., APL Mater. 1, 011002 (2013)); live API"
FIELDS = ["material_id", "formula_pretty", "band_gap", "cbm", "vbm", "is_gap_direct", "is_metal",
          "energy_above_hull", "is_stable", "theoretical", "symmetry", "density", "volume", "nsites"]
CACHE = CACHE_DIR / "mp_summary.parquet"


def _fetch(api_key: str) -> pd.DataFrame:
    from mp_api.client import MPRester
    rows = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(fields=FIELDS, chunk_size=1000)
        for d in docs:
            sym = getattr(d, "symmetry", None)
            rows.append({
                "material_id": str(d.material_id), "formula": d.formula_pretty, "band_gap": d.band_gap,
                "cbm": d.cbm, "vbm": d.vbm, "is_gap_direct": d.is_gap_direct, "is_metal": d.is_metal,
                "energy_above_hull": d.energy_above_hull, "is_stable": d.is_stable, "theoretical": d.theoretical,
                "spg": getattr(sym, "number", None) if sym else None,
                "crystal_system": str(getattr(sym, "crystal_system", "")) if sym else None,
                "density": d.density, "volume": d.volume, "nsites": d.nsites,
            })
    df = pd.DataFrame(rows)
    df.to_parquet(CACHE, index=False)
    return df


def load(ctx: dict) -> Iterator[tuple[str, dict]]:
    api_key = ctx.get("mp_api_key") or os.environ.get("MP_API_KEY")
    if CACHE.exists() and not ctx.get("refresh_mp"):
        df = pd.read_parquet(CACHE)
    elif api_key:
        df = _fetch(api_key)
    else:
        raise RuntimeError("materials_project: no MP_API_KEY provided and no cached pull found; skipped")
    for d in df.to_dict(orient="records"):
        gap = to_float(d.get("band_gap"))
        yield "record", {
            "formula": d["formula"], "method": "PBE", "gap": gap, "vbm": to_float(d.get("vbm")),
            "cbm": to_float(d.get("cbm")), "edge_reference": "internal",
            "is_direct": d.get("is_gap_direct"), "is_metal": d.get("is_metal"),
            "e_above_hull": to_float(d.get("energy_above_hull")), "spg": d.get("spg"),
            "external_id": d["material_id"], "reference": REF_MP,
            "extra": {"is_stable": d.get("is_stable"), "theoretical": d.get("theoretical"),
                      "density": to_float(d.get("density")), "nsites": d.get("nsites"),
                      "crystal_system": d.get("crystal_system"),
                      "note": "VBM/CBM referenced to cell-averaged potential; not comparable across materials"},
        }
