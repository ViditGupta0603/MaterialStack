"""Borlido et al. 2019 experimental band-gap benchmark (472 solids). Held-out test set.

Spreadsheet is the ACS Supporting Information (ct9b00322_si_002.xlsx), mirrored on Figshare.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from materialstack.chem import to_float
from materialstack.config import CACHE_DIR

REF = ("Borlido et al., J. Chem. Theory Comput. 15, 5069 (2019); "
       "472 experimental gaps with experimental structures; doi:10.1021/acs.jctc.9b00322")
XLSX = CACHE_DIR / "borlido_ct9b00322_si_002.xlsx"
URLS = [
    "https://github.com/MaxGrossmann/mbpt_benchmark/raw/master/spreadsheets/bandgap_benchmark.xlsx",
    "https://github.com/MaxGrossmann/mbpt_benchmark/raw/master/csvs/bandgap_benchmark.csv",
    "https://arxiv.org/src/2508.05247v1/anc/bandgap_benchmark.xlsx",
    "https://pubs.acs.org/doi/suppl/10.1021/acs.jctc.9b00322/suppl_file/ct9b00322_si_002.xlsx",
]


def _download() -> Path:
    if XLSX.exists() and XLSX.stat().st_size > 1000:
        return XLSX
    import urllib.request
    last_err = None
    req_headers = {"User-Agent": "materialstack/0.1 (research; +https://github.com/hackingmaterials)"}
    for url in URLS:
        dest = XLSX if url.endswith("xlsx") else XLSX.with_suffix(".csv")
        try:
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                dest.write_bytes(r.read())
            if dest.stat().st_size > 1000:
                return dest
        except Exception as e:
            last_err = e
            if dest.exists() and dest.stat().st_size < 1000:
                dest.unlink()
    raise RuntimeError(f"borlido_expt: could not download SI spreadsheet ({last_err})")


def _formula_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"compound", "formula", "system", "material", "name", "composition"}:
            return c
    return df.columns[0]


def _gap_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).strip().lower().replace("\n", " ")
        if cl in {"experimental", "experiment", "exp"}:
            return c
        if "exp" in cl and "gap" in cl:
            return c
        if cl in {"eg exp", "eg_exp", "experimental gap", "gap exp", "e_g^exp"}:
            return c
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {"gap", "eg", "e_g"}:
            return c
    raise RuntimeError(f"borlido_expt: no experimental-gap column in {list(df.columns)}")


def load(ctx: dict) -> Iterator[tuple[str, dict]]:
    path = _download()
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        xl = pd.ExcelFile(path)
        sheets = {n: pd.read_excel(path, sheet_name=n) for n in xl.sheet_names}
        df = max(sheets.values(), key=len)
    fcol, gcol = _formula_col(df), _gap_col(df)
    for i, d in enumerate(df.to_dict(orient="records")):
        formula = d.get(fcol)
        if not isinstance(formula, str) or not formula.strip():
            continue
        gap = to_float(d.get(gcol))
        if gap is None:
            continue
        extra = {str(k): (None if to_float(v) is None and not isinstance(v, str) else v)
                 for k, v in d.items() if k not in (fcol, gcol) and pd.notna(v)}
        ext = d.get("MP-ID") or d.get("ICSD-ID") or i
        yield "record", {
            "formula": formula.strip(), "method": "experiment", "gap": gap,
            "external_id": f"borlido-{ext}",
            "reference": REF, "extra": {"held_out_test": True, "icsd": d.get("ICSD-ID"), "mpid": d.get("MP-ID"),
                                        "doi": d.get("DOI"),
                                        **{k: extra[k] for k in ("HSE06", "MBJ", "PBE", "SCAN") if k in extra}},
        }
