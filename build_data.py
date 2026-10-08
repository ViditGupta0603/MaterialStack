"""Step 1 — build the data tables.

Reads the downloaded source files in data/raw/ (downloads them when missing) and the hand-curated
tables in data/curated/, applies the cleaning rules R1–R6, and writes four CSV files:

  data/band_gaps.csv           one measured band gap per material (training labels + lookup)
  data/band_gaps_rejected.csv  every report that was not used, with the rule and the reason
  data/dft_gaps.csv            one DFT band gap per formula, GGA and hybrid (the model's "DFT hint")
  data/band_edges.csv          VBM (and CBM where measured) per material: measured and/or hybrid-DFT surfaces

Sources (all peer reviewed):
  measured gaps   Zhuo, Mansouri Tehrani & Brgoch, J. Phys. Chem. Lett. 9, 1668 (2018)
                  Borlido et al., J. Chem. Theory Comput. 15, 5069 (2019)       (preferred when present)
                  data/curated/measured_band_edges.csv                          (preferred when present)
  DFT gaps        JARVIS-DFT, Choudhary et al., npj Comput. Mater. 6, 173 (2020): OptB88vdW, TBmBJ, HSE06
                  SNUMAT, Kim et al., Sci. Data 7, 387 (2020): PBE, HSE06
  band edges      data/curated/measured_band_edges.csv                          (photoemission, cited per row)
                  Kiyohara, Hinuma & Oba, J. Am. Chem. Soc. 146 (2024), SI Data S1 (hybrid-DFT oxide surfaces)

Run:  python build_data.py
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

from materialstack.chem import RADIOACTIVE, elements, formula_key, to_float
from materialstack.config import (BAND_EDGES, BAND_GAPS, BAND_GAPS_REJECTED, DFT_GAPS, MEASURED_EDGES, RAW)

BORLIDO_XLSX = RAW / "borlido2019_bandgap_benchmark.xlsx"
BORLIDO_URL = "https://raw.githubusercontent.com/MaxGrossmann/mbpt_benchmark/main/spreadsheets/bandgap_benchmark.xlsx"
KIYOHARA_XLSX = RAW / "kiyohara2024_ja3c13574_si_002.xlsx"
KIYOHARA_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11009958/supplementaryFiles"
PREFERRED = ("Borlido 2019", "curated literature")      # curated values win over the large compilation

# Cleaning thresholds (rules R3–R6)
MAX_GAP = 15.0            # R3: no ordinary solid has a gap above ~14 eV (LiF)
SLIP_RATIO = 3.0          # R4: a report > 3x (and > 3 eV above) the other reports is a transcription error
AGREE_ABS, AGREE_REL = 0.3, 0.10   # R5: reports within max(0.3 eV, 10 %) of the median agree
MIN_AGREE = 0.5           # R5: at least half of the distinct reports must agree
MAX_DFT_CONFLICT = 2.0    # R6: |measured − hybrid DFT| above this, with one side metallic, means a wrong phase or a typo
METALLIC = 0.1            # R6: a gap below this (eV) counts as "metal"; only metal-vs-insulator conflicts are rejected
STABLE_WINDOW = 0.05      # eV/atom: JARVIS polymorphs used for the DFT hint (real metastable phases lie within ~50 meV)


# --------------------------------------------------------------------------- 1. read the sources

def zhuo_gaps() -> pd.DataFrame:
    from matminer.datasets import load_dataset
    df = load_dataset("expt_gap", data_home=str(RAW))
    return pd.DataFrame({"name": df["formula"], "gap_ev": df["gap expt"], "source": "Zhuo 2018",
                         "reference": "Zhuo et al., J. Phys. Chem. Lett. 9, 1668 (2018)"})


def borlido_gaps() -> pd.DataFrame:
    if not BORLIDO_XLSX.exists():
        BORLIDO_XLSX.write_bytes(urllib.request.urlopen(BORLIDO_URL, timeout=120).read())
    df = pd.read_excel(BORLIDO_XLSX)
    return pd.DataFrame({"name": df["Composition"], "gap_ev": df["Experimental"], "source": "Borlido 2019",
                         "reference": "Borlido et al., JCTC 15, 5069 (2019); measurement doi:" + df["DOI"].astype(str)})


def measured_edge_rows() -> pd.DataFrame:
    """The curated photoemission table with VBM/CBM worked out per row (IE → VBM = −IE, EA → CBM = −EA)."""
    df = pd.read_csv(MEASURED_EDGES)
    rows = []
    for d in df.to_dict(orient="records"):
        gap, ip, ea = to_float(d["gap"]), to_float(d["ip"]), to_float(d["ea"])
        vbm = to_float(d["vbm"]) if to_float(d["vbm"]) is not None else (-ip if ip is not None else None)
        cbm = to_float(d["cbm"]) if to_float(d["cbm"]) is not None else (-ea if ea is not None else None)
        if vbm is None and cbm is not None and gap is not None:      # only EA and gap reported
            vbm = cbm - gap
        if gap is None and vbm is not None and cbm is not None:      # IE and EA reported
            gap = cbm - vbm
        organic = str(d["formula"]).startswith("organic:")
        rows.append({"name": d["material"], "formula": d["formula"] if organic else formula_key(d["formula"]),
                     "organic": organic, "gap_ev": gap, "vbm_ev": vbm, "cbm_ev": cbm, "reference": d["reference"]})
    return pd.DataFrame(rows)


def curated_gaps(edges: pd.DataFrame) -> pd.DataFrame:
    g = edges[~edges.organic & edges.gap_ev.notna()]
    return pd.DataFrame({"name": g.formula, "gap_ev": g.gap_ev, "source": "curated literature",
                         "reference": g.reference})


def dft_gaps() -> pd.DataFrame:
    """One GGA and one hybrid-level gap per formula, from the stable polymorphs only.
    'hybrid' = HSE06 or TBmBJ (both close to experiment); 'gga' = OptB88vdW or PBE (underestimate ~40 %).

    JARVIS also holds many hypothetical polymorphs (39 for Si, most of them metallic), so a JARVIS entry is
    used only if it lies within STABLE_WINDOW of the lowest-energy JARVIS entry of the same formula.
    SNUMAT contains only experimentally known (ICSD) structures, so all of its entries are used."""
    from jarvis.db.figshare import data
    rows = []
    for d in data("dft_3d", store_dir=str(RAW)):
        base = {"name": d.get("formula"), "ehull": to_float(d.get("ehull"))}
        rows.append({**base, "gga": to_float(d.get("optb88vdw_bandgap")), "hybrid": to_float(d.get("mbj_bandgap"))})
        if to_float(d.get("hse_gap")) is not None:
            rows.append({**base, "gga": None, "hybrid": to_float(d.get("hse_gap"))})
    for d in data("snumat", store_dir=str(RAW)):
        counts = Counter(d["atoms"]["elements"])
        rows.append({"name": "".join(f"{el}{n}" for el, n in sorted(counts.items())), "ehull": None,
                     "gga": to_float(d.get("Band_gap_GGA")), "hybrid": to_float(d.get("Band_gap_HSE"))})
    df = pd.DataFrame(rows)
    keys = {n: formula_key(n) for n in df.name.dropna().unique()}
    df["formula"] = df.name.map(keys)
    df = df.dropna(subset=["formula"])
    e_rel = df.ehull - df.groupby("formula").ehull.transform("min")
    df = df[df.ehull.isna() | (e_rel <= STABLE_WINDOW)]
    out = df.groupby("formula").agg(
        gap_gga=("gga", "median"), gap_hybrid=("hybrid", "median"),
        n_gga=("gga", "count"), n_hybrid=("hybrid", "count")).reset_index()
    return out.round({"gap_gga": 3, "gap_hybrid": 3})


def kiyohara_surfaces() -> pd.DataFrame:
    if not KIYOHARA_XLSX.exists():
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(KIYOHARA_URL, timeout=1800).read())) as z:
            KIYOHARA_XLSX.write_bytes(z.read("ja3c13574_si_002.xlsx"))
    df = pd.read_excel(KIYOHARA_XLSX, header=1).iloc[:, 1:]
    df.columns = ["index", "system", "formula", "space_group", "miller", "surface_energy", "ip", "ea", "gap"]
    df["formula"] = df.formula.map(formula_key)
    return df.dropna(subset=["formula", "ip"])


# --------------------------------------------------------------------------- 2. clean the band gaps

def consensus(group: pd.DataFrame) -> tuple[float | None, pd.Series, str]:
    """Rule R5. Returns (label, mask of reports that disagree, basis) for one material's reports."""
    pref = group.source.isin(PREFERRED)
    if pref.any():
        return float(group.gap_ev[pref].median()), pd.Series(False, index=group.index), \
               " + ".join(sorted(group.source[pref].unique()))
    distinct = np.sort(group.gap_ev.unique())                 # copies of one number count once
    med = float(np.median(distinct))
    tol = max(AGREE_ABS, AGREE_REL * abs(med))
    if (np.abs(distinct - med) <= tol).mean() < MIN_AGREE:
        return None, pd.Series(True, index=group.index), ""
    agree = (group.gap_ev - med).abs() <= tol
    n = group.gap_ev[agree].nunique()
    return float(group.gap_ev[agree].median()), ~agree, f"consensus of {n} report{'s' * (n > 1)}"


def clean_gaps(reports: pd.DataFrame, dft: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports = reports.dropna(subset=["gap_ev"]).copy()
    reports["formula"] = reports.name.map(lambda n: formula_key(n) if isinstance(n, str) else None)
    rejected = []

    def reject(mask: pd.Series, rule: str, reason) -> None:
        r = reports[mask].copy()
        r["rule"], r["reason"] = rule, reason(r) if callable(reason) else reason
        rejected.append(r)

    def drop(mask: pd.Series) -> pd.DataFrame:
        return reports[~mask].copy()

    m = reports.formula.isna()
    reject(m, "R1", "formula could not be read")
    reports = drop(m)

    m = reports.formula.map(lambda f: bool(set(elements(f)) & RADIOACTIVE))
    reject(m, "R2", "contains a radioactive element (Tc, Pm, actinides): not device-relevant")
    reports = drop(m)

    m = (reports.gap_ev < 0) | (reports.gap_ev > MAX_GAP)
    reject(m, "R3", lambda r: [f"gap {g:.2f} eV outside 0–{MAX_GAP:g} eV" for g in r.gap_ev])
    reports = drop(m)

    def other_median(r: pd.Series) -> float:
        others = reports.gap_ev[(reports.formula == r.formula) & (reports.index != r.name)]
        return float(others.median()) if len(others) else np.nan
    multi = reports.formula.duplicated(keep=False)
    med_other = reports[multi].apply(other_median, axis=1).reindex(reports.index)
    m = (med_other > 0) & (reports.gap_ev > SLIP_RATIO * med_other) & (reports.gap_ev - med_other > 3)
    reject(m, "R4", lambda r: [f"{g:.2f} eV is > {SLIP_RATIO:g}x the other reports (median {o:.2f} eV): "
                               f"transcription error" for g, o in zip(r.gap_ev, med_other[m])])
    reports = drop(m)

    rows, disagree, no_majority = [], [], []
    for formula, g in reports.groupby("formula"):
        label, outliers, basis = consensus(g)
        if label is None:
            no_majority.append(g.index)
            continue
        disagree.append(g.index[outliers.to_numpy()])
        kept = g[~outliers.to_numpy()]
        rows.append({"formula": formula, "gap_ev": round(label, 3), "basis": basis, "n_reports": len(kept),
                     "values": "; ".join(f"{v:g}" for v in sorted(kept.gap_ev)),
                     "sources": "; ".join(sorted(kept.source.unique()))})
    idx = pd.Index(np.concatenate([i for i in no_majority] or [[]]).astype(int))
    reject(reports.index.isin(idx), "R5", "reports for this formula disagree (likely different phases)")
    outlier_idx = pd.Index(np.concatenate([i for i in disagree] or [[]]).astype(int))
    reject(reports.index.isin(outlier_idx), "R5", "disagrees with the other reports for this formula")
    gaps = pd.DataFrame(rows)

    # R6 only when one side says "metal": then a 0.00 entry for an insulator (or the reverse) is a typo or a
    # different phase. Two clear gaps that disagree are kept: DFT can be the one that is wrong (CsF: 10 vs 7.5 eV).
    hyb = gaps.formula.map(dft.set_index("formula").gap_hybrid)
    metal_vs_gap = (gaps.gap_ev < METALLIC) | (hyb < METALLIC)
    m = ~gaps.basis.str.contains("Borlido|curated") & metal_vs_gap & ((gaps.gap_ev - hyb).abs() > MAX_DFT_CONFLICT)
    conflict = gaps[m]
    for f, label, h in zip(conflict.formula, conflict.gap_ev, hyb[m]):
        mask = (reports.formula == f) & ~reports.index.isin(outlier_idx)
        reject(mask, "R6", f"measured {label:.2f} eV vs hybrid DFT {h:.2f} eV: likely a different phase or a typo")
    gaps = gaps[~m]

    cols = ["formula", "name", "source", "gap_ev", "rule", "reason", "reference"]
    rejected_df = pd.concat(rejected)[cols].sort_values(["rule", "formula"]) if rejected else pd.DataFrame(columns=cols)
    return gaps.sort_values("formula").reset_index(drop=True), rejected_df


# --------------------------------------------------------------------------- 3. band edges

def band_edges(measured: pd.DataFrame, surfaces: pd.DataFrame) -> pd.DataFrame:
    """One row per material and basis ("measured" or "hybrid-DFT surfaces"); the tool uses the measured row when
    there is one. Keeping both lets validate.py check the fallback against the measurement. Several measurements are combined as median VBM; the CBM comes from the median CBM − VBM of the rows that
    have both (left empty when only ionization energies were measured: the tool then uses CBM = VBM + Eg)."""
    rows = []
    for formula, g in measured.dropna(subset=["formula", "vbm_ev"]).groupby("formula"):
        vbm = float(g.vbm_ev.median())
        both = g.dropna(subset=["cbm_ev"])
        rows.append({"formula": formula, "name": g.name.iloc[0].split(" [")[0], "vbm_ev": round(vbm, 3),
                     "cbm_ev": round(vbm + float((both.cbm_ev - both.vbm_ev).median()), 3) if len(both) else None,
                     "basis": "measured", "n_values": len(g),
                     "vbm_spread_ev": round(float(g.vbm_ev.max() - g.vbm_ev.min()), 3),
                     "reference": " | ".join(dict.fromkeys(g.reference))})
    for formula, g in surfaces.groupby("formula"):
        vbm = -g.ip.astype(float)
        rows.append({"formula": formula, "name": formula, "vbm_ev": round(float(vbm.median()), 3), "cbm_ev": None,
                     "basis": "hybrid-DFT surfaces", "n_values": len(g),
                     "vbm_spread_ev": round(float(vbm.max() - vbm.min()), 3),
                     "reference": "Kiyohara, Hinuma & Oba, J. Am. Chem. Soc. 146 (2024), doi:10.1021/jacs.3c13574, "
                                  "SI Data S1 (median over surfaces)"})
    return pd.DataFrame(rows).sort_values(["basis", "formula"]).reset_index(drop=True)


# --------------------------------------------------------------------------- run

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    print("reading DFT gaps (JARVIS-DFT + SNUMAT) ...")
    dft = dft_gaps()
    measured = measured_edge_rows()
    reports = pd.concat([zhuo_gaps(), borlido_gaps(), curated_gaps(measured)], ignore_index=True)
    print(f"band-gap reports: {len(reports)}  " + str(reports.source.value_counts().to_dict()))
    gaps, rejected = clean_gaps(reports, dft)
    edges = band_edges(measured, kiyohara_surfaces())

    gaps.to_csv(BAND_GAPS, index=False)
    rejected.to_csv(BAND_GAPS_REJECTED, index=False)
    dft.to_csv(DFT_GAPS, index=False)
    edges.to_csv(BAND_EDGES, index=False)

    n_metal = int((gaps.gap_ev <= 0.001).sum())
    print(f"\n{BAND_GAPS.name}: {len(gaps)} materials ({len(gaps) - n_metal} semiconductors/insulators, {n_metal} metals)")
    print(f"{BAND_GAPS_REJECTED.name}: {len(rejected)} reports  " + str(rejected.rule.value_counts().sort_index().to_dict()))
    print(f"{DFT_GAPS.name}: {len(dft)} formulas ({dft.gap_gga.notna().sum()} GGA, {dft.gap_hybrid.notna().sum()} hybrid)")
    print(f"{BAND_EDGES.name}: {len(edges)} materials  " + str(edges.basis.value_counts().to_dict()))
