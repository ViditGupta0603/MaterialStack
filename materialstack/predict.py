"""Prediction: from layer names to band edges and junction types.

For each layer
  1. name → formula     data/curated/aliases.csv (MAPbI3 → CH3NH3PbI3, Spiro-OMeTAD → organic), else the name itself
  2. band gap Eg        measured (data/band_gaps.csv)  →  else the ML model (model.py)
  3. VBM                measured (data/band_edges.csv)
                        →  hybrid-DFT surfaces: VBM = 0.8 × surface VBM + 0.2 × Butler–Ginley
                        →  Butler–Ginley estimate: VBM = −χ − Eg/2
     CBM                measured if given, else VBM + Eg
For each pair of neighbouring layers (top = earlier in the list)
  4. valence-band offset  measured interface offset (data/curated/measured_band_offsets.csv)
                          →  else vacuum alignment (Anderson's rule): VBM(top) − VBM(bottom)
  5. type I / II / III    from the four band edges: a physics definition, not learned
  6. confidence           Monte Carlo: vary every input by its typical error (SIGMA below), count the types

Organic layers (Spiro-OMeTAD, PCBM, C60 …) have no inorganic formula: they use measured values only.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Element

from materialstack import model
from materialstack.chem import butler_ginley_vbm, elements, formula_key, inorganic_surrogate, mulliken_chi, to_float
from materialstack.config import ALIASES, BAND_EDGES, BAND_GAPS, MEASURED_OFFSETS

SURFACE_WEIGHT = 0.8        # weight of the hybrid-DFT surface VBM against Butler–Ginley (chosen on benchmarks)
METAL_GAP = 0.001           # eV; a layer with a smaller gap is a metal (contact), not part of a junction

# Typical error (σ, eV) of each kind of input, from validate.py (results/metrics.csv); σ ≈ MAE / 0.8.
SIGMA_GAP = {"measured": 0.2,            # scatter between independent compilations of the same material
             "ML": 0.5}                  # cross-validated MAE of the model 0.40 eV
SIGMA_VBM = {"measured": 0.25,           # photoemission values of one material differ by ~0.2–0.3 eV
             "surface": 0.65,            # hybrid-DFT surfaces vs measured VBM: MAE 0.52 eV (5 materials)
             "estimate": 1.3}            # Butler–Ginley vs measured VBM: RMS 0.82 eV (39 materials), but CuI, CuSCN
#                                          and NiO are 1.9–2.5 eV too deep; on the device stacks σ 0.6–1.0 made
#                                          "≥ 80 %" calls right only 64–67 % of the time, 1.3 keeps them at 75 %
SIGMA_MEASURED_OFFSET = 0.2              # photoemission interface offsets are quoted ±0.1–0.2 eV
CONFIDENT = 0.8                          # below this probability a junction is flagged "check with DFT"
N_SAMPLES = 4000


@dataclass
class Layer:
    query: str                       # what the user typed
    formula: str | None              # key used in the data files (reduced formula or organic:<name>)
    display_formula: str | None      # formula as a chemist writes it (alias target)
    gap_ev: float | None = None
    gap_kind: str = "none"           # measured | ML | metal | none
    gap_source: str = ""
    vbm_ev: float | None = None
    cbm_ev: float | None = None
    edge_kind: str = "none"          # measured | surface | estimate | none
    edge_source: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- data, loaded once

@lru_cache(maxsize=1)
def _tables() -> dict[str, Any]:
    aliases = pd.read_csv(ALIASES)
    edges = pd.read_csv(BAND_EDGES)
    offsets = pd.read_csv(MEASURED_OFFSETS)
    offsets["film"] = offsets.formula_film.map(formula_key)
    offsets["substrate"] = offsets.formula_substrate.map(formula_key)
    return {
        "aliases": dict(zip(aliases.alias.str.lower(), aliases.formula)),
        "gaps": pd.read_csv(BAND_GAPS).set_index("formula"),
        "measured_edges": edges[edges.basis == "measured"].set_index("formula"),
        "surface_edges": edges[edges.basis == "hybrid-DFT surfaces"].set_index("formula"),
        "offsets": offsets,
        "by_lowercase": _lowercase_index(edges.formula),
    }


def _lowercase_index(edge_formulas: pd.Series) -> dict[str, str]:
    """'tio2' → 'TiO2' for every formula the tool has data for, when the lowercase spelling is unambiguous.
    Formulas with measured data come first, so 'sio2' is SiO2 and not the DFT-only SIO2 (sulfur iodine oxide)."""
    index: dict[str, str] = {}
    for known in (set(pd.read_csv(BAND_GAPS).formula) | set(edge_formulas), set(model.dft_table().index)):
        seen: dict[str, set[str]] = {}
        for f in known:
            seen.setdefault(str(f).lower(), set()).add(str(f))
        for k, v in seen.items():
            if k not in index and len(v) == 1:
                index[k] = next(iter(v))
    return index


# --------------------------------------------------------------------------- one layer

def resolve_layer(name: str, use_measured_edges: bool = True) -> Layer:
    """Band gap and band edges of one layer. ``use_measured_edges=False`` ignores the curated photoemission
    data (used by validate.py to test what the tool predicts for a material without measurements)."""
    t = _tables()
    target = t["aliases"].get(name.strip().lower(), name.strip())
    if target.startswith("organic:"):
        return _organic_layer(name, target, use_measured_edges)
    formula = formula_key(target)
    if formula is None and target.lower() in t["by_lowercase"]:      # 'tio2': formulas are case-sensitive
        formula = target = t["by_lowercase"][target.lower()]
    layer = Layer(query=name, formula=formula, display_formula=target)
    if formula is None:
        layer.notes.append("unknown name: not a chemical formula or a known layer name. Check the spelling; "
                           "formulas are case-sensitive (TiO2, not tio2)")
        return layer
    if target != name.strip() and name.strip().lower() == target.lower():
        layer.notes.append(f"read as {target}")

    # Band gap: measured, else metal (only metallic elements, e.g. Au, Ag, Al contacts), else ML
    if formula in t["gaps"].index:
        row = t["gaps"].loc[formula]
        layer.gap_ev, layer.gap_kind = float(row.gap_ev), "measured"
        layer.gap_source = f"measured ({row.basis})"
    elif all(Element(e).is_metal for e in elements(formula)):
        layer.gap_ev, layer.gap_kind, layer.gap_source = 0.0, "metal", "metal (only metallic elements)"
    else:
        m = model.load()
        if m is None:
            layer.notes.append("no trained model: run python train.py")
            return layer
        X = model.featurize([formula])
        layer.gap_ev, layer.gap_kind = float(model.predict(m, X)[0]), "ML"
        hint = X.dft_gap_hybrid.iloc[0] if pd.notna(X.dft_gap_hybrid.iloc[0]) else X.dft_gap_gga.iloc[0]
        layer.gap_source = ("ML model, using a DFT gap of {:.2f} eV as a hint".format(hint) if pd.notna(hint)
                            else "ML model, from the formula only (no DFT data for this formula)")
        if inorganic_surrogate(formula) != formula:
            layer.gap_source += f"; organic cation counted as Cs ({inorganic_surrogate(formula)})"
    if layer.gap_ev <= METAL_GAP:
        layer.notes.append("metal: no band gap, so it does not form a semiconductor junction")

    # Band edges: measured → hybrid-DFT surfaces → Butler–Ginley
    chi = mulliken_chi(inorganic_surrogate(formula))      # MAPbI3 → χ of CsPbI3: the organic cation sets no band edge
    if use_measured_edges and formula in t["measured_edges"].index:
        row = t["measured_edges"].loc[formula]
        layer.vbm_ev = float(row.vbm_ev)
        layer.cbm_ev = float(row.cbm_ev) if pd.notna(row.cbm_ev) else layer.vbm_ev + layer.gap_ev
        layer.edge_kind, layer.edge_source = "measured", f"measured (photoemission): {row.reference}"
        if row.vbm_spread_ev > 0.3:
            layer.notes.append(f"measured VBMs differ by {row.vbm_spread_ev:.1f} eV (surface/preparation dependent); "
                               f"median of {row.n_values} used")
        return layer
    if chi is None:
        layer.notes.append("cannot place band edges: no electronegativity for an element")
        return layer
    bg = butler_ginley_vbm(chi, layer.gap_ev)
    if formula in t["surface_edges"].index:
        row = t["surface_edges"].loc[formula]
        layer.vbm_ev = SURFACE_WEIGHT * float(row.vbm_ev) + (1 - SURFACE_WEIGHT) * bg
        layer.edge_kind = "surface"
        layer.edge_source = (f"hybrid-DFT surfaces ({row.n_values}, Kiyohara 2024): "
                             f"{SURFACE_WEIGHT:g} × surface VBM + {1 - SURFACE_WEIGHT:.1f} × Butler–Ginley")
    else:
        layer.vbm_ev, layer.edge_kind = bg, "estimate"
        layer.edge_source = f"Butler–Ginley estimate: VBM = −χ − Eg/2 with χ = {chi:.2f} eV"
    layer.cbm_ev = layer.vbm_ev + layer.gap_ev
    return layer


def _organic_layer(name: str, target: str, use_measured_edges: bool) -> Layer:
    """Molecules and polymers: no ML and no Butler–Ginley (both need an inorganic formula); measured only."""
    layer = Layer(query=name, formula=target, display_formula=target.split(":", 1)[1])
    edges = _tables()["measured_edges"]
    if not use_measured_edges or target not in edges.index or pd.isna(edges.loc[target].cbm_ev):
        layer.notes.append("organic layer with no measured ionization energy and electron affinity in the data, "
                           "so its levels cannot be placed (the model only handles inorganic formulas)")
        return layer
    row = edges.loc[target]
    layer.vbm_ev, layer.cbm_ev = float(row.vbm_ev), float(row.cbm_ev)
    layer.gap_ev, layer.gap_kind = layer.cbm_ev - layer.vbm_ev, "measured"
    layer.gap_source = "measured transport gap (EA − IE)"
    layer.edge_kind, layer.edge_source = "measured", f"measured (photoemission): {row.reference}"
    return layer


# --------------------------------------------------------------------------- one junction

def junction_type(top_vbm: float, top_cbm: float, bottom_vbm: float, bottom_cbm: float) -> str:
    """Type I (straddling: one gap inside the other), III (broken gap: the bands overlap) or II (staggered)."""
    if top_vbm > bottom_cbm or bottom_vbm > top_cbm:
        return "III"
    if (top_cbm >= bottom_cbm and top_vbm <= bottom_vbm) or (bottom_cbm >= top_cbm and bottom_vbm <= top_vbm):
        return "I"
    return "II"


def type_probabilities(gap_top: float, gap_bottom: float, vbo: float, s_gap_top: float, s_gap_bottom: float,
                       s_vbo: float, n: int = N_SAMPLES) -> dict[str, float]:
    """P(type I/II/III): sample both gaps and the offset from normal distributions, apply junction_type to each."""
    rng = np.random.default_rng(0)
    gt = np.clip(rng.normal(gap_top, s_gap_top, n), 0.0, None)
    gb = np.clip(rng.normal(gap_bottom, s_gap_bottom, n), 0.0, None)
    v = rng.normal(vbo, s_vbo, n)
    t_v, t_c, b_v, b_c = 0.0, gt, -v, -v + gb               # energies relative to the top layer's VBM
    broken = (t_v > b_c) | (b_v > t_c)
    straddling = ((t_c >= b_c) & (t_v <= b_v)) | ((b_c >= t_c) & (b_v <= t_v))
    p3, p1 = broken.mean(), (straddling & ~broken).mean()
    return {"I": float(p1), "II": float(1 - p1 - p3), "III": float(p3)}


def measured_offset(top: Layer, bottom: Layer) -> dict[str, Any] | None:
    """Measured interface offset for this pair (either order), as vbo = VBM(top) − VBM(bottom) and
    cbo = CBM(bottom) − CBM(top). The file stores VBM(substrate) − VBM(film) and CBM(substrate) − CBM(film)."""
    off = _tables()["offsets"]
    for film, sign in ((top.formula, 1.0), (bottom.formula, -1.0)):
        other = bottom.formula if sign > 0 else top.formula
        hit = off[(off.film == film) & (off.substrate == other)]
        if len(hit):
            r = hit.iloc[0]
            cbo = to_float(r.cbo_substrate_minus_film)
            return {"vbo": -sign * float(r.vbo_substrate_minus_film), "cbo": sign * cbo if cbo is not None else None,
                    "reference": r.reference}
    return None


def classify_junction(top: Layer, bottom: Layer, use_measured_edges: bool = True) -> dict[str, Any]:
    """Offsets, type and confidence for ``top`` on ``bottom``.
    vbo = VBM(top) − VBM(bottom) > 0: holes collect in the top layer;
    cbo = CBM(bottom) − CBM(top) > 0: electrons collect in the top layer."""
    out = {"interface": f"{top.query} | {bottom.query}", "type": None, "uncertain": True}
    if None in (top.gap_ev, bottom.gap_ev, top.vbm_ev, bottom.vbm_ev):
        return {**out, "reason": "a layer has no band gap or band edges"}
    if min(top.gap_ev, bottom.gap_ev) <= METAL_GAP:
        return {**out, "reason": "metal contact: no semiconductor junction type"}

    gap_t, gap_b = top.gap_ev, bottom.gap_ev
    off = measured_offset(top, bottom) if use_measured_edges else None
    if off is not None:
        vbo, s_vbo = off["vbo"], SIGMA_MEASURED_OFFSET
        source = f"measured interface offset ({off['reference']})"
        if off["cbo"] is not None:                # measured at the real interface: use it as is
            gap_b = gap_t + off["cbo"] + vbo
    else:
        vbo = top.vbm_ev - bottom.vbm_ev
        s_vbo = math.hypot(SIGMA_VBM[top.edge_kind], SIGMA_VBM[bottom.edge_kind])
        source = "vacuum alignment of the two layers (Anderson's rule)"
    cbo = -vbo + gap_b - gap_t
    s_cbo = math.sqrt(s_vbo ** 2 + SIGMA_GAP[top.gap_kind] ** 2 + SIGMA_GAP[bottom.gap_kind] ** 2)
    jtype = junction_type(0.0, gap_t, -vbo, -vbo + gap_b)
    probs = type_probabilities(gap_t, gap_b, vbo, SIGMA_GAP[top.gap_kind], SIGMA_GAP[bottom.gap_kind], s_vbo)
    conf = probs[jtype]
    return {**out, "type": jtype, "vbo_ev": vbo, "cbo_ev": cbo, "vbo_sigma_ev": s_vbo, "cbo_sigma_ev": s_cbo,
            "offset_source": source, "type_probabilities": probs,
            "confidence": conf, "uncertain": conf < CONFIDENT,
            "reason": "check with DFT" if conf < CONFIDENT else "confident"}


def predict_stack(names: list[str], use_measured_edges: bool = True) -> dict[str, Any]:
    """All layers of a device (top first) and the junction between each neighbouring pair."""
    layers = [resolve_layer(n, use_measured_edges) for n in names]
    junctions = [classify_junction(a, b, use_measured_edges) for a, b in zip(layers, layers[1:])]
    return {"layers": [L.to_dict() for L in layers], "junctions": junctions}
