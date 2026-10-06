"""Step 3 — validate: how good is the model, and how good is the whole tool?

Three checks, each against reference data the model never trained on:
  A. Band gaps   grouped 5-fold cross-validation of the ML model. Materials made of the same elements stay in
                 the same fold, so the model is always tested on chemical systems it has never seen.
  B. Band edges  the curated measured VBMs vs what the tool predicts when the measurements are hidden
                 (hybrid-DFT surfaces or the Butler–Ginley estimate).
  C. Junctions   21 measured valence-band offsets (InterMat, Table 2) and the gold-standard device stacks in
                 verif/multilayer_gold_standard.csv, each with and without the curated measured band edges.

Writes results/metrics.csv (readable table), results/metrics.json (for the web UI), and two detail tables for
debugging: results/gap_cv_predictions.csv and results/junction_details.csv.

Run:  python validate.py
"""
from __future__ import annotations

import datetime
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from materialstack import model
from materialstack.config import BAND_EDGES, GOLD_STACKS, RESULTS
from materialstack.predict import predict_stack, resolve_layer

METRICS: list[dict] = []


def add(section, metric, value, unit="", n=None, baseline=None, baseline_rule="", note=""):
    METRICS.append({"section": section, "metric": metric, "value": None if value is None or pd.isna(value)
                    else round(float(value), 3), "unit": unit, "n": n,
                    "baseline": None if baseline is None or pd.isna(baseline) else round(float(baseline), 3),
                    "baseline_rule": baseline_rule, "note": note})


def mae(a, b) -> tuple[float, int]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return (float(np.abs(a[ok] - b[ok]).mean()) if ok.any() else np.nan), int(ok.sum())


# --------------------------------------------------------------------------- A. band gaps

def validate_gaps() -> None:
    X, y, table = model.training_set()
    groups = table.formula.map(model.element_group)
    pred, mean_pred, fold = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y), int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        pred[te] = model.predict(model.fit(X.iloc[tr], y[tr]), X.iloc[te])
        mean_pred[te], fold[te] = y[tr].mean(), k
    d = table.assign(predicted=pred, error=pred - y, fold=fold, dft_gga=X.dft_gap_gga, dft_hybrid=X.dft_gap_hybrid)
    d.round(3).to_csv(RESULTS / "gap_cv_predictions.csv", index=False)

    s = "A. Band gap (ML model, grouped 5-fold CV)"
    note = "semiconductors/insulators with a measured gap; error = |predicted − measured|"
    add(s, "MAE", np.abs(d.error).mean(), "eV", len(d), np.abs(mean_pred - y).mean(), "predict the mean gap", note)
    add(s, "RMSE", np.sqrt((d.error ** 2).mean()), "eV", len(d))
    add(s, "R²", 1 - (d.error ** 2).sum() / ((y - y.mean()) ** 2).sum(), "", len(d))
    add(s, "Spearman rank correlation", spearmanr(pred, y).correlation, "", len(d))
    add(s, "Within 0.3 eV", (np.abs(d.error) <= 0.3).mean() * 100, "%", len(d))
    add(s, "Within 0.5 eV", (np.abs(d.error) <= 0.5).mean() * 100, "%", len(d))
    hint = d.dft_hybrid.notna() | d.dft_gga.notna()
    add(s, "MAE, materials with a DFT hint", np.abs(d.error[hint]).mean(), "eV", int(hint.sum()))
    add(s, "MAE, formula only (no DFT data)", np.abs(d.error[~hint]).mean(), "eV", int((~hint).sum()),
        note="what to expect for a new composition")
    h = d.dft_hybrid.notna()
    add(s, "MAE where a hybrid DFT gap exists", np.abs(d.error[h]).mean(), "eV", int(h.sum()),
        *mae(d.dft_hybrid[h], d.gap_ev[h])[:1], "use the hybrid DFT gap (HSE06/TBmBJ) directly")
    g = d.dft_gga.notna()
    add(s, "MAE where a GGA DFT gap exists", np.abs(d.error[g]).mean(), "eV", int(g.sum()),
        *mae(d.dft_gga[g], d.gap_ev[g])[:1], "use the GGA DFT gap (PBE/OptB88vdW) directly")
    b = d.basis.str.contains("Borlido")
    add(s, "MAE on the Borlido benchmark materials", np.abs(d.error[b]).mean(), "eV", int(b.sum()),
        note="best-curated experimental labels")
    solar = (d.gap_ev >= 1.0) & (d.gap_ev <= 3.5)
    add(s, "MAE for gaps of 1–3.5 eV (solar-cell layers)", np.abs(d.error[solar]).mean(), "eV", int(solar.sum()))
    print(f"A. band gap CV MAE {np.abs(d.error).mean():.3f} eV on {len(d)} materials")


# --------------------------------------------------------------------------- B. band edges

def validate_edges() -> None:
    edges = pd.read_csv(BAND_EDGES)
    meas = edges[(edges.basis == "measured") & ~edges.formula.str.startswith("organic:")]
    rows = []
    for r in meas.itertuples():
        hidden = resolve_layer(r.formula, use_measured_edges=False)
        rows.append({"formula": r.formula, "measured_vbm": r.vbm_ev, "predicted_vbm": hidden.vbm_ev,
                     "fallback": hidden.edge_kind, "spread": r.vbm_spread_ev})
    d = pd.DataFrame(rows)
    s = "B. Band edges (measured VBM hidden)"
    note = "what the tool predicts for a material without photoemission data"
    for kind, label in (("surface", "hybrid-DFT surfaces"), ("estimate", "Butler–Ginley estimate")):
        k = d[d.fallback == kind]
        v, n = mae(k.predicted_vbm, k.measured_vbm)
        add(s, f"VBM MAE, {label}", v, "eV", n, note=note)
        add(s, f"VBM bias, {label}", (k.predicted_vbm - k.measured_vbm).mean() if n else None, "eV", n,
            note="negative = predicted too deep")
    multi = meas[meas.n_values > 1]
    add(s, "Spread between independent measurements of one material", multi.vbm_spread_ev.median(), "eV", len(multi),
        note="noise floor of the reference data itself (median)")
    print(f"B. band edges: {len(d)} measured materials checked")


# --------------------------------------------------------------------------- C. junctions

INTERMAT_TABLE2 = [  # (A, B, measured |ΔEv| in eV), Choudhary & Garrity, Digital Discovery 2024, Table 2
    ("AlP", "Si", 1.35), ("GaAs", "Si", 0.23), ("CdS", "Si", 1.60), ("AlAs", "GaAs", 0.55), ("CdS", "CdSe", 0.55),
    ("InP", "GaAs", 0.19), ("ZnTe", "AlSb", 0.35), ("CdSe", "ZnTe", 0.64), ("InAs", "AlAs", 0.50),
    ("InAs", "AlSb", 0.09), ("ZnSe", "InP", 0.41), ("InAs", "InP", 0.31), ("ZnSe", "AlAs", 0.40),
    ("GaAs", "ZnSe", 0.98), ("ZnS", "Si", 1.52), ("Si", "SiC", 0.50), ("GaN", "SiC", 0.70), ("Si", "AlN", 3.50),
    ("GaN", "AlN", 0.73), ("AlN", "InN", 1.81), ("GaN", "ZnO", 0.70),
]
SCOPES = {"tool": True, "tool without measured edges": False}


def validate_measured_offsets() -> None:
    s = "C1. Measured band offsets (21 interfaces, InterMat Table 2)"
    ex = np.array([e for *_, e in INTERMAT_TABLE2])
    for scope, use in SCOPES.items():
        pred = []
        for a, b, _ in INTERMAT_TABLE2:
            j = predict_stack([a, b], use)["junctions"][0]
            pred.append(abs(j["vbo_ev"]) if j["type"] else np.nan)
        v, n = mae(pred, ex)
        add(s, f"|ΔEv| MAE, {scope}", v, "eV", n, np.abs(ex).mean(), "predict ΔEv = 0",
            "with measured edges this is partly circular: several of these layers' electron affinities come from "
            "InterMat Table 1" if use else "Butler–Ginley / hybrid-DFT surfaces only")
    print("C1. measured offsets done")


def parse_gold() -> pd.DataFrame:
    """One row per junction of the gold stacks: names, gold type, gold offsets (next − previous layer)."""
    g = pd.read_csv(GOLD_STACKS)
    g = g[g["list of devices"].notna() & g["s.no"].notna()]
    rows = []
    for r in g.itertuples(index=False):
        names = [x.strip() for x in str(r[2]).split("|")]
        try:
            cbm, vbm = ([float(x) for x in str(c).split("|")] for c in (r[4], r[5]))
        except ValueError:
            continue
        if not len(names) == len(cbm) == len(vbm):
            continue
        types = {k.strip(): t for k, t in re.findall(r"([^;:]+):\s*Type\s+(I{1,3})", str(r[6]))}
        borderline = {k.strip() for k in re.findall(r"([^;:]+):\s*Type\s+I{1,3}\s*\(borderline", str(r[6]))}
        for i in range(len(names) - 1):
            key = f"{names[i]}/{names[i + 1]}"
            rows.append({"stack": int(r[0]), "top": names[i], "bottom": names[i + 1],
                         "gold_type": types.get(key), "borderline": key in borderline,
                         "gold_dEc": cbm[i + 1] - cbm[i], "gold_dEv": vbm[i + 1] - vbm[i]})
    return pd.DataFrame(rows)


def validate_gold_stacks() -> None:
    J = parse_gold()                                   # rows are in stack order, top to bottom
    for scope, use in SCOPES.items():
        tag = "tool" if use else "nomeas"
        res = []
        for _, g in J.groupby("stack", sort=False):
            res += predict_stack([g.top.iloc[0]] + g.bottom.tolist(), use)["junctions"]
        J[f"{tag}_type"] = [p["type"] for p in res]
        J[f"{tag}_dEc"] = [p.get("cbo_ev") for p in res]               # CBM(next) − CBM(previous)
        J[f"{tag}_dEv"] = [-p["vbo_ev"] if p.get("vbo_ev") is not None else None for p in res]
        J[f"{tag}_confidence"] = [p.get("confidence") for p in res]
        J[f"{tag}_offset_source"] = [p.get("offset_source") for p in res]
    J.round(3).to_csv(RESULTS / "junction_details.csv", index=False)

    s = "C2. Gold-standard device stacks"
    scored = J[J.gold_type.notna()]
    majority = scored.gold_type.mode().iloc[0]
    clear = (scored.gold_dEc.abs() >= 0.2) & (scored.gold_dEv.abs() >= 0.2)
    for scope, use in SCOPES.items():
        tag = "tool" if use else "nomeas"
        p = scored[scored[f"{tag}_type"].notna()]
        ok = p[f"{tag}_type"] == p.gold_type
        add(s, f"Coverage, {scope}", len(p) / len(scored) * 100, "%", len(scored), note="junctions with a prediction")
        add(s, f"Type accuracy, {scope}", ok.mean() * 100, "%", len(p),
            (p.gold_type == majority).mean() * 100, f"always predict Type {majority}")
        c = clear.reindex(p.index)
        add(s, f"Type accuracy on clear junctions (both offsets ≥ 0.2 eV), {scope}", ok[c].mean() * 100, "%",
            int(c.sum()), (p.gold_type[c] == majority).mean() * 100, f"always predict Type {majority}",
            "near-zero offsets make the type hinge on a few tenths of an eV")
        conf = p[f"{tag}_confidence"] >= 0.8
        add(s, f"Type accuracy when confident (≥ 80 %), {scope}", ok[conf].mean() * 100 if conf.any() else None,
            "%", int(conf.sum()), note="junctions not flagged 'check with DFT'")
        for q, label in (("dEc", "ΔEc"), ("dEv", "ΔEv")):
            v, n = mae(p[f"{tag}_{q}"], p[f"gold_{q}"])
            add(s, f"{label} MAE, {scope}", v, "eV", n, p[f"gold_{q}"].abs().mean(), f"predict {label} = 0")
            big = p[p[f"gold_{q}"].abs() >= 0.1]
            add(s, f"{label} sign agreement (|gold| ≥ 0.1 eV), {scope}",
                (np.sign(big[f"{tag}_{q}"].astype(float)) == np.sign(big[f"gold_{q}"])).mean() * 100, "%", len(big),
                note="spike vs cliff: decides whether carriers are extracted or blocked")
    print("C2. gold stacks done")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    validate_gaps()
    validate_edges()
    validate_measured_offsets()
    validate_gold_stacks()
    table = pd.DataFrame(METRICS)
    table.to_csv(RESULTS / "metrics.csv", index=False)
    bundle = model.load()
    (RESULTS / "metrics.json").write_text(json.dumps({
        "generated": datetime.date.today().isoformat(),
        "model": {"n_features": int(bundle.n_features_in_) if bundle is not None else None},
        "metrics": METRICS}, indent=1, ensure_ascii=False))
    print(f"\nwritten {RESULTS / 'metrics.csv'} ({len(table)} metrics)")
    print(table[["section", "metric", "value", "unit", "n", "baseline"]].to_string(index=False))
