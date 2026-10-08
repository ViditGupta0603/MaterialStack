"""Step 3 — validate: how good is the model, and how good is the whole tool?

A fixed protocol over living data: this file is not edited when the data grows. Every test set is defined by a
rule over the current data files, so new curated data joins validation automatically; to add reference
offsets, add rows to data/curated/validation_band_offsets.csv. Compare two versions with
`git diff results/metrics.csv`.

  A.  Band gaps     grouped 5-fold cross-validation of the ML model. The fold of a material is fixed by its
                    element system (hash), so materials made of the same elements are always tested together
                    (unseen chemistry), and adding data never moves an existing material to another fold.
  A2.               the same errors split by label source, chemistry and gap size; the label-noise floor.
  B.  Band edges    every measured VBM vs what the tool predicts when the measurement is hidden.
  C1. Offsets       measured valence-band offsets (validation_band_offsets.csv), with and without measured edges.
  C2. Devices       the published device stacks in data/multilayer_gold_standard.csv (SCAPS band edges).
  D.  Calibration   are the σ values in predict.py right, and does "85 %" mean right 85 % of the time?
  E.  Coverage      how much data stands behind the numbers above.

Every mean-type metric has a 95 % bootstrap interval (ci_low, ci_high), resampling element systems, materials,
interfaces or device stacks. Rows marked primary are the ones to decide changes on; the rest are diagnostics.

Writes results/metrics.csv (readable table), results/metrics.json (for the web UI), and two detail tables:
results/gap_cv_predictions.csv and results/junction_details.csv.

Run:  python validate.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from materialstack import model
from materialstack.chem import elements
from materialstack.config import (BAND_EDGES, BAND_GAPS, GOLD_STACKS, MEASURED_OFFSETS, RESULTS,
                                  VALIDATION_OFFSETS)
from materialstack.predict import CONFIDENT, SIGMA_GAP, SIGMA_VBM, predict_stack, resolve_layer

# The protocol. Changing any of these changes every number, so they are fixed.
N_FOLDS = 5
N_BOOT = 1000            # bootstrap resamples for the 95 % intervals
SIGN_MIN = 0.1           # eV: a reference offset smaller than this has no meaningful sign
CLEAR_MIN = 0.2          # eV: a junction is "clear" when both reference offsets are at least this large
SMALL_N = 20             # below this many items a metric is only indicative
CONF_BINS = [0.0, 0.6, 0.7, 0.8, 0.9, 1.01]   # confidence bins for the reliability table
SCOPES = {"tool": True, "tool without measured edges": False}

METRICS: list[dict] = []


def fold_of(element_system: str) -> int:
    """Fixed fold of an element system ('Ga-As' → 0..4): a hash, so it never depends on the other data."""
    return int(hashlib.md5(element_system.encode()).hexdigest(), 16) % N_FOLDS


def mean_with_ci(values, groups, scale=1.0, root=False) -> tuple[float, tuple, int]:
    """Mean of the finite values with a 95 % bootstrap interval that resamples whole groups (exact weighted form)."""
    v, g = np.asarray(values, float), np.asarray(groups)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    if not len(v):
        return np.nan, (None, None), 0
    _, inv = np.unique(g, return_inverse=True)
    sums, counts = np.bincount(inv, weights=v), np.bincount(inv).astype(float)
    k = len(sums)
    w = np.random.default_rng(0).multinomial(k, np.full(k, 1 / k), size=N_BOOT)
    boot = (w @ sums) / (w @ counts)
    f = (lambda x: np.sqrt(x) * scale) if root else (lambda x: x * scale)
    return f(v.mean()), (f(np.percentile(boot, 2.5)), f(np.percentile(boot, 97.5))), len(v)


def add(section, metric, value, unit="", n=None, baseline=None, baseline_rule="", note="", ci=(None, None),
        primary=False):
    def num(x):
        return None if x is None or pd.isna(x) else round(float(x), 3)
    if n is not None and n < SMALL_N:
        note = "; ".join(x for x in (note, "small n, indicative") if x)
    METRICS.append({"section": section, "metric": metric, "value": num(value), "unit": unit, "n": n,
                    "ci_low": num(ci[0]), "ci_high": num(ci[1]), "baseline": num(baseline),
                    "baseline_rule": baseline_rule, "primary": primary, "note": note})


def add_mean(section, metric, values, groups, unit="eV", scale=1.0, root=False, **kw) -> None:
    """A metric that is a mean over items (MAE, bias, share in %, RMSE with root=True), with its interval."""
    value, ci, n = mean_with_ci(values, groups, scale, root)
    add(section, metric, value, unit, n, ci=ci, **kw)


def chemistry(formula: str) -> str:
    els = set(elements(formula))
    for name, anions in (("oxide", {"O"}), ("chalcogenide", {"S", "Se", "Te"}), ("halide", {"F", "Cl", "Br", "I"}),
                         ("pnictide", {"N", "P", "As", "Sb"})):
        if els & anions:
            return name
    return "other"


# --------------------------------------------------------------------------- A. band gaps

def validate_gaps() -> pd.DataFrame:
    X, y, table = model.training_set()
    groups = table.formula.map(model.element_group)
    fold = groups.map(fold_of).to_numpy()
    pred, mean_pred = np.zeros(len(y)), np.zeros(len(y))
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        pred[te] = model.predict(model.fit(X[tr], y[tr]), X[te])
        mean_pred[te] = y[tr].mean()
    d = table.assign(predicted=pred, error=pred - y, fold=fold, group=groups,
                     dft_gga=X.dft_gap_gga, dft_hybrid=X.dft_gap_hybrid)
    d.round(3).to_csv(RESULTS / "gap_cv_predictions.csv", index=False)
    err, absr, grp = d.error, d.error.abs(), d.group

    s = "A. Band gap (ML model, grouped 5-fold CV)"
    note = "semiconductors/insulators with a measured gap; error = |predicted − measured|"
    add_mean(s, "MAE", absr, grp, baseline=np.abs(mean_pred - y).mean(), baseline_rule="predict the mean gap",
             note=note, primary=True)
    add_mean(s, "RMSE", err ** 2, grp, root=True)
    add(s, "R²", 1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum(), "", len(d))
    add(s, "Spearman rank correlation", spearmanr(pred, y).correlation, "", len(d))
    add_mean(s, "Within 0.3 eV", absr <= 0.3, grp, "%", 100)
    add_mean(s, "Within 0.5 eV", absr <= 0.5, grp, "%", 100)
    hint = d.dft_hybrid.notna() | d.dft_gga.notna()
    add_mean(s, "MAE, materials with a DFT hint", absr[hint], grp[hint])
    add_mean(s, "MAE, formula only (no DFT data)", absr[~hint], grp[~hint], note="what to expect for a new composition")
    h, g = d.dft_hybrid.notna(), d.dft_gga.notna()
    add_mean(s, "MAE where a hybrid DFT gap exists", absr[h], grp[h], baseline=(d.dft_hybrid - d.gap_ev)[h].abs().mean(),
             baseline_rule="use the hybrid DFT gap (HSE06/TBmBJ) directly")
    add_mean(s, "MAE where a GGA DFT gap exists", absr[g], grp[g], baseline=(d.dft_gga - d.gap_ev)[g].abs().mean(),
             baseline_rule="use the GGA DFT gap (PBE/OptB88vdW) directly")
    b = d.basis.str.contains("Borlido")
    add_mean(s, "MAE on the Borlido benchmark materials", absr[b], grp[b], note="best-curated experimental labels")
    solar = (d.gap_ev >= 1.0) & (d.gap_ev <= 3.5)
    add_mean(s, "MAE for gaps of 1–3.5 eV (solar-cell layers)", absr[solar], grp[solar])

    s = "A2. Band gap detail"
    for label, m in (("Borlido labels", b), ("curated-literature labels", d.basis.str.contains("curated") & ~b),
                     ("consensus labels (Zhuo 2018)", d.basis.str.startswith("consensus"))):
        add_mean(s, f"MAE, {label}", absr[m], grp[m])
    chem = d.formula.map(chemistry)
    for c in ("oxide", "chalcogenide", "halide", "pnictide", "other"):
        add_mean(s, f"MAE, {c}s" if c != "other" else "MAE, other chemistries", absr[chem == c], grp[chem == c])
    for label, m in (("gaps below 1 eV", d.gap_ev < 1.0), ("gaps above 3.5 eV", d.gap_ev > 3.5)):
        add_mean(s, f"MAE, {label}", absr[m], grp[m])
    add_mean(s, "Predicted as metal (< 0.1 eV)", d.predicted < 0.1, grp, "%", 100,
             note="semiconductors the model would call metallic")
    multi = pd.read_csv(BAND_GAPS).query("n_reports >= 2 and gap_ev > 0.001")
    spread = multi["values"].astype(str).map(lambda t: np.ptp([float(x) for x in t.split(";")]))
    add(s, "Label noise floor: spread between reports of one material", spread.median(), "eV", len(spread),
        note="median max − min of the agreeing reports; errors below this cannot be judged")
    print(f"A. band gap CV MAE {absr.mean():.3f} eV on {len(d)} materials")
    return d


# --------------------------------------------------------------------------- B. band edges

def validate_edges() -> pd.DataFrame:
    edges = pd.read_csv(BAND_EDGES)
    meas = edges[(edges.basis == "measured") & ~edges.formula.str.startswith("organic:")]
    rows = []
    for r in meas.itertuples():
        hidden = resolve_layer(r.formula, use_measured_edges=False)
        rows.append({"formula": r.formula, "measured_vbm": r.vbm_ev, "predicted_vbm": hidden.vbm_ev,
                     "fallback": hidden.edge_kind})
    d = pd.DataFrame(rows)
    d["error"] = d.predicted_vbm - d.measured_vbm
    s = "B. Band edges (measured VBM hidden)"
    note = "what the tool predicts for a material without photoemission data"
    for kind, label in (("surface", "hybrid-DFT surfaces"), ("estimate", "Butler–Ginley estimate")):
        k = d[d.fallback == kind]
        add_mean(s, f"VBM MAE, {label}", k.error.abs(), k.formula, note=note)
        add_mean(s, f"VBM bias, {label}", k.error, k.formula, note="negative = predicted too deep")
    multi = meas[meas.n_values > 1]
    add(s, "Spread between independent measurements of one material", multi.vbm_spread_ev.median(), "eV", len(multi),
        note="noise floor of the reference data itself (median)")
    add_mean("B2. Band edges summary", "VBM MAE, fallback rule (all)", d.error.abs(), d.formula,
             note="surfaces where available, else Butler–Ginley", primary=True)
    print(f"B. band edges: {len(d)} measured materials checked")
    return d


# --------------------------------------------------------------------------- C. junctions

def validate_measured_offsets() -> pd.DataFrame:
    ref = pd.read_csv(VALIDATION_OFFSETS)
    s = f"C1. Measured band offsets ({len(ref)} interfaces, InterMat Table 2)"
    rows = []
    for scope, use in SCOPES.items():
        for i, r in enumerate(ref.itertuples()):
            j = predict_stack([r.a, r.b], use)["junctions"][0]
            rows.append({"scope": scope, "interface": i, "measured": r.abs_vbo_ev,
                         "predicted": abs(j["vbo_ev"]) if j["type"] else np.nan, "sigma": j.get("vbo_sigma_ev")})
        d = pd.DataFrame([x for x in rows if x["scope"] == scope])
        add_mean(s, f"|ΔEv| MAE, {scope}", (d.predicted - d.measured).abs(), d.interface,
                 baseline=d.measured.abs().mean(), baseline_rule="predict ΔEv = 0",
                 note="with measured edges this is partly circular: several of these layers' electron affinities "
                      "come from InterMat Table 1" if use else "Butler–Ginley / hybrid-DFT surfaces only",
                 primary=not use)
    print("C1. measured offsets done")
    return pd.DataFrame(rows)


def parse_gold() -> pd.DataFrame:
    """One row per junction of the gold stacks: names, gold type, gold offsets (next − previous layer)."""
    g = pd.read_csv(GOLD_STACKS)
    g = g[g["list of devices"].notna() & g["s.no"].notna()]
    rows, skipped = [], []
    for r in g.itertuples(index=False):
        names = [x.strip() for x in str(r[2]).split("|")]
        try:
            cbm, vbm = ([float(x) for x in str(c).split("|")] for c in (r[4], r[5]))
        except ValueError:
            skipped.append(f"{r[0]:g} (band edges are not numbers)")
            continue
        if not len(names) == len(cbm) == len(vbm):
            skipped.append(f"{r[0]:g} ({len(names)} layers, {len(cbm)} CBMs, {len(vbm)} VBMs)")
            continue
        types = {k.strip(): t for k, t in re.findall(r"([^;:]+):\s*Type\s+(I{1,3})", str(r[6]))}
        borderline = {k.strip() for k in re.findall(r"([^;:]+):\s*Type\s+I{1,3}\s*\(borderline", str(r[6]))}
        for i in range(len(names) - 1):
            key = f"{names[i]}/{names[i + 1]}"
            rows.append({"stack": int(r[0]), "top": names[i], "bottom": names[i + 1],
                         "gold_type": types.get(key), "borderline": key in borderline,
                         "gold_dEc": cbm[i + 1] - cbm[i], "gold_dEv": vbm[i + 1] - vbm[i]})
    if skipped:
        print(f"   gold file: skipped {len(skipped)} stack(s): " + "; ".join(skipped))
    return pd.DataFrame(rows)


def validate_gold_stacks() -> tuple[pd.DataFrame, pd.DataFrame]:
    J = parse_gold()                                   # rows are in stack order, top to bottom
    layers = []
    for scope, use in SCOPES.items():
        tag = "tool" if use else "nomeas"
        res = []
        for stack, g in J.groupby("stack", sort=False):
            out = predict_stack([g.top.iloc[0]] + g.bottom.tolist(), use)
            res += out["junctions"]
            if use:
                layers += [{"stack": stack, "gap_kind": L["gap_kind"], "edge_kind": L["edge_kind"]} for L in out["layers"]]
        J[f"{tag}_type"] = [p["type"] for p in res]
        J[f"{tag}_dEc"] = [p.get("cbo_ev") for p in res]               # CBM(next) − CBM(previous)
        J[f"{tag}_dEv"] = [-p["vbo_ev"] if p.get("vbo_ev") is not None else None for p in res]
        J[f"{tag}_confidence"] = [p.get("confidence") for p in res]
        for t in ("I", "II", "III"):
            J[f"{tag}_p{t}"] = [p["type_probabilities"][t] if p.get("type_probabilities") else None for p in res]
        J[f"{tag}_offset_source"] = [p.get("offset_source") for p in res]
    J.round(3).to_csv(RESULTS / "junction_details.csv", index=False)

    s = "C2. Gold-standard device stacks"
    scored = J[J.gold_type.notna()]
    majority = scored.gold_type.mode().iloc[0]
    clear = (scored.gold_dEc.abs() >= CLEAR_MIN) & (scored.gold_dEv.abs() >= CLEAR_MIN)
    for scope, use in SCOPES.items():
        tag = "tool" if use else "nomeas"
        p = scored[scored[f"{tag}_type"].notna()]
        ok = p[f"{tag}_type"] == p.gold_type
        add_mean(s, f"Coverage, {scope}", scored[f"{tag}_type"].notna(), scored["stack"], "%", 100,
                 note="junctions with a prediction")
        add_mean(s, f"Type accuracy, {scope}", ok, p["stack"], "%", 100,
                 baseline=(p.gold_type == majority).mean() * 100, baseline_rule=f"always predict Type {majority}")
        c = clear.reindex(p.index)
        add_mean(s, f"Type accuracy on clear junctions (both offsets ≥ {CLEAR_MIN:g} eV), {scope}", ok[c], p["stack"][c],
                 "%", 100, baseline=(p.gold_type[c] == majority).mean() * 100,
                 baseline_rule=f"always predict Type {majority}",
                 note="near-zero offsets make the type hinge on a few tenths of an eV")
        conf = p[f"{tag}_confidence"] >= CONFIDENT
        add_mean(s, f"Type accuracy when confident (≥ {CONFIDENT:.0%}), {scope}".replace("%)", " %)"), ok[conf],
                 p["stack"][conf], "%", 100, note="junctions not flagged 'check with DFT'", primary=use)
        for q, label in (("dEc", "ΔEc"), ("dEv", "ΔEv")):
            add_mean(s, f"{label} MAE, {scope}", (p[f"{tag}_{q}"] - p[f"gold_{q}"]).abs(), p["stack"],
                     baseline=p[f"gold_{q}"].abs().mean(), baseline_rule=f"predict {label} = 0", primary=use)
            big = p[p[f"gold_{q}"].abs() >= SIGN_MIN]
            add_mean(s, f"{label} sign agreement (|gold| ≥ {SIGN_MIN:g} eV), {scope}",
                     np.sign(big[f"{tag}_{q}"].astype(float)) == np.sign(big[f"gold_{q}"]), big["stack"], "%", 100,
                     note="spike vs cliff: decides whether carriers are extracted or blocked", primary=use)
    print("C2. gold stacks done")
    return J, pd.DataFrame(layers)


# --------------------------------------------------------------------------- D. calibration

def validate_calibration(gaps: pd.DataFrame, edges: pd.DataFrame, offsets: pd.DataFrame, J: pd.DataFrame) -> None:
    s = "D. Uncertainty calibration"
    rule = "σ used by the tool (predict.py)"
    checks = [("ML band gap", gaps.error, SIGMA_GAP["ML"], gaps.group)]
    for kind, label in (("surface", "VBM, hybrid-DFT surfaces"), ("estimate", "VBM, Butler–Ginley")):
        k = edges[edges.fallback == kind]
        checks.append((label, k.error, SIGMA_VBM[kind], k.formula))
    o = offsets[(offsets.scope == "tool without measured edges") & offsets.predicted.notna()]
    checks.append(("ΔEv without measured edges", o.predicted - o.measured, o.sigma.to_numpy(), o.interface))
    for label, err, sigma, grp in checks:
        z = np.abs(np.asarray(err, float)) / sigma
        add_mean(s, f"{label}: errors within ±1σ", z <= 1, grp, "%", 100, note="68 % if σ is right")
        add_mean(s, f"{label}: errors within ±2σ", z <= 2, grp, "%", 100, note="95 % if σ is right")
        add_mean(s, f"{label}: σ suggested by the errors (RMS)", np.asarray(err, float) ** 2, grp, root=True,
                 baseline=float(np.mean(sigma)), baseline_rule=rule)

    p = J[J.gold_type.notna() & J.tool_type.notna()]
    probs = p[["tool_pI", "tool_pII", "tool_pIII"]].to_numpy(float)
    truth = np.stack([(p.gold_type == t).to_numpy() for t in ("I", "II", "III")], axis=1).astype(float)
    majority = (np.array(["I", "II", "III"]) == p.gold_type.mode().iloc[0]).astype(float)
    add_mean(s, "Brier score of the junction-type probabilities (gold stacks)", ((probs - truth) ** 2).sum(axis=1),
             p["stack"], "", baseline=((majority - truth) ** 2).sum(axis=1).mean(),
             baseline_rule="always Type II with certainty", note="0 = perfect, lower is better", primary=True)
    conf, ok = p.tool_confidence.to_numpy(float), (p.tool_type == p.gold_type).to_numpy()
    bins = np.digitize(conf, CONF_BINS) - 1
    ece = sum((bins == b).mean() * abs(ok[bins == b].mean() - conf[bins == b].mean())
              for b in np.unique(bins))
    add(s, "Expected calibration error (gold stacks)", ece * 100, "%", len(p),
        note=f"|accuracy − confidence| averaged over {len(CONF_BINS) - 1} confidence bins; 0 = perfectly calibrated")
    for b in range(len(CONF_BINS) - 1):
        m = bins == b
        lo, hi = CONF_BINS[b], min(CONF_BINS[b + 1], 1.0)
        add_mean(s, f"Reliability: accuracy at confidence {lo:.0%}–{hi:.0%}", ok[m], p["stack"][m], "%", 100,
                 baseline=conf[m].mean() * 100 if m.any() else None,
                 baseline_rule="mean confidence in the bin (equal when calibrated)")
    add_mean(s, f"Share of junctions called with ≥ {CONFIDENT:.0%} confidence".replace("%", " %"), conf >= CONFIDENT,
             p["stack"], "%", 100, note="how often the tool commits; read with the accuracy when confident",
             primary=True)
    print("D. calibration done")


# --------------------------------------------------------------------------- E. coverage

def data_coverage(gaps: pd.DataFrame, layers: pd.DataFrame) -> None:
    s = "E. Data coverage"
    edges = pd.read_csv(BAND_EDGES)
    add(s, "Semiconductors with a measured gap (training labels)", len(gaps), "", len(gaps))
    add(s, "Share with a GGA DFT hint", gaps.dft_gga.notna().mean() * 100, "%", len(gaps))
    add(s, "Share with a hybrid DFT hint", gaps.dft_hybrid.notna().mean() * 100, "%", len(gaps))
    add(s, "Materials with measured band edges", (edges.basis == "measured").sum(), "")
    add(s, "Oxides with hybrid-DFT surface edges", (edges.basis == "hybrid-DFT surfaces").sum(), "")
    add(s, "Measured interface offsets (used by the tool)", len(pd.read_csv(MEASURED_OFFSETS)), "")
    add(s, "Reference offsets for C1", len(pd.read_csv(VALIDATION_OFFSETS)), "")
    add_mean(s, "Device-stack layers with a measured gap", layers.gap_kind == "measured", layers["stack"], "%", 100)
    add_mean(s, "Device-stack layers with measured band edges", layers.edge_kind == "measured", layers["stack"], "%", 100)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    gaps = validate_gaps()
    edges = validate_edges()
    offsets = validate_measured_offsets()
    J, layers = validate_gold_stacks()
    validate_calibration(gaps, edges, offsets, J)
    data_coverage(gaps, layers)
    table = pd.DataFrame(METRICS).astype({"n": "Int64"})
    table.to_csv(RESULTS / "metrics.csv", index=False)
    (RESULTS / "metrics.json").write_text(json.dumps({
        "generated": datetime.date.today().isoformat(), "metrics": METRICS}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nwritten {RESULTS / 'metrics.csv'} ({len(table)} metrics). Primary metrics:")
    print(table[table.primary][["section", "metric", "value", "unit", "ci_low", "ci_high", "n"]].to_string(index=False))
