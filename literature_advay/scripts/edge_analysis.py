"""Diagnose where VBM/CBM + band alignment error comes from. Read-only on the DB."""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

from materialstack.db import connect
from materialstack import models as M
from materialstack.chem import butler_ginley_edges

con = connect()
df, _ = M.load_edge_frame(con)
cols = joblib_cols = None
import joblib
bundle = joblib.load(M.EDGE_MODEL_PATH)
cols = bundle["feature_columns"]

print("=" * 70)
print("A. Edge training set after cleaning")
print(df.groupby(["source", "method"]).agg(n=("material_id", "size"),
      gap_mean=("gap_ev", "mean"), gap_med=("gap_ev", "median")).round(2))
print("family mix:", df.family.value_counts().head(8).to_dict())

X = M._matrix(df, cols)
y = df.delta_cbm.values
g = df.group_key.values
w = df.sample_weight.values
gap = df.gap_ev.values
cbm_bg = df.cbm_bg.values

oof = np.zeros_like(y)
for tr, te in GroupKFold(5).split(X, y, g):
    m = M.EdgeCorrectionModel()
    m.fit(X[tr], y[tr], sample_weight=w[tr], feature_names=cols)
    oof[te] = m.predict_delta(X[te])
df["oof_delta"] = oof
df["err_cbm"] = (cbm_bg + oof) - df.cbm_ev
df["err_bg"] = cbm_bg - df.cbm_ev

print("\nB. CV error by source (CBM; VBM identical because VBM = CBM - Eg_label)")
def mae(s): return float(np.mean(np.abs(s)))
out = df.groupby("source").agg(n=("err_cbm", "size"),
                               ML_MAE=("err_cbm", mae),
                               ButlerGinley_only_MAE=("err_bg", mae))
out.loc["ALL"] = [len(df), mae(df.err_cbm), mae(df.err_bg)]
print(out.round(3))

surf = df[df.source == "jarvis_surfacedb"]
print(f"  surfacedb only, ML MAE={mae(surf.err_cbm):.3f}  BG-only={mae(surf.err_bg):.3f}  "
      f"mean-delta baseline={mae(surf.delta_cbm - surf.delta_cbm.mean()):.3f}")

# --------------------------------------------------------------- C. gap swap at inference
print("\nC. Train/inference mismatch: the model is scored with the label's own DFT gap,")
print("   but at prediction time it receives the experimental / ML gap.")
best = pd.read_sql_query("SELECT material_id, best_gap, best_gap_method FROM materials_best", con)
d = df.merge(best, on="material_id", how="left")
d = d[d.best_gap_method.isin(["experiment", "literature", "HSE06"]) & (d.best_gap > 0)]
print(f"   materials with an expt/HSE06 gap: {len(d)} (of {len(df)})")
if len(d):
    print(f"   |Eg_trusted - Eg_label| mean = {mae(d.best_gap - d.gap_ev):.3f} eV  "
          f"(label methods: {d.method.value_counts().to_dict()})")
    Xs = M._matrix(d, cols).copy()
    gi = cols.index("feat_gap_ev"); bi = cols.index("feat_cbm_bg")
    cbm_bg_t = np.array([butler_ginley_edges(c, e)[0] for c, e in zip(d.chi, d.best_gap)])
    Xs[:, gi] = d.best_gap.values; Xs[:, bi] = cbm_bg_t
    # OOF model per fold for these rows
    pred_delta = np.full(len(d), np.nan)
    idx = df.reset_index().merge(d[["material_id"]], on="material_id")["index"].values
    for tr, te in GroupKFold(5).split(X, y, g):
        m = M.EdgeCorrectionModel(); m.fit(X[tr], y[tr], sample_weight=w[tr], feature_names=cols)
        sel = np.isin(idx, te)
        if sel.any():
            pred_delta[sel] = m.predict_delta(Xs[sel])
    cbm_p = cbm_bg_t + pred_delta
    vbm_p = cbm_p - d.best_gap.values
    print(f"   pipeline-style (trusted gap in): CBM MAE={mae(cbm_p - d.cbm_ev):.3f}  "
          f"VBM MAE={mae(vbm_p - d.vbm_ev):.3f}")
    print(f"   label-gap in (as reported):     CBM MAE={mae(d.err_cbm):.3f}")
    print(f"   mean shift VBM_pred-VBM_label={np.mean(vbm_p - d.vbm_ev):+.3f}  "
          f"CBM={np.mean(cbm_p - d.cbm_ev):+.3f}  (baseline splits a gap change symmetrically)")

# --------------------------------------------------------------- D. gap-correction partition
print("\nD. How a gap correction splits between VBM and CBM (surfacedb, same surface, OptB88 vs TBmBJ)")
s = pd.read_sql_query("""SELECT material_id, method, external_id, gap_ev, vbm_ev, cbm_ev, extra
                         FROM records WHERE source='jarvis_surfacedb'""", con)
s["key"] = s.external_id.str.replace(r"(?i)[-_]?(tbmbj|optb88vdw)", "", regex=True)
p = s.pivot_table(index=["material_id", "key"], columns="method",
                  values=["gap_ev", "vbm_ev", "cbm_ev"]).dropna()
if len(p):
    o, t = "DFT-surface-OptB88vdW", "DFT-surface-TBmBJ"
    dg = p[("gap_ev", t)] - p[("gap_ev", o)]
    dv = p[("vbm_ev", t)] - p[("vbm_ev", o)]
    dc = p[("cbm_ev", t)] - p[("cbm_ev", o)]
    frac_v = (-dv / dg)[dg > 0.3]
    print(f"   paired surfaces n={len(p)}; mean ΔEg={dg.mean():.2f}  ΔVBM={dv.mean():+.2f}  ΔCBM={dc.mean():+.2f}")
    print(f"   fraction of gap opening taken by VBM going down: median={frac_v.median():.2f} "
          f"(IQR {frac_v.quantile(.25):.2f}-{frac_v.quantile(.75):.2f}); this pairing is our loader's own construction (CBM = VBM + TBmBJ gap), so it is 0 by definition")
else:
    print("   could not pair OptB88/TBmBJ surfaces by external_id;", s.external_id.head(3).tolist())

# --------------------------------------------------------------- E. surface spread
print("\nE. Surface/termination spread for the same material (irreducible for bulk-only features)")
sp = s.groupby(["material_id", "method"]).agg(n=("vbm_ev", "size"),
                                               vbm_range=("vbm_ev", lambda v: v.max() - v.min()))
sp = sp[sp.n > 1]
print(f"   materials with >1 surface: {len(sp)}; VBM range median={sp.vbm_range.median():.2f} eV, "
      f"75th pct={sp.vbm_range.quantile(.75):.2f}, max={sp.vbm_range.max():.2f}")

# --------------------------------------------------------------- F. junction validation
print("\nF. End-to-end: pipeline VBO vs JARVIS DFT interface VBO (interfacedb, never used today)")
from materialstack.predict import resolve_layer, _load_models, _load_structure_models
it = pd.read_sql_query("SELECT material_a, material_b, vbo_ev, method, extra FROM interfaces "
                       "WHERE vbo_ev IS NOT NULL", con)
idmap = pd.read_sql_query("SELECT DISTINCT external_id, m.formula_reduced FROM records r "
                          "JOIN materials m USING(material_id) WHERE source='jarvis_dft_3d'", con)
idmap = dict(zip(idmap.external_id, idmap.formula_reduced))
eg_b, ed_b = _load_models(M.MODELS_DIR); st_b, px_b = _load_structure_models(M.MODELS_DIR)
cache = {}
def layer(f, rank):
    k = (f, rank)
    if k not in cache:
        cache[k] = resolve_layer(f, con=con, eg_bundle=eg_b, edge_bundle=ed_b, struct_bundle=st_b,
                                 proxy_bundles=px_b, max_lookup_rank=rank)
    return cache[k]
rows = []
for r in it.itertuples():
    fa, fb = idmap.get(r.material_a), idmap.get(r.material_b)
    if not fa or not fb:
        continue
    for rank in (2, 0):
        A, B = layer(fa, rank), layer(fb, rank)
        if None in (A.vbm_ev, B.vbm_ev, A.cbm_ev, B.cbm_ev):
            continue
        rows.append(dict(a=fa, b=fb, rank=rank, dft=r.vbo_ev,
                         pred=A.vbm_ev - B.vbm_ev, edgesrc=A.edge_source.split(":")[0] + "/" +
                         B.edge_source.split(":")[0]))
R = pd.DataFrame(rows)
print(f"   resolved {R.drop_duplicates(['a','b']).shape[0]} of {len(it)} interfaces")
for rank, sub in R.groupby("rank"):
    c_pos = np.corrcoef(sub.pred, sub.dft)[0, 1]
    sign = 1 if c_pos >= 0 else -1
    e = sub.pred - sign * sub.dft
    agree = np.mean(np.sign(sub.pred) == np.sign(sign * sub.dft))
    lbl = "lookup-first (default)" if rank == 2 else "pure ML (rank 0)"
    print(f"   {lbl:24s} n={len(sub)} corr={c_pos:+.2f}  VBO MAE={mae(e):.3f}  "
          f"sign agreement={agree:.0%}  |VBO_dft|<0.3: {np.mean(np.abs(sub.dft) < 0.3):.0%}")
    print("     by edge source:", sub.assign(e=np.abs(e)).groupby("edgesrc").e.agg(["size", "mean"])
          .round(3).to_dict("index"))
