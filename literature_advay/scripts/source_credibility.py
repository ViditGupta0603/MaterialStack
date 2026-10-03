"""Cross-check sources against each other. Read-only."""
import sqlite3

import numpy as np
import pandas as pd

from materialstack.chem import butler_ginley_edges

con = sqlite3.connect("data/materials_db.sqlite")
r = pd.read_sql_query("""SELECT r.material_id, r.source, r.method, r.gap_ev, r.vbm_ev, r.cbm_ev,
                                m.formula_reduced, m.mulliken_chi chi, m.family
                         FROM records r JOIN materials m USING(material_id)""", con)

# ---------------------------------------------------------------- experimental gaps vs each other
print("1. Experimental gap sources: do they agree with each other on shared materials?")
ex = r[r.method == "experiment"]
med = ex.groupby(["material_id", "source"]).gap_ev.median().unstack()
srcs = list(med.columns)
for i, a in enumerate(srcs):
    for b in srcs[i + 1:]:
        both = med[[a, b]].dropna()
        both = both[(both[a] > 0) & (both[b] > 0)]
        if len(both) < 20:
            continue
        d = (both[a] - both[b]).abs()
        print(f"   {a:24s} vs {b:24s} n={len(both):5d}  MAE={d.mean():.3f}  "
              f"identical(<0.01)={np.mean(d < .01):.0%}  >0.5 eV apart={np.mean(d > .5):.0%}")
within = ex[ex.source == "expt_gap"].groupby("material_id").gap_ev.agg(["size", "min", "max"])
within = within[within["size"] > 1]
print(f"   expt_gap: {len(within)} materials have >1 value; spread median={(within['max']-within['min']).median():.2f} eV, "
      f">1 eV spread: {np.mean(within['max']-within['min'] > 1):.0%}")
print(f"   share of experimental rows that are metals (gap=0): "
      f"{ex.groupby('source').gap_ev.apply(lambda g: np.mean(g <= 0.001)).round(2).to_dict()}")

# ---------------------------------------------------------------- DFT methods vs experiment
print("\n2. Each computed source vs experiment (reference = median experimental gap, non-metals)")
ref = ex.groupby("material_id").gap_ev.median()
ref = ref[ref > 0.1]
dft = r[r.method != "experiment"]
rows = []
for (src, meth), g in dft.groupby(["source", "method"]):
    gm = g.groupby("material_id").gap_ev.median()
    j = pd.concat([gm, ref], axis=1, keys=["calc", "expt"]).dropna()
    if len(j) < 15:
        continue
    e = j.calc - j.expt
    rows.append(dict(source=src, method=meth, n=len(j), MAE=e.abs().mean(), bias=e.mean(),
                     r2=1 - (e ** 2).sum() / ((j.expt - j.expt.mean()) ** 2).sum()))
print(pd.DataFrame(rows).sort_values("MAE").round(3).to_string(index=False))

# ---------------------------------------------------------------- Castelli edges = Butler-Ginley?
print("\n3. Castelli band edges: computed or derived from a formula?")
c = r[(r.source == "castelli_perovskites") & r.cbm_ev.notna() & r.chi.notna()].copy()
c["cbm_bg"] = [butler_ginley_edges(x, g)[0] for x, g in zip(c.chi, c.gap_ev)]
d = (c.cbm_ev - c.cbm_bg)
print(f"   rows={len(c)}  metals(gap=0)={np.mean(c.gap_ev <= 0.001):.0%}  "
      f"|CBM - ButlerGinley(chi, Eg)|: median={d.abs().median():.3f} eV, "
      f"within 0.1 eV={np.mean(d.abs() < .1):.0%}, corr={np.corrcoef(c.cbm_ev, c.cbm_bg)[0,1]:.3f}")
print(f"   mean offset CBM - BG = {d.mean():+.3f} eV (std {d.std():.3f})")
nm = c[c.gap_ev > 0.001]
dn = nm.cbm_ev - nm.cbm_bg
print(f"   non-metals only n={len(nm)}: median |diff|={dn.abs().median():.3f}  std={dn.std():.3f}")

# ---------------------------------------------------------------- coverage for device-relevant layers
print("\n4. Coverage of typical solar-cell layers")
layers = ["TiO2", "SnO2", "ZnO", "NiO", "Cu2O", "CuI", "CuSCN", "MoO3", "WO3", "CdS", "CdTe", "CsPbI3",
          "CH3NH3PbI3", "CsPbBr3", "FAPbI3", "Cs2AgBiBr6", "CsSnI3", "Si", "GaAs", "CIGS", "CuInSe2",
          "Cu2ZnSnS4", "Sb2Se3", "MoS2", "Ga2O3", "In2O3", "ZnS", "V2O5"]
best = pd.read_sql_query("SELECT m.formula_reduced f, b.best_gap_method gm, b.best_edge_method em "
                         "FROM materials_best b JOIN materials m USING(material_id)", con)
al = pd.read_sql_query("SELECT alias, formula FROM aliases", con)
amap = dict(zip(al.alias.str.lower(), al.formula))
from materialstack.chem import clean_material_name
out = []
for L in layers:
    f = clean_material_name(amap.get(L.lower(), L)).formula_clean
    b = best[best.f == f]
    gm = ",".join(sorted(set(b.gm.dropna()))) or "-"
    em = ",".join(sorted(set(b.em.dropna()))) or "-"
    out.append((L, f, gm, em))
print(pd.DataFrame(out, columns=["layer", "formula", "best gap method", "edge data"]).to_string(index=False))
