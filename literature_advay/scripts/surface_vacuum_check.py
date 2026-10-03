"""Findings F1 + F7: JARVIS surface edges need vacuum alignment; compare with interface offsets.

Downloads surfacedb / interfacedb (a few MB) into data/cache via jarvis-tools.
"""
import numpy as np
import pandas as pd
from jarvis.db.figshare import data

from materialstack.config import CACHE_DIR

s = pd.DataFrame(data("surfacedb", store_dir=str(CACHE_DIR)))
s = s.drop(columns=["phi", "atoms", "initial_atoms", "final_atoms"])
s["jid"] = s.name.str.extract(r"(JVASP-\d+)")
s["mill"] = s.name.str.extract(r"miller_(\d_\d_\d)")[0].str.replace("_", "")
s["vbm_vac"] = s.surf_vbm - s.avg_max          # IP = E_vac - VBM (InterMat definition)
s["cbm_vac"] = s.surf_cbm - s.avg_max
s["wf"] = s.avg_max - s.efermi

print("F1. Stored (surf_vbm) vs vacuum-aligned (surf_vbm - avg_max) VBM for known semiconductors")
known = {"Si": -5.2, "GaAs": -5.5, "ZnO": -7.6, "CdTe": -5.8, "GaN": -6.8}
k = s[s.formula.isin(known)]
pd.set_option("display.width", 200)
print(k[["formula", "jid", "mill", "surf_vbm", "vbm_vac", "wf"]].round(2).sort_values("formula").to_string(index=False))
print("approximate experimental VBM:", known)

ok = (s.avg_max < 12) & s.wf.between(2, 9) & s.mill.notna()
print(f"\nsanity filter keeps {ok.sum()} of {len(s)} surfaces")
s = s[ok]

print("\nF7. Plane-matched vacuum alignment vs JARVIS interface offsets")
S = s.groupby(["jid", "mill"]).agg(vac=("vbm_vac", "median"), raw=("surf_vbm", "median"))
i = pd.DataFrame(data("interfacedb", store_dir=str(CACHE_DIR)))
print(f"interfaces: {len(i)}, numeric offset: {i.offset.map(lambda v: isinstance(v, float)).sum()}")
i = i[i.offset.map(lambda v: isinstance(v, float))].copy()
i["offset"] = i.offset.astype(float)
x = i.jid.str.extract(r"(JVASP-\d+)_(JVASP-\d+)_film_miller_(\d_\d_\d)_sub_miller_(\d_\d_\d)")
i["a"], i["b"] = x[0], x[1]
i["ma"], i["mb"] = x[2].str.replace("_", ""), x[3].str.replace("_", "")
m = i.merge(S, left_on=["a", "ma"], right_index=True).merge(S, left_on=["b", "mb"], right_index=True,
                                                             suffixes=("_a", "_b"))
o = m.offset.to_numpy()
print(f"matched same-plane interfaces: {len(m)}")
for col in ("raw", "vac"):
    d = (m[col + "_a"] - m[col + "_b"]).to_numpy()
    for sg in (1, -1):
        e = d - sg * o
        print(f"  {col:3s} sign {sg:+d}: corr={np.corrcoef(d, sg * o)[0, 1]:+.2f}  MAE={np.abs(e).mean():.2f}  "
              f"sign agreement={np.mean(np.sign(d) == np.sign(sg * o)):.0%}")
print(f"  baseline (predict 0): MAE={np.abs(o).mean():.2f}")
