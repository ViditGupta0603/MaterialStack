"""Crystal-structure features, polymorph selection and DFT-proxy (transfer-learning) models.

features_structure: one row per structure, fast geometric/symmetry descriptors (prefix ``sf_``).
structure_proxy:    one row per structure, DFT band gaps predicted by models trained on
                    JARVIS / SNUMAT DFT data (prefix ``proxy_``). Structures that were part of a
                    proxy's training set get out-of-fold predictions, so downstream models never
                    see a proxy value fitted on that structure's own chemistry.
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from materialstack.config import DB_PATH, MODELS_DIR

log = logging.getLogger("materialstack.structure")

NEIGHBOR_CUTOFF = 5.0
CN_TOLERANCE = 1.2

# Monolayers are not the bulk phase measured in experimental band-gap datasets.
EXCLUDED_POLYMORPH_SOURCES = {"jarvis_dft_2d"}
SOURCE_PRIORITY = {
    "jarvis_dft_3d": 0,
    "snumat": 1,
    "dielectric_constant": 2,
    "jarvis_halide_perovskites": 3,
    "castelli_perovskites": 4,
}
# Stored JARVIS hull energies carry a per-chemical-system offset, so only energies relative to the
# lowest polymorph of the same formula are meaningful. Polymorphs more than this above the lowest
# are usually hypothetical (most synthesized metastable phases lie within ~50 meV/atom).
MEAN_MAX_E_REL = 0.05
POLYMORPH_POLICIES = ("ground_state", "mean")

PROXY_TARGETS: dict[str, dict[str, Any]] = {
    "optb88vdw": {"method": "OptB88vdW", "sources": ("jarvis_dft_3d",)},
    "tbmbj": {"method": "TBmBJ", "sources": ("jarvis_dft_3d",)},
    "hse06": {"method": "HSE06", "sources": ("snumat", "jarvis_halide_perovskites", "jarvis_dft_3d")},
}
PROXY_COLUMNS = [f"proxy_{k}" for k in PROXY_TARGETS]

CRYSTAL_SYSTEMS = [
    ("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74), ("tetragonal", 75, 142),
    ("trigonal", 143, 167), ("hexagonal", 168, 194), ("cubic", 195, 230),
]
_CENTRO_RANGES = [(2, 2), (10, 15), (47, 74), (83, 88), (123, 142), (147, 148), (162, 167),
                  (175, 176), (191, 194), (200, 206), (221, 230)]


# --------------------------------------------------------------------------- featurizer

@lru_cache(maxsize=None)
def _radii(symbol: str) -> tuple[float, float]:
    """(covalent radius, atomic radius) in Å with conservative fallbacks."""
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    from pymatgen.core import Element

    cov = CovalentRadius.radius.get(symbol)
    try:
        el = Element(symbol)
        atomic = float(el.atomic_radius) if el.atomic_radius is not None else None
    except Exception:
        atomic = None
    cov = float(cov) if cov is not None else (atomic or 1.5)
    atomic = atomic or cov
    return cov, atomic


def _symmetry_features(sg: int | None) -> dict[str, float]:
    out: dict[str, float] = {"sf_space_group": float(sg) if sg else np.nan}
    for name, lo, hi in CRYSTAL_SYSTEMS:
        out[f"sf_cs_{name}"] = float(bool(sg) and lo <= sg <= hi)
    out["sf_centrosymmetric"] = (float(any(lo <= sg <= hi for lo, hi in _CENTRO_RANGES))
                                 if sg else np.nan)
    return out


def structure_features(struct, space_group: int | None = None, e_above_hull: float | None = None,
                       source: str | None = None) -> dict[str, float]:
    """Fast, composition-independent geometric descriptors of one crystal structure."""
    try:
        space_group = int(space_group) if space_group is not None and not pd.isna(space_group) else None
    except (TypeError, ValueError):
        space_group = None
    n = len(struct)
    vol = float(struct.volume)
    symbols = [site.specie.symbol for site in struct]
    cov = np.array([_radii(s)[0] for s in symbols])
    atomic = np.array([_radii(s)[1] for s in symbols])

    # Only cell-choice-invariant descriptors: CIFs mix primitive and conventional cells.
    feats: dict[str, float] = {
        "sf_volume_per_atom": vol / n,
        "sf_density": float(struct.density),
        "sf_packing_fraction": float((4.0 / 3.0) * math.pi * np.sum(atomic ** 3) / vol),
        "sf_e_above_hull": float(e_above_hull) if e_above_hull is not None else np.nan,
    }
    feats.update(_symmetry_features(space_group))

    centers, points, _, dists = struct.get_neighbor_list(r=NEIGHBOR_CUTOFF)
    keep = dists > 1e-6
    centers, points, dists = centers[keep], points[keep], dists[keep]
    dmin, ratio, cn, hetero = [], [], [], []
    if len(dists):
        order = np.argsort(centers, kind="stable")
        centers, points, dists = centers[order], points[order], dists[order]
        bounds = np.searchsorted(centers, np.arange(n + 1))
        for i in range(n):
            lo, hi = bounds[i], bounds[i + 1]
            if hi <= lo:
                continue
            d, p = dists[lo:hi], points[lo:hi]
            j = int(np.argmin(d))
            d0 = float(d[j])
            shell = d <= CN_TOLERANCE * d0
            dmin.append(d0)
            ratio.append(d0 / (cov[i] + cov[p[j]]))
            cn.append(float(shell.sum()))
            hetero.append(float(np.mean([symbols[k] != symbols[i] for k in p[shell]])))

    def _stats(prefix: str, vals: list[float]) -> None:
        a = np.asarray(vals, dtype=float)
        empty = a.size == 0
        feats[f"{prefix}_mean"] = np.nan if empty else float(a.mean())
        feats[f"{prefix}_std"] = np.nan if empty else float(a.std())
        feats[f"{prefix}_min"] = np.nan if empty else float(a.min())
        feats[f"{prefix}_max"] = np.nan if empty else float(a.max())

    _stats("sf_nn_dist", dmin)
    _stats("sf_bond_ratio", ratio)
    _stats("sf_cn", cn)
    feats["sf_hetero_frac"] = float(np.mean(hetero)) if hetero else np.nan
    feats["sf_is_2d_source"] = float(source in EXCLUDED_POLYMORPH_SOURCES)
    return feats


def _parse_cif(cif: str):
    from pymatgen.core import Structure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Structure.from_str(cif, fmt="cif")


def _featurize_row(row: tuple) -> dict[str, Any] | None:
    structure_id, material_id, cif, sg, e_hull, source = row
    try:
        struct = _parse_cif(cif)
        feats = structure_features(struct, int(sg) if sg else None,
                                   None if e_hull is None else float(e_hull), source)
    except Exception:
        return None
    return {"structure_id": int(structure_id), "material_id": int(material_id), **feats}


def featurize_structures(con: sqlite3.Connection, *, n_jobs: int = 1, chunk: int = 4000,
                         rebuild: bool = False) -> int:
    """Compute ``features_structure`` for every stored CIF (parallel, resumable)."""
    if rebuild:
        con.execute("DROP TABLE IF EXISTS features_structure")
        con.commit()
    done: set[int] = set()
    if _table_exists(con, "features_structure"):
        done = {r[0] for r in con.execute("SELECT structure_id FROM features_structure")}
    ids = [r[0] for r in con.execute("SELECT structure_id FROM structures WHERE cif IS NOT NULL "
                                     "ORDER BY structure_id")]
    todo = [i for i in ids if i not in done]
    log.info("features_structure: %d structures total, %d already done, %d to featurize",
             len(ids), len(done), len(todo))
    if not todo:
        return 0

    pool = None
    if n_jobs > 1:
        from multiprocessing import Pool
        pool = Pool(processes=n_jobs)
    total, failed, t0 = 0, 0, time.time()
    try:
        for start in range(0, len(todo), chunk):
            sub = todo[start:start + chunk]
            rows = con.execute(
                f"SELECT structure_id, material_id, cif, space_group_number, e_above_hull, source "
                f"FROM structures WHERE structure_id IN ({','.join(map(str, sub))})").fetchall()
            results = (pool.map(_featurize_row, rows, chunksize=50) if pool
                       else [_featurize_row(r) for r in rows])
            good = [r for r in results if r is not None]
            failed += len(results) - len(good)
            if good:
                pd.DataFrame(good).to_sql("features_structure", con, if_exists="append", index=False)
                con.commit()
            total += len(good)
            done_n = min(start + chunk, len(todo))
            rate = done_n / max(time.time() - t0, 1e-6)
            log.info("features_structure: %d / %d  (%.0f/s, %d failed, ETA %.1f min)",
                     done_n, len(todo), rate, failed, (len(todo) - done_n) / rate / 60)
    finally:
        if pool:
            pool.close()
            pool.join()
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_fs_structure ON features_structure(structure_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_fs_material ON features_structure(material_id)")
    con.commit()
    return total


def structure_feature_columns(con: sqlite3.Connection) -> list[str]:
    cols = [r[1] for r in con.execute("PRAGMA table_info(features_structure)")]
    return [c for c in cols if c.startswith("sf_") and c not in EXCLUDED_STRUCTURE_FEATURES]


# Absolute stored hull energy is dominated by the per-system offset described above.
EXCLUDED_STRUCTURE_FEATURES = {"sf_e_above_hull"}


# --------------------------------------------------------------------------- polymorph selection

def candidate_structures(con: sqlite3.Connection, material_ids: list[int] | None = None) -> pd.DataFrame:
    """Bulk candidate polymorphs (2D monolayers excluded), sorted best-first per material."""
    where = ""
    if material_ids is not None:
        if not material_ids:
            return pd.DataFrame(columns=["structure_id", "material_id", "source", "space_group_number",
                                         "e_above_hull", "n_sites"])
        where = f"AND material_id IN ({','.join(str(int(m)) for m in material_ids)})"
    excl = ",".join(f"'{s}'" for s in EXCLUDED_POLYMORPH_SOURCES)
    df = pd.read_sql_query(
        f"""SELECT structure_id, material_id, source, space_group_number, e_above_hull, n_sites
            FROM structures WHERE cif IS NOT NULL AND source NOT IN ({excl}) {where}""", con)
    df["e_rel"] = df.e_above_hull - df.groupby("material_id").e_above_hull.transform("min")
    df["_has_hull"] = df.e_rel.notna().astype(int)
    df["_prio"] = df.source.map(lambda s: SOURCE_PRIORITY.get(s, 9))
    df = df.sort_values(["material_id", "_has_hull", "e_rel", "_prio", "n_sites"],
                        ascending=[True, False, True, True, True], na_position="last")
    return df.drop(columns=["_has_hull", "_prio"]).reset_index(drop=True)


def select_structures(candidates: pd.DataFrame, policy: str = "ground_state") -> pd.DataFrame:
    """Pick structures per material.

    ground_state: lowest DFT energy among the formula's polymorphs (JARVIS), else most trusted
                  source — the conventional choice of the thermodynamically stable phase.
    mean:         one structure per distinct space group within MEAN_MAX_E_REL eV/atom of the
                  lowest polymorph (unranked sources only when no energies are known);
                  predictions are averaged downstream.
    """
    if policy not in POLYMORPH_POLICIES:
        raise ValueError(f"unknown polymorph policy {policy!r}; choose from {POLYMORPH_POLICIES}")
    if candidates.empty:
        return candidates
    if policy == "ground_state":
        return candidates.groupby("material_id", sort=False).head(1).reset_index(drop=True)
    has_energy = candidates.groupby("material_id").e_rel.transform(lambda s: s.notna().any())
    ok = (has_energy & (candidates.e_rel <= MEAN_MAX_E_REL)) | (~has_energy)
    sub = candidates[ok].copy()
    sub["_sg"] = sub.space_group_number.fillna(-sub.structure_id)
    sub = sub.drop_duplicates(["material_id", "_sg"], keep="first").drop(columns=["_sg"])
    missing = set(candidates.material_id) - set(sub.material_id)
    if missing:
        sub = pd.concat([sub, select_structures(candidates[candidates.material_id.isin(missing)])])
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------------- DFT proxy models

def _group_key_series(con: sqlite3.Connection, material_ids: pd.Series) -> np.ndarray:
    mats = pd.read_sql_query("SELECT material_id, elements, family FROM materials", con)
    key = mats.set_index("material_id").apply(
        lambda r: r.elements if isinstance(r.elements, str) and r.elements.strip() else (r.family or "unknown"),
        axis=1)
    return material_ids.map(key).fillna("unknown").to_numpy()


def _proxy_feature_frame(con: sqlite3.Connection) -> tuple[pd.DataFrame, list[str]]:
    """features_structure joined to features_composition (one row per structure)."""
    from materialstack.models import _feature_columns

    fs = pd.read_sql_query("SELECT * FROM features_structure", con)
    fc = pd.read_sql_query("SELECT * FROM features_composition", con)
    df = fs.merge(fc, on="material_id", how="inner")
    cols = [c for c in _feature_columns(df) if c != "structure_id" and c not in EXCLUDED_STRUCTURE_FEATURES]
    return df, cols


def train_dft_proxies(*, db_path: Path = DB_PATH, models_dir: Path = MODELS_DIR,
                      n_splits: int = 5) -> dict[str, Any]:
    """Train one two-stage LightGBM per DFT method on (composition + structure) features.

    Writes ``structure_proxy`` with out-of-fold predictions for training structures and
    full-model predictions for every other structure.
    """
    from materialstack.db import connect
    from materialstack.models import TwoStageGapModel

    models_dir = Path(models_dir)
    con = connect(db_path)
    t0 = time.time()
    if not _table_exists(con, "features_structure"):
        raise RuntimeError("features_structure missing — run `featurize-structures` first")

    df, cols = _proxy_feature_frame(con)
    log.info("Proxy feature frame: %d structures × %d features (composition + structure)", len(df), len(cols))
    X_all = df[cols].to_numpy(dtype=float)
    out = pd.DataFrame({"structure_id": df.structure_id.to_numpy()})
    report: dict[str, Any] = {"feature_columns": cols, "targets": {}}

    for key, spec in PROXY_TARGETS.items():
        src = ",".join(f"'{s}'" for s in spec["sources"])
        labels = pd.read_sql_query(
            f"""SELECT structure_id, AVG(gap_ev) AS gap_ev FROM records
                WHERE method = ? AND source IN ({src}) AND structure_id IS NOT NULL AND gap_ev IS NOT NULL
                GROUP BY structure_id""", con, params=(spec["method"],))
        tr = df[["structure_id", "material_id"]].reset_index().merge(labels, on="structure_id", how="inner")
        if len(tr) < 500:
            log.warning("proxy %s: only %d labelled structures — skipped", key, len(tr))
            continue
        idx = tr["index"].to_numpy()
        X, y = X_all[idx], tr.gap_ev.to_numpy(dtype=float)
        groups = _group_key_series(con, tr.material_id)
        log.info("proxy %s (%s): n=%d structures, %d element-set groups, %d metals",
                 key, spec["method"], len(y), pd.Series(groups).nunique(), int((y <= 1e-3).sum()))

        oof = np.zeros_like(y)
        for k, (a, b) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
            m = TwoStageGapModel(backend="lgbm", use_log1p=True).fit(X[a], y[a], feature_names=cols)
            oof[b] = m.predict(X[b])
            log.info("  fold %d/%d: MAE=%.3f eV", k + 1, n_splits, mean_absolute_error(y[b], oof[b]))
        full = TwoStageGapModel(backend="lgbm", use_log1p=True).fit(X, y, feature_names=cols)

        pred = full.predict(X_all)
        pred[idx] = oof
        out[f"proxy_{key}"] = pred
        non = y > 1e-3
        report["targets"][key] = {
            "method": spec["method"],
            "n_train": int(len(y)),
            "oof_mae_all": float(mean_absolute_error(y, oof)),
            "oof_mae_nonmetal": float(mean_absolute_error(y[non], oof[non])) if non.any() else None,
            "oof_r2_nonmetal": float(r2_score(y[non], oof[non])) if non.any() else None,
            "top_features": full.feature_importance(10),
        }
        log.info("proxy %s: OOF MAE=%.3f eV (non-metal %.3f, R² %.3f)", key,
                 report["targets"][key]["oof_mae_all"], report["targets"][key]["oof_mae_nonmetal"] or np.nan,
                 report["targets"][key]["oof_r2_nonmetal"] or np.nan)
        joblib.dump({"model": full, "feature_columns": cols, "kind": "dft_proxy", "method": spec["method"]},
                    models_dir / f"proxy_{key}.joblib")

    out.to_sql("structure_proxy", con, if_exists="replace", index=False)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_sp_structure ON structure_proxy(structure_id)")
    con.commit()
    con.close()
    report["seconds"] = round(time.time() - t0, 1)
    (models_dir / "proxy_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_proxy_bundles(models_dir: Path = MODELS_DIR) -> dict[str, dict[str, Any]]:
    out = {}
    for key in PROXY_TARGETS:
        p = Path(models_dir) / f"proxy_{key}.joblib"
        if p.exists():
            out[key] = joblib.load(p)
    return out


# --------------------------------------------------------------------------- stored DFT gaps as features

DFT_FEATURE_METHODS = {
    "OptB88vdW": "dft_optb88vdw",
    "TBmBJ": "dft_tbmbj",
    "HSE06": "dft_hse06",
    "PBE": "dft_pbe",
    "GLLB-SC": "dft_gllb_sc",
}
DFT_COLUMNS = list(DFT_FEATURE_METHODS.values())


def dft_gap_features(con: sqlite3.Connection, pairs: pd.DataFrame) -> pd.DataFrame:
    """DFT band gaps stored in the DB for each (material_id, structure_id) pair.

    Uses the calculation done on that exact structure when it exists, otherwise the material's
    median over all records of that method. ``structure_id`` may be missing (NaN).
    """
    out = pairs[["material_id"]].copy()
    if "structure_id" not in pairs:
        pairs = pairs.assign(structure_id=np.nan)
    mids = sorted(set(int(m) for m in pairs.material_id))
    if not mids:
        for col in DFT_COLUMNS:
            out[col] = np.nan
        return out
    methods = ",".join(f"'{m}'" for m in DFT_FEATURE_METHODS)
    recs = pd.read_sql_query(
        f"""SELECT material_id, structure_id, method, gap_ev FROM records
            WHERE method IN ({methods}) AND gap_ev IS NOT NULL
              AND material_id IN ({','.join(map(str, mids))})""", con)
    recs["col"] = recs.method.map(DFT_FEATURE_METHODS)
    by_mat = recs.groupby(["material_id", "col"]).gap_ev.median().unstack()
    by_struct = recs.dropna(subset=["structure_id"]).groupby(["structure_id", "col"]).gap_ev.mean().unstack()
    for col in DFT_COLUMNS:
        mat_vals = pairs.material_id.map(by_mat[col]) if col in by_mat else pd.Series(np.nan, index=pairs.index)
        st_vals = (pairs.structure_id.map(by_struct[col]) if col in by_struct
                   else pd.Series(np.nan, index=pairs.index))
        out[col] = st_vals.where(st_vals.notna(), mat_vals).to_numpy(dtype=float)
    return out


# --------------------------------------------------------------------------- inference helpers

def structure_rows(con: sqlite3.Connection, material_id: int, policy: str,
                   proxy_bundles: dict[str, dict[str, Any]] | None = None,
                   comp_row: pd.Series | None = None) -> list[dict[str, Any]]:
    """Structure + proxy feature rows for the polymorph(s) chosen by ``policy``."""
    chosen = select_structures(candidate_structures(con, [int(material_id)]), policy)
    rows: list[dict[str, Any]] = []
    for rec in chosen.itertuples(index=False):
        sid = int(rec.structure_id)
        feats = pd.read_sql_query("SELECT * FROM features_structure WHERE structure_id=?", con, params=(sid,))
        if feats.empty:
            cif = con.execute("SELECT cif FROM structures WHERE structure_id=?", (sid,)).fetchone()
            if not cif or not cif[0]:
                continue
            try:
                f = structure_features(_parse_cif(cif[0]), rec.space_group_number,
                                       rec.e_above_hull if pd.notna(rec.e_above_hull) else None, rec.source)
            except Exception:
                continue
        else:
            f = {c: v for c, v in feats.iloc[0].items() if c.startswith("sf_")}

        proxies: dict[str, float] = {}
        if _table_exists(con, "structure_proxy"):
            pr = pd.read_sql_query("SELECT * FROM structure_proxy WHERE structure_id=?", con, params=(sid,))
            if not pr.empty:
                proxies = {c: float(v) for c, v in pr.iloc[0].items() if c.startswith("proxy_") and pd.notna(v)}
        if proxy_bundles and comp_row is not None:
            for key, bundle in proxy_bundles.items():
                col = f"proxy_{key}"
                if col in proxies:
                    continue
                full = {**pd.Series(comp_row).to_dict(), **f}
                X = np.array([[_to_float(full.get(c)) for c in bundle["feature_columns"]]])
                proxies[col] = float(bundle["model"].predict(X)[0])

        dft = dft_gap_features(con, pd.DataFrame({"material_id": [int(material_id)], "structure_id": [sid]}))
        rows.append({
            "structure_id": sid,
            "source": rec.source,
            "space_group": int(rec.space_group_number) if pd.notna(rec.space_group_number) else None,
            "e_rel": float(rec.e_rel) if pd.notna(rec.e_rel) else None,
            "features": {**f, **proxies, **{c: float(dft[c].iloc[0]) for c in DFT_COLUMNS}},
        })
    return rows


def material_dft_features(con: sqlite3.Connection, material_id: int) -> dict[str, float]:
    dft = dft_gap_features(con, pd.DataFrame({"material_id": [int(material_id)]}))
    return {c: float(dft[c].iloc[0]) for c in DFT_COLUMNS}


def _to_float(v) -> float:
    try:
        if v is None:
            return np.nan
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None
