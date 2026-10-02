"""Composition featurization, cached into the materials DB.

features_composition: one row per material (157 matminer features + family flags).
Structure features live in materialstack.structure (features_structure table).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from materialstack.config import DB_PATH
from materialstack.db import connect

log = logging.getLogger("materialstack.features")

FAMILIES = ["oxide", "halide", "chalcogenide", "pnictide", "perovskite_ABX3", "double_perovskite",
            "hybrid_perovskite", "intermetallic", "organic", "other"]
ORBITALS = ["s", "p", "d", "f"]


def composition_featurizer():
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (AtomicOrbitals, BandCenter, ElementProperty, IonProperty,
                                                  Stoichiometry, ValenceOrbital)
    return MultipleFeaturizer([ElementProperty.from_preset("magpie"), Stoichiometry(), ValenceOrbital(),
                               IonProperty(fast=True), AtomicOrbitals(), BandCenter()])


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Turn AtomicOrbitals string columns into numeric columns so the table is fully numeric."""
    out = df.copy()
    for col in ("HOMO_character", "LUMO_character"):
        if col in out:
            for o in ORBITALS:
                out[f"{col}_{o}"] = (out[col] == o).astype(int)
            out = out.drop(columns=[col])
    for col in ("HOMO_element", "LUMO_element"):
        if col in out:
            from pymatgen.core import Element
            out[f"{col}_Z"] = out[col].map(lambda s: Element(s).Z if isinstance(s, str) else np.nan)
            out = out.drop(columns=[col])
    if "compound possible" in out:
        out["compound possible"] = out["compound possible"].astype(float)
    for c in ("is_centrosymmetric",):
        if c in out:
            out[c] = out[c].astype(float)
    if "crystal_system" in out:
        out = out.drop(columns=["crystal_system"])
    return out


def featurize_compositions(con: sqlite3.Connection, *, n_jobs: int = 1, chunk: int = 1000,
                           only_missing: bool = True) -> int:
    mats = pd.read_sql_query(
        "SELECT material_id, formula_reduced, family, n_elements, anonymized_formula, is_perovskite_like, "
        "tolerance_factor, octahedral_factor, mulliken_chi FROM materials", con)
    if only_missing and _table_exists(con, "features_composition"):
        done = {r[0] for r in con.execute("SELECT material_id FROM features_composition")}
        mats = mats[~mats.material_id.isin(done)]
    if mats.empty:
        return 0
    feat = composition_featurizer()
    labels = feat.feature_labels()
    total = 0
    print(f"featurizing {len(mats)} compositions ...", flush=True)
    for start in range(0, len(mats), chunk):
        part = mats.iloc[start:start + chunk].copy()
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rec in part.itertuples(index=False):
                try:
                    vals = feat.featurize(Composition(rec.formula_reduced))
                except Exception:
                    vals = [np.nan] * len(labels)
                rows.append(vals)
        feat_df = pd.DataFrame(rows, columns=labels, index=part.index)
        part = pd.concat([part.drop(columns=["formula_reduced"]), feat_df], axis=1)
        for fam in FAMILIES:
            part[f"family_{fam}"] = (part.family == fam).astype(int)
        part = part.drop(columns=["family"])
        part = _encode_categoricals(part)
        part.to_sql("features_composition", con, if_exists="append", index=False)
        con.commit()
        total += len(part)
        print(f"features_composition: {min(start + chunk, len(mats))} / {len(mats)}", flush=True)
        log.info("features_composition: %d / %d", min(start + chunk, len(mats)), len(mats))
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_fc_material ON features_composition(material_id)")
    con.commit()
    return total


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def feature_row(con: sqlite3.Connection, material_id: int) -> pd.Series | None:
    df = pd.read_sql_query("SELECT * FROM features_composition WHERE material_id=?", con, params=(material_id,))
    return None if df.empty else df.iloc[0]


def featurize_formula(formula: str, *, family: str | None = None) -> pd.Series:
    """On-the-fly Magpie-style features for a formula not (yet) stored in the DB."""
    from materialstack.chem import classify_family, mulliken_chi, try_composition

    comp = try_composition(formula)
    if comp is None:
        raise ValueError(f"unparseable formula: {formula}")
    meta = classify_family(comp)
    fam = family or meta["family"]
    chi = mulliken_chi(comp)
    feat = composition_featurizer()
    labels = feat.feature_labels()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            vals = feat.featurize(comp)
        except Exception:
            vals = [np.nan] * len(labels)
    row = {lab: vals[i] for i, lab in enumerate(labels)}
    row.update({
        "material_id": -1,
        "n_elements": meta["n_elements"],
        "anonymized_formula": meta["anonymized_formula"],
        "is_perovskite_like": meta["is_perovskite_like"],
        "tolerance_factor": meta["tolerance_factor"],
        "octahedral_factor": meta["octahedral_factor"],
        "mulliken_chi": chi,
    })
    for f in FAMILIES:
        row[f"family_{f}"] = 1 if fam == f else 0
    return _encode_categoricals(pd.DataFrame([row])).iloc[0]


def run(n_jobs: int = 1, structures: bool = True, db_path=DB_PATH) -> dict:
    from materialstack.structure import featurize_structures

    con = connect(db_path)
    n_c = featurize_compositions(con, n_jobs=n_jobs)
    n_s = featurize_structures(con, n_jobs=n_jobs) if structures else 0
    con.close()
    return {"features_composition_added": n_c, "features_structure_added": n_s}
