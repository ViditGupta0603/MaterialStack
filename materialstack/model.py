"""The one ML model: band gap of a semiconductor from its formula, plus a DFT gap when one is known.

Inputs for each material (one row of numbers):
  - ~140 composition features computed by matminer from the formula: statistics (mean, min, max, range…)
    of element properties such as electronegativity, atomic radius and valence electrons (Magpie set),
    stoichiometry, valence-orbital fractions, ionic character, atomic-orbital energies and band centre
  - Mulliken electronegativity χ and the number of elements
  - dft_gap_gga and dft_gap_hybrid from data/dft_gaps.csv: the "DFT hint". Empty when the formula is not in
    JARVIS/SNUMAT; LightGBM handles missing values itself, so the model then relies on the formula alone.
Hybrid perovskites are described by their inorganic analogue (MAPbI3 → CsPbI3, chem.inorganic_surrogate): the
organic cation sets no band edge, and its H, C, N would dominate the element statistics (cross-validated on the
12 MA/FA lead and tin halides: MAE 0.46 → 0.28 eV).
Target: log(1 + Eg) of measured gaps of semiconductors and insulators. Metals (Eg = 0) are looked up in
band_gaps.csv, not predicted: a junction needs two semiconductors.
"""
from __future__ import annotations

import re
import warnings
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from materialstack.chem import composition, elements, inorganic_surrogate, mulliken_chi
from materialstack.config import BAND_GAPS, DFT_GAPS, MODEL

LGBM_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                   random_state=0, verbosity=-1)


@lru_cache(maxsize=1)
def _featurizer():
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (AtomicOrbitals, BandCenter, ElementProperty, IonProperty,
                                                  Stoichiometry, ValenceOrbital)
    return MultipleFeaturizer([ElementProperty.from_preset("magpie"), Stoichiometry(), ValenceOrbital(),
                               IonProperty(fast=True), AtomicOrbitals(), BandCenter()])


@lru_cache(maxsize=1)
def dft_table() -> pd.DataFrame:
    return pd.read_csv(DFT_GAPS).set_index("formula")


def featurize(formulas: list[str]) -> pd.DataFrame:
    """One row of numeric features per formula (rows of unreadable formulas are all NaN)."""
    feat = _featurizer()
    labels = feat.feature_labels()
    keys = [inorganic_surrogate(f) for f in formulas]          # what the features describe
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in keys:
            comp = composition(f)
            try:
                rows.append(feat.featurize(comp) if comp is not None else [np.nan] * len(labels))
            except Exception:
                rows.append([np.nan] * len(labels))
    X = pd.DataFrame(rows, columns=labels, index=formulas)
    X = X.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")   # drops text columns (orbital names)
    X.columns = [re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_") for c in X.columns]   # names LightGBM accepts as is
    X["mulliken_chi"] = [mulliken_chi(f) for f in keys]
    X["n_elements"] = [len(elements(f)) for f in keys]
    dft = dft_table()
    X["dft_gap_gga"] = [dft.gap_gga.get(f, np.nan) for f in keys]
    X["dft_gap_hybrid"] = [dft.gap_hybrid.get(f, np.nan) for f in keys]
    return X


def new_model():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**LGBM_PARAMS)


def training_set() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Features X, measured gaps y and the matching rows of band_gaps.csv (semiconductors/insulators only)."""
    table = pd.read_csv(BAND_GAPS)
    table = table[table.gap_ev > 0.001].reset_index(drop=True)
    X = featurize(table.formula.tolist()).reset_index(drop=True)
    return X, table.gap_ev.to_numpy(), table


def element_group(formula: str) -> str:
    """Materials made of the same elements (CsPbI3, Cs4PbI6 …) are kept in one cross-validation fold."""
    return "-".join(elements(formula))


def fit(X: pd.DataFrame, y: np.ndarray):
    return new_model().fit(X, np.log1p(y))


def predict(model, X: pd.DataFrame) -> np.ndarray:
    return np.clip(np.expm1(model.predict(X[model.feature_name_])), 0.0, None)


def save(model) -> None:
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model}, MODEL)


@lru_cache(maxsize=1)
def load():
    return joblib.load(MODEL)["model"] if MODEL.exists() else None
