"""LightGBM models for band gap (Model A) and vacuum band-edge correction (Model B).

Primary stack (thesis-aligned):
  - LGBMClassifier: metal vs non-metal
  - LGBMRegressor: Eg on non-metals (log1p target)
  - LGBMRegressor: delta_CBM = CBM_true - CBM_ButlerGinley; VBM = CBM - Eg

XGBoost is evaluated under the same GroupKFold and stored as a baseline in metrics.json.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import GroupKFold

from materialstack.chem import butler_ginley_edges
from materialstack.clean import (CleanAudit, CleanConfig, DEFAULT_EG_METHODS, clean_edge_frame,
                             clean_eg_frame, load_dft_gap_medians, materials_with_dft_conflict,
                             materials_with_expt_scatter)
from materialstack.config import DB_PATH, MODELS_DIR

log = logging.getLogger("materialstack.models")

# Default: experiment + literature only (HSE/TBmBJ excluded from Model A targets).
EG_TRAIN_METHODS = DEFAULT_EG_METHODS
METAL_GAP_EPS = 1e-3
N_SPLITS = 5

EG_MODEL_PATH = MODELS_DIR / "eg_model.joblib"
EDGE_MODEL_PATH = MODELS_DIR / "edge_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
CLEAN_AUDIT_PATH = MODELS_DIR / "train_clean_audit.json"

# Sample weights by edge source family (literature highest when present).
EDGE_SOURCE_WEIGHT = {
    "literature_csv": 3.0,
    "jarvis_surfacedb": 2.0,
    "castelli_perovskites": 1.0,
}


# --------------------------------------------------------------------------- feature matrix helpers

def _feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"material_id", "gap_ev", "cbm_ev", "vbm_ev", "chi", "cbm_bg", "delta_cbm", "is_metal",
            "method", "source", "family", "elements", "anonymized_formula", "formula_reduced",
            "sample_weight", "held_out", "edge_reference", "group_key"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(dtype=float)


def _group_key(elements_json: str | None, family: str | None) -> str:
    """GroupKFold key: element set string so substituted variants stay together."""
    if isinstance(elements_json, str) and elements_json.strip():
        return elements_json
    return family or "unknown"


# --------------------------------------------------------------------------- training frames

def load_eg_frame(
    con,
    *,
    include_methods: tuple[str, ...] = EG_TRAIN_METHODS,
    clean_config: CleanConfig | None = None,
    audit: CleanAudit | None = None,
) -> tuple[pd.DataFrame, CleanAudit]:
    """One row per material: best-ranked gap among include_methods, joined to composition features.

    Borlido rows are kept with held_out=1 and must be excluded from fitting / CV.
    Label cleaning is non-destructive (SQLite unchanged).
    """
    methods = tuple(include_methods)
    cfg = clean_config or CleanConfig(eg_methods=methods)
    audit = audit or CleanAudit()
    audit.eg_methods = list(methods)
    log.info("Eg label methods (allow-list): %s", ", ".join(methods))

    placeholders = ",".join("?" * len(methods))
    recs = pd.read_sql_query(
        f"""SELECT r.material_id, r.source, r.method, r.method_rank, r.gap_ev, r.extra,
                   m.family, m.elements, m.anonymized_formula, m.formula_reduced, m.mulliken_chi
            FROM records r JOIN materials m USING(material_id)
            WHERE r.method IN ({placeholders}) AND r.gap_ev IS NOT NULL""",
        con, params=methods,
    )
    if recs.empty:
        raise RuntimeError("no Eg training records found")

    # Borlido is a held-out test set: usable only when the material has no other training source.
    is_borlido = recs.source.eq("borlido_expt") | recs.extra.fillna("").str.contains("held_out_test", na=False)
    recs = recs.assign(is_borlido=is_borlido.astype(int))

    src_counts = recs.loc[recs.is_borlido == 0].groupby(["source", "method"]).size().reset_index(name="n")
    log.info("Eg raw records by source/method (excl. borlido rows):\n%s",
             src_counts.sort_values("n", ascending=False).to_string(index=False))

    scatter_ids = materials_with_expt_scatter(
        recs,
        max_spread=cfg.max_expt_spread,
        max_iqr=cfg.max_expt_iqr,
        metal_eps=cfg.metal_eps,
    )
    log.info("Eg scatter candidates (expt disagreement): %d materials", len(scatter_ids))

    rows = []
    for mid, g in recs.groupby("material_id", sort=False):
        non_b = g[g.is_borlido == 0]
        only_holdout = non_b.empty
        use = g if only_holdout else non_b
        best_rank = use.method_rank.min()
        sub = use[use.method_rank == best_rank]
        method = sub.method.mode().iloc[0] if not sub.method.mode().empty else sub.method.iloc[0]
        rows.append({
            "material_id": int(mid),
            "gap_ev": float(sub.gap_ev.median()),
            "method": method,
            "source": sub.source.iloc[0],
            "held_out": int(only_holdout),
            "family": g.family.iloc[0],
            "elements": g.elements.iloc[0],
            "anonymized_formula": g.anonymized_formula.iloc[0],
            "formula_reduced": g.formula_reduced.iloc[0],
            "chi": g.mulliken_chi.iloc[0],
        })
    labels = pd.DataFrame(rows)
    n_before_join = len(labels)
    audit.add_rule(
        "method_allow_list_aggregate",
        n_before=int(recs.material_id.nunique()),
        n_after=n_before_join,
        detail=f"one row per material; best method_rank among {methods}",
        threshold={"methods": list(methods)},
    )

    feat = pd.read_sql_query("SELECT * FROM features_composition", con)
    df = labels.merge(feat, on="material_id", how="inner")
    audit.add_rule(
        "features_composition_inner_join",
        n_before=n_before_join,
        n_after=len(df),
        detail="require Magpie/composition feature row",
    )
    df["is_metal"] = (df.gap_ev <= METAL_GAP_EPS).astype(int)
    df["group_key"] = [_group_key(e, f) for e, f in zip(df.elements, df.family)]
    for m in methods:
        df[f"label_method_{m}"] = (df.method == m).astype(int)

    dft_medians = load_dft_gap_medians(con)
    conflict_ids = materials_with_dft_conflict(
        df, dft_medians, max_conflict=cfg.max_dft_conflict, metal_eps=cfg.metal_eps,
    )
    log.info("Eg DFT-conflict candidates: %d materials", len(conflict_ids))

    feat_cols = _feature_columns(df)
    df, audit = clean_eg_frame(
        df,
        feature_cols=feat_cols,
        config=cfg,
        scatter_ids=scatter_ids,
        conflict_ids=conflict_ids,
        audit=audit,
    )
    df["is_metal"] = (df.gap_ev <= METAL_GAP_EPS).astype(int)

    log.info("Eg frame after clean: %d materials (%d train, %d borlido-only holdout, %d metals)",
             len(df), (df.held_out == 0).sum(), (df.held_out == 1).sum(), int(df.is_metal.sum()))
    return df, audit


def load_edge_frame(
    con,
    *,
    clean_config: CleanConfig | None = None,
    audit: CleanAudit | None = None,
) -> tuple[pd.DataFrame, CleanAudit]:
    """Vacuum-referenced CBM/VBM rows with Butler-Ginley baseline and delta_CBM target."""
    cfg = clean_config or CleanConfig()
    audit = audit or CleanAudit()
    recs = pd.read_sql_query(
        """SELECT r.material_id, r.source, r.method, r.gap_ev, r.cbm_ev, r.vbm_ev, r.edge_reference,
                  m.family, m.elements, m.anonymized_formula, m.formula_reduced, m.mulliken_chi AS chi
           FROM records r JOIN materials m USING(material_id)
           WHERE r.cbm_ev IS NOT NULL AND r.vbm_ev IS NOT NULL
             AND r.edge_reference IN ('vacuum', 'nhe')
             AND r.gap_ev IS NOT NULL AND r.gap_ev > 0
             AND m.mulliken_chi IS NOT NULL""",
        con,
    )
    if recs.empty:
        raise RuntimeError("no vacuum-edge training records found")

    def _bg_row(r):
        cbm_bg, _ = butler_ginley_edges(float(r.chi), float(r.gap_ev))
        return cbm_bg

    recs = recs.copy()
    recs["cbm_bg"] = recs.apply(_bg_row, axis=1)
    recs["delta_cbm"] = recs.cbm_ev - recs.cbm_bg
    recs["sample_weight"] = recs.source.map(lambda s: EDGE_SOURCE_WEIGHT.get(s, 1.0))
    recs["group_key"] = [_group_key(e, f) for e, f in zip(recs.elements, recs.family)]

    rows = []
    for mid, g in recs.groupby("material_id", sort=False):
        wmax = g.sample_weight.max()
        sub = g[g.sample_weight == wmax]
        rows.append({
            "material_id": int(mid),
            "gap_ev": float(sub.gap_ev.median()),
            "cbm_ev": float(sub.cbm_ev.median()),
            "vbm_ev": float(sub.vbm_ev.median()),
            "chi": float(sub.chi.iloc[0]),
            "cbm_bg": float(sub.cbm_bg.median()),
            "delta_cbm": float(sub.delta_cbm.median()),
            "sample_weight": float(wmax),
            "source": sub.source.iloc[0],
            "method": sub.method.iloc[0],
            "family": g.family.iloc[0],
            "elements": g.elements.iloc[0],
            "group_key": g.group_key.iloc[0],
            "formula_reduced": g.formula_reduced.iloc[0],
        })
    labels = pd.DataFrame(rows)
    feat = pd.read_sql_query("SELECT * FROM features_composition", con)
    df = labels.merge(feat, on="material_id", how="inner")
    df["feat_gap_ev"] = df.gap_ev
    df["feat_cbm_bg"] = df.cbm_bg
    log.info("Edge frame before clean: %d materials (sources: %s)", len(df),
             df.source.value_counts().to_dict())
    log.info("Butler-Ginley: CBM_BG = -(χ − Eg/2),  δCBM = CBM_true − CBM_BG,  VBM = CBM − Eg")

    df, audit = clean_edge_frame(df, config=cfg, audit=audit)
    log.info("Edge frame after clean: %d materials (sources: %s)", len(df),
             df.source.value_counts().to_dict() if len(df) else {})
    return df, audit


# --------------------------------------------------------------------------- estimators

def _lgbm_classifier(**kw):
    from lightgbm import LGBMClassifier
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                  colsample_bytree=0.8, random_state=0, verbosity=-1, n_jobs=-1)
    params.update(kw)
    return LGBMClassifier(**params)


def _lgbm_regressor(**kw):
    from lightgbm import LGBMRegressor
    params = dict(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                  colsample_bytree=0.8, random_state=0, verbosity=-1, n_jobs=-1)
    params.update(kw)
    return LGBMRegressor(**params)


def _xgb_classifier(**kw):
    from xgboost import XGBClassifier
    params = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, random_state=0, n_jobs=-1, verbosity=0)
    params.update(kw)
    return XGBClassifier(**params)


def _xgb_regressor(**kw):
    from xgboost import XGBRegressor
    params = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, random_state=0, n_jobs=-1, verbosity=0)
    params.update(kw)
    return XGBRegressor(**params)


class TwoStageGapModel(BaseEstimator, RegressorMixin):
    """Metal classifier + non-metal Eg regressor. Predicts Eg in eV (0 for metals)."""

    def __init__(self, backend: str = "lgbm", use_log1p: bool = True):
        self.backend = backend
        self.use_log1p = use_log1p
        self.feature_names_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.classifier_ = None
        self.regressor_ = None
        self.cv_metrics_: dict[str, Any] = {}

    def _make_clf(self):
        return _lgbm_classifier() if self.backend == "lgbm" else _xgb_classifier()

    def _make_reg(self):
        return _lgbm_regressor() if self.backend == "lgbm" else _xgb_regressor()

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None):
        self.feature_names_ = list(feature_names or [f"f{i}" for i in range(X.shape[1])])
        self.imputer_ = SimpleImputer(strategy="median")
        Xi = self.imputer_.fit_transform(X)
        is_metal = (np.asarray(y) <= METAL_GAP_EPS).astype(int)
        self.classifier_ = self._make_clf()
        self.classifier_.fit(Xi, is_metal)
        mask = is_metal == 0
        self.regressor_ = self._make_reg()
        if mask.sum() == 0:
            raise RuntimeError("no non-metal rows to train Eg regressor")
        yt = np.log1p(np.asarray(y)[mask]) if self.use_log1p else np.asarray(y)[mask]
        self.regressor_.fit(Xi[mask], yt)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xi = self.imputer_.transform(X)
        metal = self.classifier_.predict(Xi).astype(int)
        out = np.zeros(len(Xi), dtype=float)
        non = metal == 0
        if non.any():
            pred = self.regressor_.predict(Xi[non])
            out[non] = np.expm1(pred) if self.use_log1p else pred
            out[non] = np.clip(out[non], 0.0, None)
        return out

    def predict_proba_metal(self, X: np.ndarray) -> np.ndarray:
        Xi = self.imputer_.transform(X)
        if hasattr(self.classifier_, "predict_proba"):
            return self.classifier_.predict_proba(Xi)[:, 1]
        return self.classifier_.predict(Xi).astype(float)

    def feature_importance(self, top_n: int = 30) -> list[dict[str, Any]]:
        if self.regressor_ is None or not hasattr(self.regressor_, "feature_importances_"):
            return []
        imp = self.regressor_.feature_importances_
        order = np.argsort(imp)[::-1][:top_n]
        return [{"feature": self.feature_names_[i], "importance": float(imp[i])} for i in order]


class EdgeCorrectionModel(BaseEstimator, RegressorMixin):
    """Predicts delta_CBM; reconstructs CBM = cbm_bg + delta, VBM = CBM - Eg."""

    def __init__(self, backend: str = "lgbm"):
        self.backend = backend
        self.feature_names_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.regressor_ = None
        self.cv_metrics_: dict[str, Any] = {}

    def _make_reg(self):
        return _lgbm_regressor() if self.backend == "lgbm" else _xgb_regressor()

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None,
            feature_names: list[str] | None = None):
        self.feature_names_ = list(feature_names or [f"f{i}" for i in range(X.shape[1])])
        self.imputer_ = SimpleImputer(strategy="median")
        Xi = self.imputer_.fit_transform(X)
        self.regressor_ = self._make_reg()
        fit_kw = {}
        if sample_weight is not None:
            fit_kw["sample_weight"] = sample_weight
        self.regressor_.fit(Xi, y, **fit_kw)
        return self

    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        return self.regressor_.predict(self.imputer_.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_delta(X)

    def predict_edges(self, X: np.ndarray, gap: np.ndarray, cbm_bg: np.ndarray) -> dict[str, np.ndarray]:
        delta = self.predict_delta(X)
        cbm = np.asarray(cbm_bg) + delta
        vbm = cbm - np.asarray(gap)
        return {"delta_cbm": delta, "cbm": cbm, "vbm": vbm, "gap": np.asarray(gap)}

    def feature_importance(self, top_n: int = 30) -> list[dict[str, Any]]:
        if self.regressor_ is None or not hasattr(self.regressor_, "feature_importances_"):
            return []
        imp = self.regressor_.feature_importances_
        order = np.argsort(imp)[::-1][:top_n]
        return [{"feature": self.feature_names_[i], "importance": float(imp[i])} for i in order]


# --------------------------------------------------------------------------- CV

def _grouped_cv_gap(model: TwoStageGapModel, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_splits: int = N_SPLITS) -> dict[str, Any]:
    gkf = GroupKFold(n_splits=n_splits)
    metal_true = (y <= METAL_GAP_EPS).astype(int)
    metal_pred = np.zeros_like(metal_true)
    gap_pred = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in gkf.split(X, y, groups):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx], feature_names=model.feature_names_ or None)
        metal_pred[test_idx] = m.classifier_.predict(m.imputer_.transform(X[test_idx])).astype(int)
        gap_pred[test_idx] = m.predict(X[test_idx])

    non = metal_true == 0
    out = {
        "metal_accuracy": float(accuracy_score(metal_true, metal_pred)),
        "metal_f1": float(f1_score(metal_true, metal_pred)),
        "gap_mae_nonmetal": float(mean_absolute_error(y[non], gap_pred[non])) if non.any() else None,
        "gap_rmse_nonmetal": float(mean_squared_error(y[non], gap_pred[non]) ** 0.5) if non.any() else None,
        "gap_r2_nonmetal": float(r2_score(y[non], gap_pred[non])) if non.any() else None,
        "gap_mae_all": float(mean_absolute_error(y, gap_pred)),
        "n_train_materials": int(len(y)),
        "n_metals": int(metal_true.sum()),
        "n_nonmetals": int((~metal_true.astype(bool)).sum()),
        "n_splits": n_splits,
    }
    return out, gap_pred, metal_pred


def _grouped_cv_edge(model: EdgeCorrectionModel, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                     sample_weight: np.ndarray, gap: np.ndarray, cbm_bg: np.ndarray, cbm_true: np.ndarray,
                     vbm_true: np.ndarray, n_splits: int = N_SPLITS) -> dict[str, Any]:
    gkf = GroupKFold(n_splits=min(n_splits, max(2, len(np.unique(groups)))))
    # Ensure enough groups
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups) if n_groups >= 2 else 2
    if n_groups < 2:
        # fall back: no CV, just fit metrics on train
        model.fit(X, y, sample_weight=sample_weight, feature_names=model.feature_names_ or None)
        pred = model.predict_edges(X, gap, cbm_bg)
        return {
            "delta_mae": float(mean_absolute_error(y, pred["delta_cbm"])),
            "cbm_mae": float(mean_absolute_error(cbm_true, pred["cbm"])),
            "vbm_mae": float(mean_absolute_error(vbm_true, pred["vbm"])),
            "n_train_materials": int(len(y)),
            "n_splits": 0,
            "note": "fewer than 2 groups; metrics are in-sample",
        }, pred["delta_cbm"]

    gkf = GroupKFold(n_splits=splits)
    delta_pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups):
        m = clone(model)
        sw = sample_weight[train_idx] if sample_weight is not None else None
        m.fit(X[train_idx], y[train_idx], sample_weight=sw, feature_names=model.feature_names_ or None)
        delta_pred[test_idx] = m.predict_delta(X[test_idx])
    cbm_pred = cbm_bg + delta_pred
    vbm_pred = cbm_pred - gap
    return {
        "delta_mae": float(mean_absolute_error(y, delta_pred)),
        "cbm_mae": float(mean_absolute_error(cbm_true, cbm_pred)),
        "vbm_mae": float(mean_absolute_error(vbm_true, vbm_pred)),
        "delta_rmse": float(mean_squared_error(y, delta_pred) ** 0.5),
        "cbm_r2": float(r2_score(cbm_true, cbm_pred)),
        "n_train_materials": int(len(y)),
        "n_splits": splits,
    }, delta_pred


# --------------------------------------------------------------------------- structure-aware Eg model

def _gap_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    if len(y) == 0:
        return {"n": 0, "n_nonmetals": 0, "mae_nonmetal": None, "rmse_nonmetal": None, "r2_nonmetal": None,
                "mae_all": None, "within_0p5_all": None, "within_0p5_nonmetal": None, "metal_accuracy": None}
    non = y > METAL_GAP_EPS
    metal_true = (~non).astype(int)
    metal_pred = (pred <= METAL_GAP_EPS).astype(int)
    err = np.abs(pred - y)
    return {
        "n": int(len(y)),
        "n_nonmetals": int(non.sum()),
        "mae_nonmetal": float(err[non].mean()) if non.any() else None,
        "rmse_nonmetal": float(np.sqrt(np.mean(err[non] ** 2))) if non.any() else None,
        "r2_nonmetal": float(r2_score(y[non], pred[non])) if non.sum() > 1 else None,
        "mae_all": float(err.mean()),
        "within_0p5_all": float(np.mean(err <= 0.5)),
        "within_0p5_nonmetal": float(np.mean(err[non] <= 0.5)) if non.any() else None,
        "metal_accuracy": float(accuracy_score(metal_true, metal_pred)),
    }


def _log_gap_metrics(label: str, m: dict[str, Any]) -> None:
    if not m["n"]:
        log.info("  %-34s n=0", label)
        return
    log.info("  %-34s n=%5d  MAE(non-metal)=%.3f eV  R²=%.3f  metal ACC=%.3f  ≤0.5 eV=%.1f%%",
             label, m["n"], m["mae_nonmetal"] or np.nan, m["r2_nonmetal"] or np.nan,
             m["metal_accuracy"], 100 * m["within_0p5_all"])


def _train_structure_hybrid(con, train_eg: pd.DataFrame, hold_eg: pd.DataFrame, comp_cols: list[str],
                            comp_oof: np.ndarray, comp_model: TwoStageGapModel, *, n_splits: int,
                            models_dir: Path) -> dict[str, Any] | None:
    """Composition + crystal-structure + DFT-proxy + stored-DFT-gap Eg model.

    Trained on every material: structure / DFT columns are NaN when the DB has none, which
    LightGBM routes natively, so no training data is lost. Evaluated on exactly the folds of the
    composition-only model, with a feature ablation and both polymorph policies.
    """
    from materialstack.structure import (DFT_COLUMNS, PROXY_COLUMNS, _table_exists, candidate_structures,
                                         dft_gap_features, select_structures, structure_feature_columns)

    if not _table_exists(con, "features_structure"):
        log.warning("features_structure missing — skipping structure-aware model "
                    "(run `featurize-structures` and `train-proxies`)")
        return None
    has_proxy = _table_exists(con, "structure_proxy")
    if not has_proxy:
        log.warning("structure_proxy missing — hybrid model will use structure features without DFT proxies")

    log.info("=== Structure-aware Eg model ===")
    all_ids = pd.concat([train_eg.material_id, hold_eg.material_id]).astype(int).tolist()
    cands = candidate_structures(con, all_ids)
    gs = select_structures(cands, "ground_state")
    mean_sel = select_structures(cands, "mean")
    sids = sorted(set(gs.structure_id) | set(mean_sel.structure_id)) or [-1]
    id_list = ",".join(map(str, sids))
    sf_cols = structure_feature_columns(con)
    struct_tbl = pd.read_sql_query(
        f"SELECT structure_id, {', '.join(sf_cols)} FROM features_structure WHERE structure_id IN ({id_list})", con)
    px_cols: list[str] = []
    if has_proxy:
        prox = pd.read_sql_query(f"SELECT * FROM structure_proxy WHERE structure_id IN ({id_list})", con)
        px_cols = [c for c in PROXY_COLUMNS if c in prox.columns]
        struct_tbl = struct_tbl.merge(prox[["structure_id"] + px_cols], on="structure_id", how="left")

    def _attach(frame: pd.DataFrame, sel: pd.DataFrame, how: str) -> pd.DataFrame:
        f = frame.assign(_row=np.arange(len(frame)))
        f = f.merge(sel[["material_id", "structure_id"]], on="material_id", how=how)
        f = f.merge(struct_tbl, on="structure_id", how="left")
        dft = dft_gap_features(con, f[["material_id", "structure_id"]])
        for c in DFT_COLUMNS:
            f[c] = dft[c].to_numpy()
        return f

    full = _attach(train_eg, gs, "left").sort_values("_row").reset_index(drop=True)
    poly = _attach(train_eg, mean_sel, "inner")
    has_struct = full.structure_id.notna().to_numpy()
    has_dft = full[DFT_COLUMNS].notna().any(axis=1).to_numpy()
    n_poly = poly.groupby("material_id").size()
    log.info("Coverage of %d training materials: structure %d (%.1f%%), stored DFT gap %d (%.1f%%), "
             "neither %d", len(full), has_struct.sum(), 100 * has_struct.mean(), has_dft.sum(),
             100 * has_dft.mean(), int((~has_struct & ~has_dft).sum()))
    log.info("Polymorphs: %d materials have >1 distinct bulk polymorph (mean %.2f each)",
             int((n_poly > 1).sum()), n_poly.mean() if len(n_poly) else 0)
    log.info("Polymorph policy ground_state: lowest DFT energy among the formula's polymorphs (JARVIS) → "
             "source priority → smallest cell")
    log.info("Polymorph policy mean: one structure per space group within 0.05 eV/atom of the lowest, "
             "Êg averaged")
    log.info("Stored DFT gaps: value computed on the chosen structure if present, else material median")

    y = full.gap_ev.to_numpy(dtype=float)
    groups = full.group_key.to_numpy()
    # Same rows, order and groups as the composition-only CV → identical GroupKFold folds.
    folds = list(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups))
    mid_to_pos = pd.Series(np.arange(len(full)), index=full.material_id.to_numpy())
    known = has_struct | has_dft

    def _cv(cols: list[str], with_mean: bool = False) -> tuple[np.ndarray, np.ndarray]:
        X = _matrix(full, cols)
        pred = np.zeros_like(y)
        pred_mean = np.full_like(y, np.nan)
        for tr, te in folds:
            m = TwoStageGapModel(backend="lgbm", use_log1p=True).fit(X[tr], y[tr], feature_names=cols)
            pred[te] = m.predict(X[te])
            if with_mean:
                p = poly[poly.material_id.isin(full.material_id.iloc[te])]
                if len(p):
                    avg = pd.Series(m.predict(_matrix(p, cols)), index=p.material_id.to_numpy()).groupby(level=0).mean()
                    pred_mean[mid_to_pos.loc[avg.index].to_numpy()] = avg.to_numpy()
        return pred, np.where(np.isnan(pred_mean), pred, pred_mean)

    def _report(label: str, pred: np.ndarray) -> dict[str, Any]:
        out = {
            "all": _gap_metrics(y, pred),
            "with_structure": _gap_metrics(y[has_struct], pred[has_struct]),
            "without_structure": _gap_metrics(y[~has_struct], pred[~has_struct]),
            "no_structure_no_dft": _gap_metrics(y[~known], pred[~known]),
        }
        _log_gap_metrics(label, out["all"])
        _log_gap_metrics("    └ materials with a structure", out["with_structure"])
        _log_gap_metrics("    └ no structure, no stored DFT", out["no_structure_no_dft"])
        return out

    feature_sets: list[tuple[str, list[str]]] = [("composition + structure", comp_cols + sf_cols)]
    if px_cols:
        feature_sets.append(("composition + structure + DFT proxies", comp_cols + sf_cols + px_cols))
    feature_sets.append(("+ stored DFT gaps (final)", comp_cols + sf_cols + px_cols + DFT_COLUMNS))
    final_cols = feature_sets[-1][1]
    log.info("Features (final): %d composition + %d structure + %d DFT-proxy + %d stored-DFT = %d",
             len(comp_cols), len(sf_cols), len(px_cols), len(DFT_COLUMNS), len(final_cols))

    log.info("Ablation on identical GroupKFold folds (n=%d):", len(y))
    ablation = {"composition only": _report("composition only", comp_oof)}
    pred_gs = pred_mean = None
    for label, cols in feature_sets:
        is_final = cols is final_cols
        pred, pmean = _cv(cols, with_mean=is_final)
        ablation[label] = _report(label, pred)
        if is_final:
            pred_gs, pred_mean = pred, pmean
    m_mean = _report("final, mean over polymorphs", pred_mean)

    gs_mae = ablation[feature_sets[-1][0]]["with_structure"]["mae_nonmetal"] or 9
    mean_mae = m_mean["with_structure"]["mae_nonmetal"] or 9
    recommended = "mean" if mean_mae < gs_mae - 0.005 else "ground_state"
    comp_unknown = ablation["composition only"]["no_structure_no_dft"]["mae_nonmetal"]
    hyb_unknown = ablation[feature_sets[-1][0]]["no_structure_no_dft"]["mae_nonmetal"]
    fallback = ("hybrid" if comp_unknown is None or (hyb_unknown is not None and hyb_unknown <= comp_unknown)
                else "composition")
    log.info("Recommended polymorph policy: %s (with-structure MAE ground state %.3f vs mean %.3f)",
             recommended, gs_mae, mean_mae)
    log.info("Materials with no structure and no stored DFT gap → %s model", fallback)

    sys_pred = np.where(known | (fallback == "hybrid"), pred_mean if recommended == "mean" else pred_gs, comp_oof)
    system = {"composition_only": _gap_metrics(y, comp_oof), "structure_aware": _gap_metrics(y, sys_pred)}
    log.info("System as deployed (structure-aware model + fallback rule):")
    _log_gap_metrics("composition-only (previous)", system["composition_only"])
    _log_gap_metrics("structure-aware (new)", system["structure_aware"])

    final = TwoStageGapModel(backend="lgbm", use_log1p=True).fit(_matrix(full, final_cols), y,
                                                                 feature_names=final_cols)
    out: dict[str, Any] = {
        "n_train_total": int(len(full)),
        "n_train_with_structure": int(has_struct.sum()),
        "n_train_with_dft": int(has_dft.sum()),
        "coverage": float(has_struct.mean()),
        "coverage_dft": float(has_dft.mean()),
        "n_with_multiple_polymorphs": int((n_poly > 1).sum()),
        "n_features": {"composition": len(comp_cols), "structure": len(sf_cols), "dft_proxy": len(px_cols),
                       "stored_dft": len(DFT_COLUMNS)},
        "ablation": ablation,
        "final_mean_polymorph": m_mean,
        "system": system,
        "recommended_policy": recommended,
        "fallback_without_structure": fallback,
        "top_features": final.feature_importance(25),
        "n_splits": n_splits,
    }

    if not hold_eg.empty:
        hfull = _attach(hold_eg, gs, "left").sort_values("_row").reset_index(drop=True)
        yh = hfull.gap_ev.to_numpy(dtype=float)
        ph_comp = comp_model.predict(_matrix(hfull, comp_cols))
        hknown = hfull.structure_id.notna().to_numpy() | hfull[DFT_COLUMNS].notna().any(axis=1).to_numpy()
        ph_sys = np.where(hknown | (fallback == "hybrid"), final.predict(_matrix(hfull, final_cols)), ph_comp)
        out["borlido_holdout"] = {"composition_only": _gap_metrics(yh, ph_comp),
                                  "structure_aware": _gap_metrics(yh, ph_sys),
                                  "n_known": int(hknown.sum())}
        log.info("Borlido holdout (never trained on):")
        _log_gap_metrics("composition-only", out["borlido_holdout"]["composition_only"])
        _log_gap_metrics("structure-aware", out["borlido_holdout"]["structure_aware"])

    joblib.dump({"model": final, "feature_columns": final_cols, "composition_columns": comp_cols,
                 "structure_columns": sf_cols, "proxy_columns": px_cols, "dft_columns": list(DFT_COLUMNS),
                 "kind": "eg_two_stage_structure", "recommended_policy": recommended,
                 "fallback_without_structure": fallback},
                models_dir / "eg_structure_model.joblib")
    return out


# --------------------------------------------------------------------------- train orchestration

@dataclass
class TrainResult:
    eg_path: str
    edge_path: str
    metrics_path: str
    metrics: dict[str, Any]


def train(*, db_path: Path = DB_PATH, models_dir: Path = MODELS_DIR, n_splits: int = N_SPLITS,
          run_xgb_baseline: bool = True,
          eg_methods: tuple[str, ...] = EG_TRAIN_METHODS,
          clean_config: CleanConfig | None = None,
          use_structure: bool = True) -> TrainResult:
    from materialstack.db import connect

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    t0 = time.time()

    cfg = clean_config or CleanConfig(eg_methods=tuple(eg_methods))
    cfg.eg_methods = tuple(eg_methods)
    audit = CleanAudit()

    log.info("=== MaterialStack train ===")
    log.info("Primary backend: LightGBM | XGBoost baseline: %s | GroupKFold n_splits=%d",
             run_xgb_baseline, n_splits)
    log.info("DB: %s", db_path)
    log.info("Cleaning: SQLite unchanged; filters apply to training frames only")
    log.info("Model A target: two-stage metal clf + log1p(Eg) regressor on non-metals")
    log.info("  metal if Eg ≤ %.3g eV → Êg=0; else train t=log(1+Eg), predict Êg=exp(t̂)−1",
             METAL_GAP_EPS)

    eg_df, audit = load_eg_frame(con, include_methods=tuple(eg_methods), clean_config=cfg, audit=audit)
    edge_df, audit = load_edge_frame(con, clean_config=cfg, audit=audit)
    audit.write(models_dir / "train_clean_audit.json")

    # ---- Model A
    train_eg = eg_df[eg_df.held_out == 0].copy()
    hold_eg = eg_df[eg_df.held_out == 1].copy()
    if train_eg.empty:
        raise RuntimeError("no Eg training materials left after cleaning")
    feat_cols = _feature_columns(train_eg)
    X = _matrix(train_eg, feat_cols)
    y = train_eg.gap_ev.to_numpy(dtype=float)
    groups = train_eg.group_key.to_numpy()

    lgbm_gap = TwoStageGapModel(backend="lgbm", use_log1p=True)
    lgbm_gap.feature_names_ = feat_cols
    log.info("CV LightGBM Eg model (n_train=%d, n_features=%d, n_groups=%d) ...",
             len(train_eg), len(feat_cols), pd.Series(groups).nunique())
    lgbm_metrics, comp_oof, _ = _grouped_cv_gap(lgbm_gap, X, y, groups, n_splits=n_splits)
    lgbm_gap.fit(X, y, feature_names=feat_cols)
    lgbm_gap.cv_metrics_ = lgbm_metrics
    lgbm_metrics["top_features"] = lgbm_gap.feature_importance(25)
    lgbm_metrics["backend"] = "lightgbm"

    metrics: dict[str, Any] = {
        "primary_backend": "lightgbm",
        "decision": ("LightGBM chosen as primary (thesis-aligned; MAE within ~0.02 eV of XGBoost). "
                     "XGBoost retained as baseline comparison only."),
        "eg_model": {"lightgbm": lgbm_metrics},
        "feature_columns": feat_cols,
        "eg_train_methods": list(eg_methods),
        "n_eg_train": int(len(train_eg)),
        "n_eg_holdout_borlido_only": int(len(hold_eg)),
        "clean_audit_path": str(models_dir / "train_clean_audit.json"),
        "clean_summary": {
            "n_eg_before": audit.n_eg_before,
            "n_eg_after": audit.n_eg_after,
            "n_edge_before": audit.n_edge_before,
            "n_edge_after": audit.n_edge_after,
            "rules": audit.rules,
        },
    }

    if run_xgb_baseline:
        log.info("CV XGBoost Eg baseline ...")
        xgb_gap = TwoStageGapModel(backend="xgb", use_log1p=True)
        xgb_gap.feature_names_ = feat_cols
        xgb_metrics, _, _ = _grouped_cv_gap(xgb_gap, X, y, groups, n_splits=n_splits)
        xgb_metrics["backend"] = "xgboost"
        metrics["eg_model"]["xgboost_baseline"] = xgb_metrics

    # Borlido hold-out evaluation with primary model
    if not hold_eg.empty:
        Xh = _matrix(hold_eg, feat_cols)
        yh = hold_eg.gap_ev.to_numpy(dtype=float)
        ph = lgbm_gap.predict(Xh)
        non = yh > METAL_GAP_EPS
        metrics["eg_model"]["borlido_holdout"] = {
            "n": int(len(hold_eg)),
            "mae_all": float(mean_absolute_error(yh, ph)),
            "mae_nonmetal": float(mean_absolute_error(yh[non], ph[non])) if non.any() else None,
            "note": "materials present only in borlido_expt; never used in training",
        }

    joblib.dump({"model": lgbm_gap, "feature_columns": feat_cols, "kind": "eg_two_stage",
                 "eg_train_methods": list(eg_methods)},
                models_dir / "eg_model.joblib")

    if use_structure:
        hybrid = _train_structure_hybrid(con, train_eg, hold_eg, feat_cols, comp_oof, lgbm_gap,
                                         n_splits=n_splits, models_dir=models_dir)
        if hybrid is not None:
            metrics["eg_model"]["structure_hybrid"] = hybrid

    # ---- Model B
    if edge_df.empty:
        raise RuntimeError("no edge training materials left after cleaning")
    edge_feat_cols = _feature_columns(edge_df)
    for extra in ("feat_gap_ev", "feat_cbm_bg"):
        if extra in edge_df.columns and extra not in edge_feat_cols:
            edge_feat_cols.append(extra)
    Xe = _matrix(edge_df, edge_feat_cols)
    ye = edge_df.delta_cbm.to_numpy(dtype=float)
    ge = edge_df.group_key.to_numpy()
    we = edge_df.sample_weight.to_numpy(dtype=float)
    gap_e = edge_df.gap_ev.to_numpy(dtype=float)
    cbm_bg = edge_df.cbm_bg.to_numpy(dtype=float)
    cbm_t = edge_df.cbm_ev.to_numpy(dtype=float)
    vbm_t = edge_df.vbm_ev.to_numpy(dtype=float)

    lgbm_edge = EdgeCorrectionModel(backend="lgbm")
    lgbm_edge.feature_names_ = edge_feat_cols
    log.info("CV LightGBM edge model (n=%d) ...", len(edge_df))
    edge_metrics, _ = _grouped_cv_edge(lgbm_edge, Xe, ye, ge, we, gap_e, cbm_bg, cbm_t, vbm_t, n_splits=n_splits)
    lgbm_edge.fit(Xe, ye, sample_weight=we, feature_names=edge_feat_cols)
    lgbm_edge.cv_metrics_ = edge_metrics
    edge_metrics["top_features"] = lgbm_edge.feature_importance(25)
    edge_metrics["backend"] = "lightgbm"
    edge_metrics["sources"] = edge_df.source.value_counts().to_dict()
    metrics["edge_model"] = {"lightgbm": edge_metrics}

    if run_xgb_baseline:
        log.info("CV XGBoost edge baseline ...")
        xgb_edge = EdgeCorrectionModel(backend="xgb")
        xgb_edge.feature_names_ = edge_feat_cols
        xgb_edge_metrics, _ = _grouped_cv_edge(xgb_edge, Xe, ye, ge, we, gap_e, cbm_bg, cbm_t, vbm_t,
                                               n_splits=n_splits)
        xgb_edge_metrics["backend"] = "xgboost"
        metrics["edge_model"]["xgboost_baseline"] = xgb_edge_metrics

    joblib.dump({"model": lgbm_edge, "feature_columns": edge_feat_cols, "kind": "edge_delta_cbm"},
                models_dir / "edge_model.joblib")

    metrics["seconds"] = round(time.time() - t0, 1)
    metrics["paths"] = {
        "eg_model": str(models_dir / "eg_model.joblib"),
        "eg_structure_model": str(models_dir / "eg_structure_model.joblib"),
        "edge_model": str(models_dir / "edge_model.joblib"),
        "metrics": str(models_dir / "metrics.json"),
        "clean_audit": str(models_dir / "train_clean_audit.json"),
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    con.close()
    log.info("Training done in %.1fs — Eg MAE(nonmetal)=%.3f eV, edge CBM MAE=%.3f eV",
             metrics["seconds"], lgbm_metrics.get("gap_mae_nonmetal") or float("nan"),
             edge_metrics.get("cbm_mae") or float("nan"))
    log.info("Note: junctions within ~0.3 eV of a Type boundary remain uncertain at this MAE.")
    return TrainResult(eg_path=str(models_dir / "eg_model.joblib"),
                       edge_path=str(models_dir / "edge_model.joblib"),
                       metrics_path=str(models_dir / "metrics.json"),
                       metrics=metrics)


# --------------------------------------------------------------------------- inference helpers

def load_eg_model(path: Path = EG_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def load_edge_model(path: Path = EDGE_MODEL_PATH) -> dict[str, Any]:
    return joblib.load(path)


def predict_eg(bundle: dict[str, Any], feature_row: pd.Series | dict) -> dict[str, float]:
    model: TwoStageGapModel = bundle["model"]
    cols = bundle["feature_columns"]
    row = pd.Series(feature_row)
    # default label_method_experiment=1 for inference if columns present
    for c in cols:
        if c.startswith("label_method_") and c not in row:
            row[c] = 1.0 if c == "label_method_experiment" else 0.0
    def _f(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    X = np.array([[_f(row.get(c, np.nan)) for c in cols]], dtype=float)
    eg = float(model.predict(X)[0])
    p_metal = float(model.predict_proba_metal(X)[0])
    return {"gap_ev": eg, "p_metal": p_metal, "source": "model"}


def predict_edges(bundle: dict[str, Any], feature_row: pd.Series | dict, gap: float,
                  chi: float) -> dict[str, float]:
    model: EdgeCorrectionModel = bundle["model"]
    cols = bundle["feature_columns"]
    row = pd.Series(feature_row).copy()
    cbm_bg, vbm_bg = butler_ginley_edges(float(chi), float(gap))
    row["feat_gap_ev"] = gap
    row["feat_cbm_bg"] = cbm_bg

    def _f(v):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    X = np.array([[_f(row.get(c, np.nan)) for c in cols]], dtype=float)
    out = model.predict_edges(X, np.array([gap]), np.array([cbm_bg]))
    return {
        "gap_ev": float(gap),
        "chi": float(chi),
        "cbm_bg": float(cbm_bg),
        "vbm_bg": float(vbm_bg),
        "delta_cbm": float(out["delta_cbm"][0]),
        "cbm_ev": float(out["cbm"][0]),
        "vbm_ev": float(out["vbm"][0]),
        "source": "model",
    }
