"""Non-destructive training-label cleaning for Model A (Eg) and Model B (edges).

SQLite is never mutated. Filters apply only when building training frames.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("materialstack.clean")

# Default Model A: experimental / literature only (drops HSE/TBmBJ from supervised target).
DEFAULT_EG_METHODS = ("experiment", "literature")
DFT_CONFLICT_METHODS = ("HSE06", "TBmBJ")


@dataclass
class CleanConfig:
    eg_methods: tuple[str, ...] = DEFAULT_EG_METHODS
    max_gap: float = 12.0
    max_expt_spread: float = 1.0
    max_expt_iqr: float = 0.5
    max_dft_conflict: float = 2.0
    max_nan_frac: float = 0.5
    metal_eps: float = 1e-3
    edge_gap_consistency: float = 0.35
    max_abs_delta_cbm: float = 5.0


@dataclass
class CleanAudit:
    """Drop counts and rule metadata written to models/train_clean_audit.json."""
    rules: list[dict[str, Any]] = field(default_factory=list)
    eg_methods: list[str] = field(default_factory=list)
    n_eg_before: int = 0
    n_eg_after: int = 0
    n_edge_before: int = 0
    n_edge_after: int = 0
    note: str = ("Filters apply to training frames only; data/materials_db.sqlite is not modified.")

    def add_rule(self, name: str, *, n_before: int, n_after: int, detail: str = "",
                 threshold: Any = None) -> None:
        dropped = n_before - n_after
        entry = {
            "rule": name,
            "n_before": n_before,
            "n_after": n_after,
            "n_dropped": dropped,
            "detail": detail,
            "threshold": threshold,
        }
        self.rules.append(entry)
        log.info("clean[%s]: %d → %d (dropped %d)%s",
                 name, n_before, n_after, dropped,
                 f" | {detail}" if detail else "")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        log.info("Wrote clean audit: %s", path)


def parse_eg_methods(s: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in s.split(",") if p.strip())
    if not parts:
        raise ValueError("eg-methods must be a non-empty comma-separated list")
    return parts


def materials_with_expt_scatter(
    recs: pd.DataFrame,
    *,
    max_spread: float,
    max_iqr: float,
    metal_eps: float,
    expt_methods: tuple[str, ...] = ("experiment", "literature"),
) -> set[int]:
    """Material IDs whose experiment/literature gaps disagree too much."""
    bad: set[int] = set()
    mask = recs.method.isin(expt_methods)
    if "is_borlido" in recs.columns:
        mask = mask & (recs.is_borlido == 0)
    expt = recs.loc[mask]
    if expt.empty:
        return bad
    for mid, g in expt.groupby("material_id", sort=False):
        vals = g.gap_ev.to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        if np.all(vals <= metal_eps):
            continue
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        q75, q25 = np.nanpercentile(vals, [75, 25])
        iqr = float(q75 - q25)
        if spread > max_spread or iqr > max_iqr:
            bad.add(int(mid))
    return bad


def materials_with_dft_conflict(
    labels: pd.DataFrame,
    dft_recs: pd.DataFrame,
    *,
    max_conflict: float,
    metal_eps: float,
    expt_methods: tuple[str, ...] = ("experiment", "literature"),
) -> set[int]:
    """Drop experiment-labeled materials whose nearest HSE/TBmBJ gap differs by > max_conflict.

    Uses min over available DFT methods of |Eg_exp − median(Eg_method)|.
    """
    if dft_recs.empty or labels.empty:
        return set()
    med = (
        dft_recs.groupby(["material_id", "method"], sort=False)["gap_ev"]
        .median()
        .reset_index()
    )
    by_mid: dict[int, list[float]] = {}
    for _, r in med.iterrows():
        by_mid.setdefault(int(r.material_id), []).append(float(r.gap_ev))

    bad: set[int] = set()
    train = labels[(labels.held_out == 0) & labels.method.isin(expt_methods)]
    for mid, row in train.set_index("material_id").iterrows():
        eg = float(row.gap_ev)
        if eg <= metal_eps:
            continue
        dft_vals = by_mid.get(int(mid))
        if not dft_vals:
            continue
        nearest = min(abs(eg - d) for d in dft_vals)
        if nearest > max_conflict:
            bad.add(int(mid))
    return bad


def load_dft_gap_medians(con) -> pd.DataFrame:
    placeholders = ",".join("?" * len(DFT_CONFLICT_METHODS))
    return pd.read_sql_query(
        f"""SELECT material_id, method, gap_ev FROM records
            WHERE method IN ({placeholders}) AND gap_ev IS NOT NULL""",
        con, params=DFT_CONFLICT_METHODS,
    )


def clean_eg_frame(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    config: CleanConfig,
    scatter_ids: set[int] | None = None,
    conflict_ids: set[int] | None = None,
    audit: CleanAudit | None = None,
) -> tuple[pd.DataFrame, CleanAudit]:
    """Apply physical / conflict / feature filters. Does not re-filter by method (caller sets methods)."""
    audit = audit or CleanAudit()
    audit.eg_methods = list(config.eg_methods)
    audit.n_eg_before = int(len(df))
    out = df.copy()

    # 1) Within-material experiment scatter (IDs precomputed from raw records).
    if scatter_ids:
        n0 = len(out)
        # Keep borlido holdout even if scatter flagged.
        mask = out.material_id.isin(scatter_ids) & (out.held_out == 0)
        out = out.loc[~mask].copy()
        audit.add_rule(
            "expt_label_scatter",
            n_before=n0, n_after=len(out),
            detail=f"drop train materials with IQR>{config.max_expt_iqr} eV or "
                   f"max-min>{config.max_expt_spread} eV among experiment/literature",
            threshold={"max_iqr": config.max_expt_iqr, "max_spread": config.max_expt_spread},
        )

    # 2) Physical gap bounds (metals Eg≈0 kept; non-metals ≤ max_gap).
    n0 = len(out)
    metal = out.gap_ev <= config.metal_eps
    ok = metal | ((out.gap_ev > config.metal_eps) & (out.gap_ev <= config.max_gap) & (out.gap_ev >= 0))
    out = out.loc[ok].copy()
    audit.add_rule(
        "physical_gap_bounds",
        n_before=n0, n_after=len(out),
        detail=f"keep metals (Eg≤{config.metal_eps}) or {config.metal_eps}<Eg≤{config.max_gap} eV",
        threshold={"metal_eps": config.metal_eps, "max_gap": config.max_gap},
    )

    # 3) DFT conflict vs experiment.
    if conflict_ids:
        n0 = len(out)
        mask = out.material_id.isin(conflict_ids) & (out.held_out == 0)
        out = out.loc[~mask].copy()
        audit.add_rule(
            "dft_conflict",
            n_before=n0, n_after=len(out),
            detail=f"drop experiment materials with |Eg_exp−Eg_DFT|>{config.max_dft_conflict} eV "
                   f"(DFT ∈ {DFT_CONFLICT_METHODS})",
            threshold={"max_dft_conflict": config.max_dft_conflict},
        )

    # 4) Feature completeness among numeric Magpie-like columns.
    use_cols = [c for c in feature_cols if c in out.columns]
    if use_cols:
        n0 = len(out)
        nan_frac = out[use_cols].isna().mean(axis=1)
        out = out.loc[nan_frac <= config.max_nan_frac].copy()
        audit.add_rule(
            "feature_nan_frac",
            n_before=n0, n_after=len(out),
            detail=f"drop rows with >{config.max_nan_frac:.0%} NaN in feature columns",
            threshold={"max_nan_frac": config.max_nan_frac, "n_feature_cols": len(use_cols)},
        )

    audit.n_eg_after = int(len(out))
    return out, audit


def clean_edge_frame(
    df: pd.DataFrame,
    *,
    config: CleanConfig,
    audit: CleanAudit | None = None,
) -> tuple[pd.DataFrame, CleanAudit]:
    audit = audit or CleanAudit()
    audit.n_edge_before = int(len(df))
    out = df.copy()

    n0 = len(out)
    finite = np.isfinite(out.gap_ev) & np.isfinite(out.cbm_ev) & np.isfinite(out.vbm_ev)
    out = out.loc[finite & (out.gap_ev > 0)].copy()
    audit.add_rule(
        "edge_finite_positive_gap",
        n_before=n0, n_after=len(out),
        detail="require finite CBM/VBM and gap_ev>0",
    )

    n0 = len(out)
    gap_from_edges = out.cbm_ev - out.vbm_ev
    ok = (gap_from_edges - out.gap_ev).abs() <= config.edge_gap_consistency
    out = out.loc[ok].copy()
    audit.add_rule(
        "edge_gap_consistency",
        n_before=n0, n_after=len(out),
        detail=f"|(CBM−VBM)−Eg|≤{config.edge_gap_consistency} eV",
        threshold={"edge_gap_consistency": config.edge_gap_consistency},
    )

    n0 = len(out)
    if "delta_cbm" in out.columns:
        out = out.loc[out.delta_cbm.abs() <= config.max_abs_delta_cbm].copy()
        audit.add_rule(
            "delta_cbm_outlier",
            n_before=n0, n_after=len(out),
            detail=f"|δCBM|=|CBM−CBM_ButlerGinley|≤{config.max_abs_delta_cbm} eV",
            threshold={"max_abs_delta_cbm": config.max_abs_delta_cbm},
        )

    audit.n_edge_after = int(len(out))
    return out, audit
