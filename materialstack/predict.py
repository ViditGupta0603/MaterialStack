"""Lookup-first layer property resolution for MaterialStack.

Prefer method-ranked DB values (experiment / literature / HSE06 by default).
Fall back to Magpie + LightGBM only when the DB has no trusted label.
Junction type is computed deterministically from band edges (not learned).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from materialstack.chem import butler_ginley_edges, clean_material_name, mulliken_chi, try_composition
from materialstack.config import DB_PATH, METHOD_RANK, MODELS_DIR
from materialstack.db import connect, lookup, resolve_alias
from materialstack.features import feature_row, featurize_formula
from materialstack.models import EG_MODEL_PATH, EDGE_MODEL_PATH, load_edge_model, load_eg_model, predict_edges, predict_eg

log = logging.getLogger("materialstack.predict")

# Use DB gap if best method rank is <= this (1=experiment/literature, 2=HSE06).
DEFAULT_MAX_LOOKUP_RANK = 2
JUNCTION_BOUNDARY_EV = 0.3


@dataclass
class LayerResult:
    query: str
    formula: str | None
    material_id: int | None
    gap_ev: float | None
    cbm_ev: float | None
    vbm_ev: float | None
    gap_source: str  # lookup:<method>:<source> | model | butler_ginley | missing
    edge_source: str
    gap_method: str | None = None
    edge_method: str | None = None
    p_metal: float | None = None
    chi: float | None = None
    trusted_gap: bool = False
    trusted_edges: bool = False
    notes: list[str] | None = None
    polymorph_policy: str | None = None
    structures_used: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = self.notes or []
        d["structures_used"] = self.structures_used or []
        return d


def method_rank(method: str | None) -> int:
    if not method:
        return 99
    return int(METHOD_RANK.get(method, 8))


def _pick_best_row(best: pd.DataFrame) -> pd.Series | None:
    if best.empty:
        return None
    df = best.copy()
    df["_rank"] = df["best_gap_method"].map(method_rank)
    df = df.sort_values(["_rank", "n_records"], ascending=[True, False], na_position="last")
    return df.iloc[0]


def _load_models(models_dir: Path = MODELS_DIR):
    eg_path = Path(models_dir) / "eg_model.joblib"
    edge_path = Path(models_dir) / "edge_model.joblib"
    eg = load_eg_model(eg_path) if eg_path.exists() else None
    edge = load_edge_model(edge_path) if edge_path.exists() else None
    return eg, edge


def _load_structure_models(models_dir: Path = MODELS_DIR):
    import joblib

    from materialstack.structure import load_proxy_bundles

    path = Path(models_dir) / "eg_structure_model.joblib"
    if not path.exists():
        return None, {}
    return joblib.load(path), load_proxy_bundles(models_dir)


def _predict_eg_structure_aware(con, material_id: int | None, feats: pd.Series, policy: str,
                                struct_bundle, proxy_bundles) -> tuple[float, float, list[dict[str, Any]], str] | None:
    """Structure-aware Eg. Returns (gap, p_metal, structures_used, mode) or None → composition model.

    mode is 'structure' (DB crystal structure(s)), 'dft_only' (stored DFT gaps, no structure) or
    'composition_fields' (nothing known; hybrid model chosen as fallback at training time).
    """
    import math

    from materialstack.structure import material_dft_features, structure_rows

    rows = structure_rows(con, material_id, policy, proxy_bundles, feats) if material_id is not None else []
    if not rows:
        dft = material_dft_features(con, material_id) if material_id is not None else {}
        has_dft = any(not math.isnan(v) for v in dft.values())
        if not has_dft and struct_bundle.get("fallback_without_structure", "composition") != "hybrid":
            return None
        full = pd.concat([feats, pd.Series(dft, dtype=float)])
        full = full[~full.index.duplicated(keep="last")]
        pred = predict_eg(struct_bundle, full)
        return float(pred["gap_ev"]), float(pred["p_metal"]), [], "dft_only" if has_dft else "composition_fields"
    used = []
    for s in rows:
        full = pd.concat([feats, pd.Series(s["features"], dtype=float)])
        full = full[~full.index.duplicated(keep="last")]
        pred = predict_eg(struct_bundle, full)
        used.append({k: s[k] for k in ("structure_id", "source", "space_group", "e_rel")} |
                    {"gap_ev": float(pred["gap_ev"]), "p_metal": float(pred["p_metal"])})
    gap = float(sum(u["gap_ev"] for u in used) / len(used))
    p_metal = float(sum(u["p_metal"] for u in used) / len(used))
    return gap, p_metal, used, "structure"


def _features_for_material(con, material_id: int | None, formula: str,
                           family: str | None = None) -> pd.Series | None:
    if material_id is not None:
        row = feature_row(con, int(material_id))
        if row is not None:
            return row
    try:
        return featurize_formula(formula, family=family)
    except Exception as exc:
        log.warning("featurize failed for %s: %s", formula, exc)
        return None


def resolve_layer(
    name: str,
    *,
    db_path: Path = DB_PATH,
    models_dir: Path = MODELS_DIR,
    max_lookup_rank: int = DEFAULT_MAX_LOOKUP_RANK,
    con=None,
    eg_bundle=None,
    edge_bundle=None,
    polymorph: str | None = None,
    struct_bundle=None,
    proxy_bundles=None,
) -> LayerResult:
    """Resolve Eg / CBM / VBM for one layer: DB lookup first, then ML.

    The ML band gap uses the structure-aware model when the DB holds a bulk crystal structure
    for the material (``polymorph`` = ground_state | mean; None → trained recommendation),
    otherwise the composition-only model.
    """
    own_con = con is None
    if own_con:
        con = connect(db_path)
    notes: list[str] = []

    alias = resolve_alias(con, name)
    query = name
    if alias:
        notes.append(f"alias '{name}' → {alias.get('formula') or alias.get('note')}")
        if not alias.get("formula"):
            if own_con:
                con.close()
            return LayerResult(
                query=query, formula=None, material_id=None,
                gap_ev=None, cbm_ev=None, vbm_ev=None,
                gap_source="missing", edge_source="missing",
                notes=notes + ["organic/alias without formula — add literature CSV"],
            )
        name = alias["formula"]

    cn = clean_material_name(name)
    formula = cn.formula_clean
    if not formula:
        if own_con:
            con.close()
        return LayerResult(
            query=query, formula=None, material_id=None,
            gap_ev=None, cbm_ev=None, vbm_ev=None,
            gap_source="missing", edge_source="missing",
            notes=notes + ["formula could not be parsed"],
        )

    hit = lookup(con, formula)
    best = _pick_best_row(hit["best"])
    material_id = int(best.material_id) if best is not None else None
    family = None
    chi = None
    if not hit["materials"].empty:
        m0 = hit["materials"].iloc[0]
        family = m0.get("family")
        chi = m0.get("mulliken_chi")
        if material_id is None:
            material_id = int(m0.material_id)

    if chi is None:
        comp = try_composition(formula)
        if comp is not None:
            chi = mulliken_chi(comp)

    gap_ev = None
    gap_source = "missing"
    gap_method = None
    trusted_gap = False

    if best is not None and pd.notna(best.get("best_gap")):
        rank = method_rank(best.get("best_gap_method"))
        if rank <= max_lookup_rank:
            gap_ev = float(best.best_gap)
            gap_method = str(best.best_gap_method)
            gap_source = f"lookup:{gap_method}:{best.best_gap_source}"
            trusted_gap = True
        else:
            notes.append(
                f"DB has gap={best.best_gap:.3f} eV via {best.best_gap_method} "
                f"(rank {rank} > max_lookup_rank {max_lookup_rank}); using ML fallback"
            )

    cbm_ev = vbm_ev = None
    edge_source = "missing"
    edge_method = None
    trusted_edges = False

    if best is not None and pd.notna(best.get("best_cbm")) and pd.notna(best.get("best_vbm")):
        erank = method_rank(best.get("best_edge_method"))
        # Accept any vacuum edge in DB for edges (scarce); still prefer low rank.
        cbm_ev = float(best.best_cbm)
        vbm_ev = float(best.best_vbm)
        edge_method = str(best.best_edge_method) if pd.notna(best.get("best_edge_method")) else None
        edge_source = f"lookup:{edge_method}:{best.best_edge_source}"
        trusted_edges = erank <= max(max_lookup_rank, 4)  # GLLB-SC edges (Castelli) ok
        if gap_ev is None and trusted_edges:
            # Derive gap from edges if no trusted gap yet.
            gap_ev = abs(cbm_ev - vbm_ev)
            gap_source = f"derived_from_edges:{edge_source}"
            notes.append("Eg derived from looked-up CBM−VBM")

    p_metal = None
    policy_used = None
    structures_used = None
    need_ml_gap = gap_ev is None
    need_ml_edges = cbm_ev is None or vbm_ev is None

    if need_ml_gap or need_ml_edges:
        if eg_bundle is None or edge_bundle is None:
            eg_bundle, edge_bundle = _load_models(models_dir)
        feats = _features_for_material(con, material_id, formula, family=family)

        if need_ml_gap:
            if struct_bundle is None:
                struct_bundle, proxy_bundles = _load_structure_models(models_dir)
            hybrid = None
            if struct_bundle is not None and feats is not None:
                policy_used = polymorph or struct_bundle.get("recommended_policy", "ground_state")
                hybrid = _predict_eg_structure_aware(con, material_id, feats, policy_used,
                                                     struct_bundle, proxy_bundles or {})
            if hybrid is not None:
                gap_ev, p_metal, structures_used, mode = hybrid
                gap_source = "model:lightgbm_structure"
                gap_method = "ml_prediction"
                trusted_gap = False
                if mode != "structure":
                    policy_used = None
                    notes.append("no crystal structure in DB — structure-aware model using stored DFT gaps"
                                 if mode == "dft_only" else
                                 "not in DB — structure-aware model with composition features only")
                elif len(structures_used) == 1:
                    s = structures_used[0]
                    hull = f", ΔE={s['e_rel']:.3f} eV/atom vs lowest polymorph" if s["e_rel"] is not None else ""
                    notes.append(f"structure-aware ML ({policy_used}): {s['source']} SG {s['space_group']}{hull}")
                else:
                    parts = ", ".join(f"SG {s['space_group']}: {s['gap_ev']:.2f} eV" for s in structures_used)
                    notes.append(f"structure-aware ML, mean over {len(structures_used)} polymorphs ({parts})")
            elif eg_bundle is None or feats is None:
                notes.append("ML Eg unavailable (missing model or features)")
            else:
                pred = predict_eg(eg_bundle, feats)
                gap_ev = float(pred["gap_ev"])
                p_metal = float(pred["p_metal"])
                gap_source = "model:lightgbm"
                gap_method = "ml_prediction"
                trusted_gap = False
                policy_used = None
                if struct_bundle is not None:
                    notes.append("no crystal structure or DFT data in DB — composition-only model")

        if need_ml_edges and gap_ev is not None and chi is not None:
            if edge_bundle is not None and feats is not None:
                pred = predict_edges(edge_bundle, feats, gap_ev, float(chi))
                cbm_ev = float(pred["cbm_ev"])
                vbm_ev = float(pred["vbm_ev"])
                edge_source = "model:lightgbm_delta_cbm"
                edge_method = "ml_prediction"
                trusted_edges = False
            else:
                cbm_bg, vbm_bg = butler_ginley_edges(float(chi), float(gap_ev))
                cbm_ev, vbm_ev = float(cbm_bg), float(vbm_bg)
                edge_source = "butler_ginley"
                edge_method = "Butler-Ginley"
                trusted_edges = False
                notes.append("edge model missing — Butler–Ginley only")
        elif need_ml_edges and chi is None:
            notes.append("cannot place edges: missing Mulliken χ")

    if own_con:
        con.close()

    # Consistency: if we have gap + CBM, enforce VBM = CBM − Eg when edges came from ML/BG
    if gap_ev is not None and cbm_ev is not None and edge_source.startswith(("model", "butler")):
        vbm_ev = float(cbm_ev) - float(gap_ev)

    return LayerResult(
        query=query,
        formula=formula,
        material_id=material_id,
        gap_ev=gap_ev,
        cbm_ev=cbm_ev,
        vbm_ev=vbm_ev,
        gap_source=gap_source,
        edge_source=edge_source,
        gap_method=gap_method,
        edge_method=edge_method,
        p_metal=p_metal,
        chi=float(chi) if chi is not None else None,
        trusted_gap=trusted_gap,
        trusted_edges=trusted_edges,
        notes=notes,
        polymorph_policy=policy_used,
        structures_used=structures_used,
    )


def classify_junction(top: LayerResult, bottom: LayerResult) -> dict[str, Any]:
    """Deterministic Type I / II / III from vacuum-aligned edges (top over bottom)."""
    if None in (top.cbm_ev, top.vbm_ev, bottom.cbm_ev, bottom.vbm_ev):
        return {"type": None, "uncertain": True, "reason": "missing band edges"}

    # Convention: more negative = deeper vs vacuum
    cbo = float(bottom.cbm_ev) - float(top.cbm_ev)   # >0: bottom CBM lower (electrons to bottom)
    vbo = float(top.vbm_ev) - float(bottom.vbm_ev)     # >0: top VBM higher (holes to top)

    # Band extents
    t_c, t_v = float(top.cbm_ev), float(top.vbm_ev)
    b_c, b_v = float(bottom.cbm_ev), float(bottom.vbm_ev)

    # Type III broken-gap: gaps do not overlap in energy
    if t_v > b_c or b_v > t_c:
        jtype = "III"
    # Type I straddling: one gap fully contains the other
    elif (t_c >= b_c and t_v <= b_v) or (b_c >= t_c and b_v <= t_v):
        jtype = "I"
    else:
        jtype = "II"

    margin = min(abs(cbo), abs(vbo), abs(t_c - b_c), abs(t_v - b_v))
    either_ml = (not top.trusted_gap) or (not bottom.trusted_gap) or \
                (not top.trusted_edges) or (not bottom.trusted_edges)
    uncertain = margin < JUNCTION_BOUNDARY_EV or (either_ml and margin < 2 * JUNCTION_BOUNDARY_EV)

    return {
        "type": jtype,
        "cbo_ev": cbo,
        "vbo_ev": vbo,
        "margin_ev": margin,
        "uncertain": uncertain,
        "reason": (
            f"offset margin {margin:.3f} eV near {JUNCTION_BOUNDARY_EV} eV boundary"
            if uncertain and margin < JUNCTION_BOUNDARY_EV
            else ("ML-involved edges near boundary" if uncertain else "ok")
        ),
    }


def resolve_stack(
    names: list[str],
    *,
    db_path: Path = DB_PATH,
    models_dir: Path = MODELS_DIR,
    max_lookup_rank: int = DEFAULT_MAX_LOOKUP_RANK,
    polymorph: str | None = None,
) -> dict[str, Any]:
    con = connect(db_path)
    eg_bundle, edge_bundle = _load_models(models_dir)
    struct_bundle, proxy_bundles = _load_structure_models(models_dir)
    layers = [
        resolve_layer(n, db_path=db_path, models_dir=models_dir, max_lookup_rank=max_lookup_rank,
                      con=con, eg_bundle=eg_bundle, edge_bundle=edge_bundle, polymorph=polymorph,
                      struct_bundle=struct_bundle, proxy_bundles=proxy_bundles)
        for n in names
    ]
    con.close()
    junctions = []
    for i in range(len(layers) - 1):
        junctions.append({
            "interface": f"{layers[i].formula or layers[i].query} | {layers[i+1].formula or layers[i+1].query}",
            **classify_junction(layers[i], layers[i + 1]),
        })
    return {
        "max_lookup_rank": max_lookup_rank,
        "polymorph": polymorph or (struct_bundle or {}).get("recommended_policy", "ground_state"),
        "layers": [L.to_dict() for L in layers],
        "junctions": junctions,
    }
