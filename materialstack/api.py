"""MaterialStack HTTP API — lookup-first predict + performance metrics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from materialstack.config import DB_PATH, MODELS_DIR, ROOT
from materialstack.db import connect, stats
from materialstack.predict import resolve_stack

FRONTEND_DIST = ROOT / "frontend" / "dist"
METRICS_PATH = MODELS_DIR / "metrics.json"
CLEAN_AUDIT_PATH = MODELS_DIR / "train_clean_audit.json"

app = FastAPI(title="MaterialStack", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    materials: list[str] = Field(..., min_length=1)
    max_lookup_rank: int = 2
    polymorph: Literal["ground_state", "mean"] | None = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "product": "MaterialStack",
        "db_exists": DB_PATH.exists(),
        "eg_model": (MODELS_DIR / "eg_model.joblib").exists(),
        "eg_structure_model": (MODELS_DIR / "eg_structure_model.joblib").exists(),
        "edge_model": (MODELS_DIR / "edge_model.joblib").exists(),
        "metrics": METRICS_PATH.exists(),
    }


@app.get("/api/stats")
def api_stats() -> dict[str, Any]:
    if not DB_PATH.exists():
        raise HTTPException(503, "materials database not found")
    con = connect(DB_PATH)
    s = stats(con)
    con.close()
    return {
        "materials": s["materials"],
        "records": s["records"],
        "structures": s["structures"],
        "interfaces": s["interfaces"],
        "aliases": s["aliases"],
        "by_family": s["by_family"].to_dict(orient="records"),
    }


@app.get("/api/metrics")
def api_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise HTTPException(404, "metrics.json not found — run training first")
    data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    audit = None
    if CLEAN_AUDIT_PATH.exists():
        audit = json.loads(CLEAN_AUDIT_PATH.read_text(encoding="utf-8"))
    eg = data["eg_model"]["lightgbm"]
    edge = data["edge_model"]["lightgbm"]
    xgb_eg = data["eg_model"].get("xgboost_baseline")
    xgb_edge = data["edge_model"].get("xgboost_baseline")
    return {
        "product": "MaterialStack",
        "primary_backend": data.get("primary_backend"),
        "decision": data.get("decision"),
        "eg_train_methods": data.get("eg_train_methods"),
        "n_eg_train": data.get("n_eg_train"),
        "seconds": data.get("seconds"),
        "eg": {
            "mae_nonmetal": eg.get("gap_mae_nonmetal"),
            "rmse_nonmetal": eg.get("gap_rmse_nonmetal"),
            "r2_nonmetal": eg.get("gap_r2_nonmetal"),
            "mae_all": eg.get("gap_mae_all"),
            "metal_accuracy": eg.get("metal_accuracy"),
            "metal_f1": eg.get("metal_f1"),
            "n_train": eg.get("n_train_materials"),
            "n_metals": eg.get("n_metals"),
            "n_nonmetals": eg.get("n_nonmetals"),
            "n_splits": eg.get("n_splits"),
            "top_features": eg.get("top_features", [])[:12],
        },
        "eg_xgb_baseline": {
            "mae_nonmetal": xgb_eg.get("gap_mae_nonmetal") if xgb_eg else None,
            "metal_accuracy": xgb_eg.get("metal_accuracy") if xgb_eg else None,
        },
        "edge": {
            "cbm_mae": edge.get("cbm_mae"),
            "vbm_mae": edge.get("vbm_mae"),
            "delta_mae": edge.get("delta_mae"),
            "n_train": edge.get("n_train_materials"),
            "sources": edge.get("sources"),
            "top_features": edge.get("top_features", [])[:8],
        },
        "edge_xgb_baseline": {
            "cbm_mae": xgb_edge.get("cbm_mae") if xgb_edge else None,
        },
        "borlido_holdout": data["eg_model"].get("borlido_holdout"),
        "structure_hybrid": _structure_summary(data["eg_model"].get("structure_hybrid")),
        "clean_audit": audit,
        "clean_summary": data.get("clean_summary"),
    }


def _structure_summary(h: dict[str, Any] | None) -> dict[str, Any] | None:
    if not h:
        return None
    ablation = [
        {"label": label, "all": m["all"], "with_structure": m["with_structure"]}
        for label, m in h["ablation"].items()
    ]
    ablation.append({"label": "final, mean over polymorphs", "all": h["final_mean_polymorph"]["all"],
                     "with_structure": h["final_mean_polymorph"]["with_structure"]})
    return {
        "recommended_policy": h.get("recommended_policy", "ground_state"),
        "fallback_without_structure": h.get("fallback_without_structure"),
        "coverage": h.get("coverage"),
        "coverage_dft": h.get("coverage_dft"),
        "n_with_structure": h.get("n_train_with_structure"),
        "n_with_dft": h.get("n_train_with_dft"),
        "n_features": h.get("n_features"),
        "ablation": ablation,
        "system_composition_only": h["system"]["composition_only"],
        "system_structure_aware": h["system"]["structure_aware"],
        "borlido_holdout": h.get("borlido_holdout"),
        "top_features": h.get("top_features", [])[:12],
    }


@app.post("/api/predict")
def api_predict(body: PredictRequest) -> dict[str, Any]:
    names = [n.strip() for n in body.materials if n.strip()]
    if not names:
        raise HTTPException(400, "provide at least one material")
    if not DB_PATH.exists():
        raise HTTPException(503, "materials database not found")
    try:
        return resolve_stack(names, max_lookup_rank=body.max_lookup_rank, polymorph=body.polymorph)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str = ""):
        index = FRONTEND_DIST / "index.html"
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)
