"""Web API: predict a device stack, and show the validation metrics. Also serves the built web UI."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from materialstack.config import BAND_EDGES, BAND_GAPS, MODEL, RESULTS, ROOT
from materialstack.predict import predict_stack

FRONTEND_DIST = ROOT / "frontend" / "dist"
METRICS = RESULTS / "metrics.json"

app = FastAPI(title="MaterialStack")


class PredictRequest(BaseModel):
    materials: list[str] = Field(..., min_length=1, max_length=20)   # a device stack, not a batch job


@app.get("/api/health")
def health() -> dict[str, Any]:
    edges = pd.read_csv(BAND_EDGES) if BAND_EDGES.exists() else pd.DataFrame(columns=["basis"])
    return {"ok": BAND_GAPS.exists() and MODEL.exists(),
            "measured_gaps": len(pd.read_csv(BAND_GAPS)) if BAND_GAPS.exists() else 0,
            "measured_edges": int((edges.basis == "measured").sum()),
            "surface_edges": int((edges.basis == "hybrid-DFT surfaces").sum()),
            "model": MODEL.exists()}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    if not METRICS.exists():
        raise HTTPException(404, "results/metrics.json not found: run python validate.py")
    return json.loads(METRICS.read_text(encoding="utf-8"))


@app.post("/api/predict")
def predict(body: PredictRequest) -> dict[str, Any]:
    names = [n.strip() for n in body.materials if n.strip()]
    if not names:
        raise HTTPException(400, "provide at least one material")
    if not BAND_GAPS.exists():
        raise HTTPException(503, "data not built: run python build_data.py")
    try:
        return predict_stack(names)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


if FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str = ""):
        candidate = FRONTEND_DIST / full_path
        return FileResponse(candidate if full_path and candidate.is_file() else FRONTEND_DIST / "index.html")
