# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MaterialStack predicts band gap (Eg) and vacuum band edges (CBM/VBM) for each layer of a solar/optoelectronic stack, then derives heterojunction type (I/II/III) from the band offsets. Python package `materialstack` (pipeline + FastAPI) plus a React/Vite frontend in `frontend/`. `README.md` has the full source table and schema; `MaterialStack_EVALUATION.md` is a presentation/evaluation guide (partly in Hinglish). `literature_advay/` holds the data-source credibility audit, candidate datasets, paper notes and a dated findings log; keep it updated when you learn something about the data or model accuracy.

## Commands

A virtualenv lives in `.venv/` (`pip install -r requirements.txt`). Everything goes through the CLI (`materialstack/cli.py`):

```bash
python -m materialstack build-db                     # download sources -> data/materials_db.sqlite (~1-2 GB)
python -m materialstack build-db --only jarvis_surfacedb literature_csv   # re-ingest selected sources
python -m materialstack stats | lookup TiO2
python -m materialstack featurize --n-jobs 4         # composition features -> features_composition
python -m materialstack featurize-structures --n-jobs 7
python -m materialstack train-proxies                # DFT-gap proxy models (needs featurize-structures)
python -m materialstack train [--no-xgb-baseline] [--no-structure]
python -m materialstack predict TiO2 MAPbI3          # stack -> junctions
python -m materialstack serve [--reload]             # API + built UI on :8000
```

Frontend (`frontend/`): `npm run dev` (Vite on :5173, proxies `/api` to :8000), `npm run build` (`tsc -b && vite build` -> `frontend/dist`, served by FastAPI), `npm run lint` (oxlint).

There is no test suite. Verify changes by running `predict`/`lookup` against the DB, or by checking CV metrics written to `models/metrics.json` after `train`.

The SQLite DB, `data/cache/`, and `models/*.joblib` are gitignored generated artifacts; most commands require `build-db` (and for prediction, `train`) to have been run first. `MATERIALSTACK_ROOT` env var overrides the root for data/models paths (`config.py`).

## Architecture

Pipeline stages, each persisting to `data/materials_db.sqlite` or `models/`:

1. **Ingest** (`build_db.py`, `sources/`): each loader in `sources/__init__.py:LOADERS` is a generator yielding `("record" | "interface" | "alias", dict)` tuples; `db.Writer` normalizes formulas (`chem.py`), creates `materials`/`structures`/`records` rows, and logs to `build_log`. Re-ingesting a source deletes its prior rows first. To add a source: write a loader, register it in `LOADERS`, and add its key to `DEFAULT_SOURCES`/`HEAVY_SOURCES` in `config.py`.
2. **Trust ranking**: `config.METHOD_RANK` (experiment/literature=1 … PBE=6, ml_prediction=9) drives the `materials_best` SQL view in `db.py`, which picks the best-ranked gap and edges per material. New method tags must be added there.
3. **Features** (`features.py`: Magpie/matminer composition; `structure.py`: cell-choice-invariant structure descriptors `sf_*` + DFT-proxy models) are cached in `features_composition`, `features_structure`, `structure_proxy` tables.
4. **Training** (`models.py`, `clean.py`): `TwoStageGapModel` (LightGBM metal classifier + Eg regressor) and `EdgeCorrectionModel` (predicts δCBM over the Butler–Ginley electronegativity estimate; VBM = CBM − Eg). CV is always `GroupKFold` grouped by element set to avoid leakage across substitutions. `clean.py` filters only the training frames (never the DB) and writes `models/train_clean_audit.json`. Borlido materials are held out of Eg training and scored separately. A structure-aware model is trained alongside the composition-only one; missing structure/DFT features stay NaN for LightGBM to route.
5. **Prediction** (`predict.py`): lookup-first. `resolve_layer` resolves alias -> formula -> `materials_best`; uses DB Eg if its rank ≤ `max_lookup_rank` (default 2), otherwise ML (structure-aware model if the DB has a structure/DFT gaps, else composition-only on-the-fly Magpie features). Polymorph policy `ground_state` vs `mean`. `classify_junction` is deterministic from CBO/VBO (not ML) and flags UNCERTAIN near ~0.3 eV boundaries or when ML edges are involved.
6. **Serving** (`api.py`): FastAPI endpoints `/api/health`, `/api/stats`, `/api/metrics` (reads `models/metrics.json`), `POST /api/predict`; serves `frontend/dist` as an SPA when built. Frontend API client is `frontend/src/api.ts`; UI is essentially all in `App.tsx`.

Project constraints (from README/evaluation doc): LightGBM is the primary model (XGBoost only as a baseline); `automatminer` is intentionally not used; junction type must remain physics-derived, not learned.
