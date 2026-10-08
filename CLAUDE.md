# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MaterialStack screens band alignment of device stacks: per layer the band gap (Eg) and band edges vs vacuum (VBM/CBM), per interface the band offsets and junction type (I/II/III) with a confidence. It is a B.Tech project, so simplicity matters: a few CSV files, **one** model, three scripts. `README.md` is the overview. Everything that is not code or data lives outside the repo in `../MaterialStack_extras/`: `literature_advay/` (source-credibility audit, paper notes, dated findings log; keep it updated when you learn something about the data or accuracy), `CHANGELOG.md` and the user's notes. Keep this folder to code, data and outputs only. The user only wants peer-reviewed sources for data.

## Commands

A virtualenv lives in `.venv/` (`pip install -r requirements.txt`).

```bash
python build_data.py      # data/raw (downloads if missing) -> data/band_gaps.csv, band_gaps_rejected.csv, dft_gaps.csv, band_edges.csv
python train.py           # -> models/gap_model.joblib
python validate.py        # -> results/metrics.csv, metrics.json, gap_cv_predictions.csv, junction_details.csv
python -m materialstack predict TiO2 MAPbI3 Spiro-OMeTAD
python -m materialstack serve [--port 8000] [--reload]
.venv/bin/python -m pytest -q   # tests/; two tests need the built data and model
```

Frontend (`frontend/`): `npm run dev` (Vite on :5173, proxies `/api` to :8000), `npm run build` (served by `serve`), `npm run lint`. Headless UI check: `../MaterialStack_extras/literature_advay/scripts/ui_screenshots.py` (server on :8001). Log every pipeline change in `../MaterialStack_extras/CHANGELOG.md`; report validation numbers from `results/metrics.csv`.

## Architecture

- **Data** (CSV only, no database). Hand-curated, cited: `data/curated/measured_band_edges.csv`, `measured_band_offsets.csv`, `aliases.csv` (the only data files edited by hand). Built by `build_data.py`: `band_gaps.csv` (one measured gap per material: Zhuo 2018 + Borlido 2019 + curated; Borlido/curated preferred, else consensus), `band_gaps_rejected.csv` (rules R1–R6 with reasons), `dft_gaps.csv` (median GGA and hybrid gap per formula from JARVIS + SNUMAT, JARVIS polymorphs within 50 meV/atom of the lowest only), `band_edges.csv` (one row per material and basis: `measured` or `hybrid-DFT surfaces` from Kiyohara 2024). Validation reference: `data/multilayer_gold_standard.csv` (50 device stacks). Formula key everywhere = pymatgen reduced formula (`chem.formula_key`); organics use `organic:<name>`.
- **Model** (`materialstack/model.py`): one LightGBM regressor on log(1+Eg), semiconductors only, features = matminer Magpie composition set + Mulliken χ + n_elements + `dft_gap_gga`/`dft_gap_hybrid` (NaN when unknown, no imputation). `train.py` fits it on everything; `validate.py` scores it with `GroupKFold(5)` by element set.
- **Prediction** (`materialstack/predict.py`): alias → formula; Eg measured → metal (only metallic elements) → ML; VBM measured → 0.8·hybrid surface + 0.2·Butler–Ginley → Butler–Ginley (−χ − Eg/2); CBM measured or VBM + Eg; organics measured only. Junction: measured interface offset (with its CBO) → vacuum alignment; type from the four edges (`junction_type`), confidence by Monte Carlo with σ constants (`SIGMA_GAP`, `SIGMA_VBM`) taken from `validate.py` results. `use_measured_edges=False` hides curated edges/offsets (used by validation as a hold-out).
- **Sign conventions:** vbo = VBM(top) − VBM(bottom); cbo = CBM(bottom) − CBM(top). Offsets file stores substrate − film.
- **Serving** (`api.py`): `/api/health`, `/api/metrics` (results/metrics.json), `POST /api/predict`; serves `frontend/dist`. UI is in `frontend/src/App.tsx`, types in `api.ts`.

Constraints: LightGBM only (no second model, no XGBoost/automatminer); junction type stays physics-derived, never learned; CV always grouped by element set.
