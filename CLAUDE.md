# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MaterialStack screens band alignment of device stacks: per layer the band gap (Eg) and band edges vs vacuum (VBM/CBM), per interface the band offsets and junction type (I/II/III) with a confidence. It is a B.Tech project, so simplicity matters: a few CSV files, **one** model, three scripts. `README.md` is the overview. Everything that is not code or data lives outside the repo in `../MaterialStack_extras/`: `literature_advay/` (source-credibility audit, paper notes, dated findings log; keep it updated when you learn something about the data or accuracy), `CHANGELOG.md` and the user's notes. Keep this folder to code, data and outputs only. The user only wants peer-reviewed sources for data.

## Commands

A virtualenv lives in `.venv/` (`pip install -r requirements.txt`; versions are pinned to the ones that produced `results/metrics.csv`, which needs Python 3.12+).

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

- **Data** (CSV only, no database). Hand-curated, cited: `data/curated/measured_band_edges.csv`, `measured_band_offsets.csv`, `aliases.csv` (the only data files edited by hand). Built by `build_data.py`: `band_gaps.csv` (one measured gap per material: Zhuo 2018 + Borlido 2019 + curated; Borlido/curated preferred, else consensus), `band_gaps_rejected.csv` (rules R1–R6 with reasons; R6 = metal-vs-insulator conflict with hybrid DFT only), `dft_gaps.csv` (median GGA and hybrid gap per formula from JARVIS + SNUMAT, JARVIS polymorphs within 50 meV/atom of the lowest only), `band_edges.csv` (one row per material and basis: `measured` or `hybrid-DFT surfaces` from Kiyohara 2024). Validation reference: `data/multilayer_gold_standard.csv` (50 device stacks). Formula key everywhere = pymatgen reduced formula (`chem.formula_key`); organics use `organic:<name>`.
- **Model** (`materialstack/model.py`): one LightGBM regressor on log(1+Eg), semiconductors only, features = matminer Magpie composition set + Mulliken χ + n_elements + `dft_gap_gga`/`dft_gap_hybrid` (NaN when unknown, no imputation). `train.py` fits it on everything; `validate.py` scores it with 5-fold CV whose fold is fixed per element system (md5 hash), so adding data never reshuffles folds.
- **Prediction** (`materialstack/predict.py`): alias → formula; Eg measured → metal (only metallic elements) → ML; VBM measured → 0.8·hybrid surface + 0.2·Butler–Ginley → Butler–Ginley (−χ − Eg/2); hybrid perovskites (MA/FA) are described by their Cs analogue in both the gap features and χ (`chem.inorganic_surrogate`); CBM measured or VBM + Eg; organics measured only. Lowercase formulas resolve when unambiguous (`tio2` → TiO2, measured data first). Junction: measured interface offset (with its CBO) → vacuum alignment, σ for both ΔEv and ΔEc; type from the four edges (`junction_type`), confidence by Monte Carlo with σ constants (`SIGMA_GAP`, `SIGMA_VBM`) taken from `validate.py` results. `use_measured_edges=False` hides curated edges/offsets (used by validation as a hold-out).
- **Sign conventions:** vbo = VBM(top) − VBM(bottom); cbo = CBM(bottom) − CBM(top). Offsets file stores substrate − film.
- **Serving** (`api.py`): `/api/health`, `/api/metrics` (results/metrics.json), `POST /api/predict`; serves `frontend/dist`. UI: `frontend/src/App.tsx` (top tab bar + hash routes `#/predict`, `#/validation`, `#/method`, `#/guide`), one file per page in `frontend/src/pages/`, `BandDiagram.tsx`, helpers in `format.ts`, API types in `api.ts`. The Guide page holds the how-to and the tool's notes (limitations); keep them in step with the method.

Constraints: LightGBM only (no second model, no XGBoost/automatminer); junction type stays physics-derived, never learned; CV always grouped by element set.

Validation is a fixed protocol (`validate.py` docstring): do not edit it when data or model change; every test set is a rule over the current data, reference offsets are rows in `data/curated/validation_band_offsets.csv`. Judge a change on the rows with `primary` = True (each has a 95 % bootstrap interval) via `git diff results/metrics.csv`; sections A and B and the metric names the UI reads must stay as they are.
