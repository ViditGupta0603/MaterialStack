# Findings log

Newest first. Each entry says how the number was obtained.

---

## 2026-10-03: Data source and edge-model audit

Setup: DB as built on 2026-10-02; models retrained with `train --no-xgb-baseline`
(Eg MAE 0.451 eV composition-only, 0.419 eV structure-aware; edge CBM MAE reported 0.184 eV).
Scripts: `scripts/edge_analysis.py`, `scripts/source_credibility.py`, `scripts/surface_vacuum_check.py`.

### F1. JARVIS surface edges are stored without vacuum alignment (bug)
- `materialstack/sources/jarvis_sources.py` (`surfacedb`) stores `surf_vbm`/`surf_cbm` as vacuum-referenced, but these are on the slab's internal energy scale. InterMat defines IP = E_vac − VBM with E_vac = `avg_max`.
- App output before fix: Si VBM **+0.72 eV** (above vacuum), GaAs −1.97, ZnO −2.64, CdTe −2.47 (expected about −5.2, −5.5, −7.6, −5.8). Si/GaAs is classified **Type III**.
- `surf_vbm − avg_max` gives Si −4.9 to −5.4, GaAs(110) −5.7, GaN −6.2 to −7.5, CdTe about −6.6: physically sensible.
- About 120 of 607 surfaces have broken vacuum levels (`avg_max` up to 167 eV); filter with `avg_max < 12` and `2 < avg_max − efermi < 9` (485 survive).
- Stored work function (`−efermi`) has the same problem (Si shows −0.72 eV).
- Status: **open**.

### F2. Castelli band edges are the Butler–Ginley formula
- CBM − BG(χ, Eg): median |diff| 0.028 eV, 82% within 0.1 eV, corr 0.95 (18,928 rows; 96% metals).
- 683 of 809 edge-model training materials are Castelli, where Butler–Ginley alone scores 0.078 eV and the ML model 0.106 eV. The headline 0.18 eV edge MAE mostly measures reproduction of a formula.
- Castelli hypothetical "InInO3"/"GaGaO3" supply the edges shown for In2O3 and Ga2O3.
- Status: **open**: drop Castelli as edge labels and lookups.

### F3. Edge model on real DFT surfaces is no better than a constant
- On jarvis_surfacedb rows (n=126, wrong energy scale per F1): ML 0.608 eV, constant-δ baseline 0.544 eV, Butler–Ginley alone 3.93 eV.
- Re-check after fixing F1.

### F4. Train/inference gap mismatch
- Edge model is trained and scored using the label's own DFT gap (OptB88vdW / GLLB-SC). At prediction time it gets an experimental/HSE/ML gap; these differ by 1.18 eV on average (n=140).
- Scored the way the app uses it: CBM MAE 0.61, **VBM MAE 0.69 eV** (vs 0.29 eV with the label gap). Contaminated by F1.
- ~~`VBM = CBM − Eg` sends all of the gap opening to the VBM.~~ **Corrected 2026-10-03:** wrong. The baseline CBM = −χ + Eg/2 already splits a gap change symmetrically (+½ to CBM, −½ to VBM), and the measured shifts agree (VBM −0.43 eV, CBM +0.32 eV). The real issue is the train/inference gap mismatch above, plus surface-scale labels (F1). The symmetric split itself is an assumption to test against Hinuma 2014.

### F5. Surface dependence is a large irreducible error for bulk-only features
- Same material, different surfaces: VBM range median 1.39 eV (75th pct 2.27) across 142 materials. Measured on uncorrected values; recompute after F1.

### F6. Method mixing in edge training
- `load_edge_frame` takes medians over all rows sharing the top source weight, mixing OptB88vdW and TBmBJ CBMs of the same material. Likely why only 126 of 322 surface materials survive cleaning. Unverified root cause.

### F7. Interface data unused; vacuum alignment vs direct DFT offsets
- 425 interfaces have numeric offsets (168 store a dict instead; currently dropped).
- Plane-matched vacuum alignment from surfaces vs interface offsets (n=371): corr 0.42, MAE 1.51 eV; predicting 0 gives 0.87 eV. The paper reports 0.45 eV vs experiment for vacuum alignment on 8 interfaces, so either our matching/sign handling is off or many interfaces have large interface dipoles. Needs follow-up with the paper's convention (positive ΔEv at A/B means VBM higher in B).

### F8. Source credibility (experimental and computed gaps)
- Experimental sources: `expt_gap` vs `expt_gap_kingsbury` 95% identical (duplicate); Borlido vs Zhuo MAE 0.22 eV (label noise floor).
- Computed vs experiment MAE: SNUMAT HSE06 0.62, JARVIS TBmBJ 0.81, PBE/OptB88vdW 1.0–1.15 (about −1 eV bias), Castelli GLLB-SC 1.96 and Wolverton PBE 1.90 (hypothetical structures).
- MAPbI3 gets 2.64 eV from `jarvis_halide_perovskites` HSE06 (experiment about 1.6 eV); hybrid perovskites should use experimental values only.

### F11. Dead code / leftover files audit (vulture ≥60% confidence + manual checks)
- Python package is lean (about 3.7k lines). Real leftovers only: unused import `Iterable` (`db.py:8`), unused `transaction()` helper (`db.py:150`), unused parameter `source_tag` in `jarvis_sources._bulk`, unread `CleanName.dopant_tag`/`parseable` fields, `borlido_expt` registered twice in `LOADERS`, a redundant first `GroupKFold` in `_grouped_cv_edge`.
- Leftover files: `catboost_info/` (CatBoost isn't used anywhere), `data/exports/` (no code produces or reads it), Vite template assets `frontend/public/favicon.svg`, `icons.svg`, `src/assets/hero.png` (not referenced), template `frontend/README.md`.
- Vulture false positives: FastAPI route functions (registered by decorator), `LayerResult.polymorph_policy` (read via `to_dict`), `cv_metrics_` (saved in model bundles).
- Not waste: XGBoost baseline (comparison for the report), `--heavy` sources (optional).

### F12. Approach assessment
- Keep: lookup-first; two-stage LightGBM for Eg (gradient boosting stays competitive on about 5k experimental labels, e.g. Ye 2025); DFT-proxy features (a form of multi-fidelity learning).
- Change: edge formulation. Predict VBM (−IP) per structure (later per surface) from vacuum-referenced surface data (fixed JARVIS + Kiyohara oxides), then CBM = VBM + best Eg. Replaces the correction-to-Butler–Ginley model trained on formula-derived labels.
- Add: per-layer uncertainty (quantile or conformal) → junction-type probabilities instead of a fixed 0.3 eV flag; later a pairwise correction trained on interface offsets (JARVIS direct interface calculations + experiment) to go beyond Anderson's rule.
- Lowest priority: switching to graph neural networks; gains are mainly on large DFT datasets, not our small experimental ones.

### F10. Inputs are formula-only; band edges never see structure
- CLI/API/UI accept only formula or alias strings (`api.py` `PredictRequest.materials: list[str]`); no CIF, MP or JARVIS ID.
- Lookup is per formula (polymorphs pooled). The ML gap uses DB-stored structures when present (code-chosen polymorph: `ground_state` or `mean`), otherwise composition only.
- Edge model (`load_edge_frame`) uses composition features + gap + Butler–Ginley only, although structure/surface drive much of edge variation (F5) and the state of the art uses structure (InterMat graph neural network 0.26 eV) or surface plane (Kiyohara 2024, about 0.22 eV).
- Options: accept structure input (CIF / database ID, optional Miller index); add the existing `features_structure` columns to the edge model; surface-resolved labels to learn orientation.

### F9. Coverage of device layers
- No experimental or correctly-computed edges for TiO2, SnO2, NiO, Cu2O, MoO3, WO3, halide perovskites, CIGS, CZTS, Sb2Se3. `literature_csv` is empty.
